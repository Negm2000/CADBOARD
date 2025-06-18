import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import ezdxf
import math
import numpy as np
import os
import traceback
from scipy.special import factorial

# Import the library for IDS cameras
# This library needs to be installed, e.g., via 'pip install pyueye'
try:
    from pyueye import ueye
except ImportError:
    # Create a mock ueye object if the library is not installed
    # This allows the GUI to run for development without a camera connected
    print("WARNING: pyueye library not found. IDS Camera functionality will be disabled.")
    class MockUeye:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                print(f"MOCK UEYE: Call to {name} with args={args} kwargs={kwargs}")
                if name == 'is_InitCamera':
                    return -1 # Simulate camera not found
                return 0
            return method
    ueye = MockUeye()


class InspectionApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("DIE-VIS: Visualizer & Inspector (Zernike Engine)")
        self.root.geometry("1200x850")

        # --- State Variables ---
        self.image_path = None
        self.dxf_path = None
        self.cv_image = None
        self.original_cv_image = None # Store original for resets
        self.cad_features = {'outline': np.array([]), 'holes': [], 'creases': []}
        
        # --- UI Colors and Styles ---
        self.colors = {
            "bg": "#2E2E2E", "fg": "#FFFFFF", "btn": "#4A4A4A",
            "btn_active": "#5A5A5A", "accent": "#007ACC"
        }
        self.root.configure(bg=self.colors["bg"])
        self.setup_styles()

        # --- UI Layout ---
        main_frame = tk.Frame(root, bg=self.colors["bg"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = tk.Frame(main_frame, bg=self.colors["bg"])
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.image_frame = tk.Frame(main_frame, bg="#000000", relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # --- Control Widgets ---
        self.btn_load_image = ttk.Button(control_frame, text="1a. Load Image File", command=self.load_image)
        self.btn_load_image.pack(side=tk.LEFT, padx=5)

        # --- NEW WIDGET: Button to capture from IDS camera ---
        self.btn_capture_ids = ttk.Button(control_frame, text="1b. Capture from IDS", command=self.capture_from_ids)
        self.btn_capture_ids.pack(side=tk.LEFT, padx=5)

        self.btn_load_dxf = ttk.Button(control_frame, text="2. Load DXF", command=self.load_dxf)
        self.btn_load_dxf.pack(side=tk.LEFT, padx=(15,5))
        
        self.btn_visualize = ttk.Button(control_frame, text="Visualize Inputs", state=tk.DISABLED, command=self.run_visualization)
        self.btn_visualize.pack(side=tk.LEFT, padx=(20, 5))
        
        self.btn_debug_alignment = ttk.Button(control_frame, text="Debug Alignment", state=tk.DISABLED, command=self.run_alignment_debug)
        self.btn_debug_alignment.pack(side=tk.LEFT, padx=5)
        
        self.btn_inspect = ttk.Button(control_frame, text="Align & Inspect", state=tk.DISABLED, command=self.run_alignment_and_inspection)
        self.btn_inspect.pack(side=tk.LEFT, padx=5)
        
        self.lbl_image_status = tk.Label(control_frame, text="Image: None", bg=self.colors["bg"], fg=self.colors["fg"], padx=10)
        self.lbl_image_status.pack(side=tk.LEFT)
        
        self.lbl_dxf_status = tk.Label(control_frame, text="DXF: None", bg=self.colors["bg"], fg=self.colors["fg"], padx=10)
        self.lbl_dxf_status.pack(side=tk.LEFT)

        # --- Image Display ---
        self.image_label = tk.Label(self.image_frame, bg="#000000")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label.bind("<Configure>", self.on_resize)
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat", background=self.colors["btn"], foreground=self.colors["fg"], font=('Helvetica', 10, 'bold'), borderwidth=0)
        style.map("TButton", 
                  background=[('active', self.colors["btn_active"]), ('disabled', '#3D3D3D')],
                  foreground=[('disabled', '#888888')])
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["fg"])

    # --- UPDATED METHOD: Handles the entire camera capture process ---
    def capture_from_ids(self):
        """Initializes, captures, and closes an IDS camera."""
        h_cam = ueye.HIDS(0) # 0 = First available camera
        mem_ptr = ueye.c_mem_p()
        mem_id = ueye.int()

        try:
            # 1. Initialize Camera
            ret = ueye.is_InitCamera(h_cam, None)
            if ret != ueye.IS_SUCCESS:
                messagebox.showerror("IDS Camera Error", f"Could not initialize camera. Error code: {ret}")
                return

            # 2. Get sensor info to determine image size
            sensor_info = ueye.SENSORINFO()
            ueye.is_GetSensorInfo(h_cam, sensor_info)
            width = int(sensor_info.nMaxWidth)
            height = int(sensor_info.nMaxHeight)
            
            # *** FIX 1: Set bits per pixel to 24 for color ***
            bits_per_pixel = 24

            # *** FIX 2: Set color mode to BGR8_PACKED for 24-bit color ***
            ret = ueye.is_SetColorMode(h_cam, ueye.IS_CM_BGR8_PACKED)
            if ret != ueye.IS_SUCCESS:
                 messagebox.showerror("IDS Camera Error", f"Could not set color mode. Is camera color?")
                 return

            # 4. Allocate memory for the image
            ueye.is_AllocImageMem(h_cam, width, height, bits_per_pixel, mem_ptr, mem_id)
            
            # 5. Set this memory as the active buffer
            ueye.is_SetImageMem(h_cam, mem_ptr, mem_id)

            # 6. Capture a single image ("Freeze Video")
            print("Capturing image from IDS camera...")
            ret = ueye.is_FreezeVideo(h_cam, ueye.IS_WAIT)
            if ret != ueye.IS_SUCCESS:
                messagebox.showerror("IDS Camera Error", "Could not capture image from camera.")
                return

            # 7. Access the image data from the buffer
            array = ueye.get_data(mem_ptr, width, height, bits_per_pixel, width * bits_per_pixel // 8, copy=True)
            
            # *** FIX 3: Reshape the array for 3 color channels (height, width, 3) ***
            frame_bgr = np.reshape(array, (height, width, 3))
            
            # Note: No cvtColor is needed as the data is already in BGR format

            # 8. Update the application's state with the new image
            self.original_cv_image = frame_bgr
            self.cv_image = self.original_cv_image.copy()
            self.lbl_image_status.config(text="Image: From IDS Camera")
            self.image_path = "IDS_Capture" # A placeholder to satisfy logic checks
            
            self.update_image_display(self.cv_image)
            self._check_files_loaded()
            
            messagebox.showinfo("Success", "Image captured successfully from IDS camera.")

        except Exception as e:
            error_details = traceback.format_exc()
            messagebox.showerror("IDS Camera Error", f"An unexpected error occurred: {e}\n\n{error_details}\n\n- Is the pyueye library installed?\n- Is the camera connected?\n- Are the IDS drivers installed?")
        finally:
            # 9. Clean up: Free memory and exit the camera handle
            if mem_ptr:
                ueye.is_FreeImageMem(h_cam, mem_ptr, mem_id)
            if h_cam:
                ueye.is_ExitCamera(h_cam)
            print("IDS Camera resources released.")

    def load_image(self):
        filepath = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if not filepath: return
        self.image_path = filepath
        try:
            self.original_cv_image = cv2.imread(self.image_path)
            if self.original_cv_image is None: raise ValueError("OpenCV could not read the image file.")
            self.cv_image = self.original_cv_image.copy()
            self.lbl_image_status.config(text=f"Image: {os.path.basename(self.image_path)}")
            self.update_image_display(self.cv_image)
            self._check_files_loaded()
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Failed to load image: {e}")
            self.reset_image_state()

    def load_dxf(self):
        filepath = filedialog.askopenfilename(title="Select a DXF File", filetypes=[("DXF Files", "*.dxf"), ("All files", "*.*")])
        if not filepath: return
        self.dxf_path = filepath
        try:
            print("--- Starting DXF Parsing and Path Assembly ---")
            doc = ezdxf.readfile(self.dxf_path)
            msp = doc.modelspace()
            self.cad_features = {'outline': np.array([]), 'holes': [], 'creases': []}
            
            def process_layer(layer_name):
                segments = []
                for entity in msp.query(f'*[layer=="{layer_name}"]'):
                    points = self._extract_entity_points(entity)
                    if len(points) > 1:
                        for i in range(len(points) - 1): segments.append((points[i], points[i+1]))
                
                assembled_paths = self._assemble_paths(segments)
                return [np.array(p, dtype=np.float32) for p in assembled_paths]

            outlines = process_layer("OUTLINE")
            if outlines:
                self.cad_features['outline'] = max(outlines, key=lambda p: cv2.arcLength(p.reshape(-1, 1, 2), False))
            
            self.cad_features['holes'] = process_layer("HOLES")
            self.cad_features['creases'] = process_layer("CREASES")

            self.lbl_dxf_status.config(text=f"DXF: {os.path.basename(self.dxf_path)}")
            self._check_files_loaded()
            print("--- DXF Parse Complete ---")
        except Exception as e:
            error_details = traceback.format_exc()
            messagebox.showerror("DXF Load Error", f"An error occurred during DXF processing:\n\n{e}\n\nDetails:\n{error_details}")
            self.reset_dxf_state()

    def _check_files_loaded(self):
        all_loaded = self.image_path is not None and self.dxf_path is not None and self.cad_features['outline'].size > 0
        state = tk.NORMAL if all_loaded else tk.DISABLED
        self.btn_visualize.config(state=state)
        self.btn_inspect.config(state=state)
        self.btn_debug_alignment.config(state=state)

    def on_resize(self, event=None):
        if self.cv_image is not None: self.update_image_display(self.cv_image)

    def update_image_display(self, image_to_show):
        if image_to_show is None: return
        self.cv_image = image_to_show
        frame_w, frame_h = self.image_frame.winfo_width(), self.image_frame.winfo_height()
        if frame_w <= 1 or frame_h <= 1: return
        img_h, img_w = image_to_show.shape[:2]
        
        scale = min(frame_w / img_w, frame_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        display_image = cv2.resize(image_to_show, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.tk_image = ImageTk.PhotoImage(image=pil_image)
        self.image_label.config(image=self.tk_image)
        
    def _extract_entity_points(self, entity):
        points = []
        dxf_type = entity.dxftype()
        try:
            if dxf_type == 'LINE':
                start, end = entity.dxf.start, entity.dxf.end
                points = [(start.x, start.y), (end.x, end.y)]
            elif dxf_type in ('LWPOLYLINE', 'POLYLINE'):
                points = [(p[0], p[1]) for p in entity.get_points('xy')]
            elif dxf_type in ('CIRCLE', 'ARC', 'ELLIPSE', 'SPLINE'):
                points = [(p.x, p.y) for p in entity.flattening(sagitta=0.01)]
        except Exception as e:
            print(f"   Warning: Could not process entity {dxf_type}: {e}")
        return points

    def _assemble_paths(self, segments):
        if not segments: return []
        paths = []
        tolerance = 1e-6
        while segments:
            current_path = list(segments.pop(0))
            while True:
                extended = False
                for i, segment in enumerate(segments):
                    p_start, p_end = segment
                    last_point_in_path = current_path[-1]
                    if math.dist(last_point_in_path, p_start) < tolerance:
                        current_path.append(p_end)
                        segments.pop(i)
                        extended = True
                        break
                    elif math.dist(last_point_in_path, p_end) < tolerance:
                        current_path.append(p_start)
                        segments.pop(i)
                        extended = True
                        break
                if not extended: break
            paths.append(current_path)
        return paths

    def reset_image_state(self):
        self.image_path, self.cv_image, self.original_cv_image = None, None, None
        self.lbl_image_status.config(text="Image: None")
        self.image_label.config(image='')
        self._check_files_loaded()

    def reset_dxf_state(self):
        self.dxf_path = None
        self.cad_features = {'outline': np.array([]), 'holes': [], 'creases': []}
        self.lbl_dxf_status.config(text="DXF: None")
        self._check_files_loaded()

    # ===================================================================
    # START: ZERNIKE MOMENT ALIGNMENT LOGIC (UNCHANGED)
    # ===================================================================

    def run_alignment_and_inspection(self):
        """
        Replaces the old vertex matching with the robust Zernike Moments approach.
        """
        print("\n--- Starting Alignment & Inspection (Zernike Moments Methodology) ---")
        if self.cv_image is None or self.cad_features['outline'].size == 0:
            messagebox.showerror("Input Error", "Please load both an image and a DXF file with an 'OUTLINE' layer.")
            return

        transform_matrix = self.align_shapes_with_zernike_moments()
        
        if transform_matrix is None:
            messagebox.showerror("Alignment Error", "Could not compute the transformation matrix. Check debug output for details.")
            return
        
        print("1. Zernike-based transformation matrix calculated successfully.")
        print("2. Transforming all DXF features to image space...")

        result_image = self.original_cv_image.copy()
        
        transformed_outline = cv2.transform(self.cad_features['outline'].reshape(-1, 1, 2), transform_matrix)
        cv2.polylines(result_image, [transformed_outline.astype(np.int32)], True, (0, 255, 255), 2) # Cyan outline

        for hole in self.cad_features['holes']:
            transformed_hole = cv2.transform(hole.reshape(-1, 1, 2), transform_matrix)
            cv2.polylines(result_image, [transformed_hole.astype(np.int32)], True, (255, 0, 255), 2) # Magenta holes
        for crease in self.cad_features['creases']:
            transformed_crease = cv2.transform(crease.reshape(-1, 1, 2), transform_matrix)
            cv2.polylines(result_image, [transformed_crease.astype(np.int32)], False, (255, 255, 0), 2) # Yellow creases
        
        print("3. Drawing final results...")
        self.update_image_display(result_image)
        messagebox.showinfo("Inspection Complete", "The DXF features have been aligned and drawn on the image using Zernike Moments.")
        print("--- Inspection Complete ---")
        
    def align_shapes_with_zernike_moments(self):
        """
        Orchestrates the Zernike moment pipeline as per the research paper.
        Returns: A 2x3 affine transformation matrix or None on failure.
        """
        cad_contour = self.cad_features['outline'].reshape(-1, 1, 2)
        img_contour = self.find_image_contour(self.original_cv_image)
        if img_contour is None:
            print("Error: No contour found in the image.")
            return None

        cad_centroid, cad_radius = self._normalize_contour(cad_contour)
        img_centroid, img_radius = self._normalize_contour(img_contour)
        if cad_radius == 0 or img_radius == 0:
            print("Error: Could not determine a valid normalization radius (shape has zero area?).")
            return None
        print(f"CAD Normalization: Centroid={cad_centroid}, Radius={cad_radius:.2f}")
        print(f"Image Normalization: Centroid={img_centroid}, Radius={img_radius:.2f}")
        
        cad_mask = self._create_binary_mask_from_contour(cad_contour)
        img_mask = self._create_binary_mask_from_contour(img_contour)
        
        print("Calculating Zernike moments for CAD shape...")
        zm_cad = self._calculate_complex_zernike_moments(cad_mask, degree=10)
        print("Calculating Zernike moments for Image shape...")
        zm_img = self._calculate_complex_zernike_moments(img_mask, degree=10)

        if zm_cad is None or zm_img is None:
            print("Error: Failed to calculate Zernike moments.")
            return None
            
        scale = img_radius / cad_radius
        rotation_rad = self._recover_rotation_from_zms(zm_cad, zm_img)
        rotation_deg = np.rad2deg(rotation_rad)
        
        print(f"Recovered Parameters: Scale={scale:.4f}, Rotation={rotation_deg:.2f} degrees")

        cad_cx, cad_cy = cad_centroid
        img_cx, img_cy = img_centroid

        M = cv2.getRotationMatrix2D((cad_cx, cad_cy), rotation_deg, scale)
        rotated_scaled_cad_centroid = np.dot(M, [cad_cx, cad_cy, 1])
        tx = img_cx - rotated_scaled_cad_centroid[0]
        ty = img_cy - rotated_scaled_cad_centroid[1]
        M[0, 2] += tx
        M[1, 2] += ty
        
        return M

    def _normalize_contour(self, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, 0
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        distances = [np.linalg.norm(np.array((cx, cy)) - pt[0]) for pt in contour]
        radius = np.max(distances)
        return (cx, cy), radius

    def _create_binary_mask_from_contour(self, contour, canvas_size=(512, 512)):
        """
        Creates a binary mask of a standardized size from a contour.
        This version correctly handles high-precision floating-point contours by
        performing all scaling calculations before the final conversion to integers,
        preserving the shape's detail.
        """
        # If the contour is empty, return a blank canvas.
        if contour.size == 0:
            return np.zeros(canvas_size, dtype=np.uint8)

        # 1. Manually calculate the bounding box from the float contour.
        #    This avoids using cv2.boundingRect which requires integer input.
        x_coords = contour[:, 0, 0]
        y_coords = contour[:, 0, 1]
        x_min, y_min = np.min(x_coords), np.min(y_coords)
        x_max, y_max = np.max(x_coords), np.max(y_coords)
        w, h = x_max - x_min, y_max - y_min

        if w == 0 or h == 0:
            return np.zeros(canvas_size, dtype=np.uint8)

        # 2. Determine scaling factor using floating-point dimensions.
        padding = 20
        canvas_w, canvas_h = canvas_size
        scale = min((canvas_w - 2 * padding) / w, (canvas_h - 2 * padding) / h)

        # 3. Calculate new dimensions and offsets with high precision.
        new_w, new_h = w * scale, h * scale
        offset_x = (canvas_w - new_w) / 2
        offset_y = (canvas_h - new_h) / 2

        # 4. Perform all transformations using floating-point arithmetic.
        #    Shift the original contour's origin to its top-left corner.
        shifted_contour = contour - [x_min, y_min]
        # Scale the contour up and translate it to the center of the canvas.
        transformed_contour = (shifted_contour * scale) + [offset_x, offset_y]

        # 5. THE CRITICAL STEP: Convert to integers ONLY AFTER all scaling and
        #    translation operations are complete.
        final_contour = transformed_contour.astype(np.int32)

        # 6. Create the final mask and draw the high-fidelity contour.
        mask = np.zeros(canvas_size, dtype=np.uint8)
        cv2.drawContours(mask, [final_contour], -1, 255, cv2.FILLED)
    
        return mask

    def _radial_poly(self, rho, n, m):
        if (n - m) % 2 != 0:
            return np.zeros_like(rho)
        radial = np.zeros_like(rho, dtype=float)
        for k in range((n - m) // 2 + 1):
            num = ((-1)**k * factorial(n - k))
            den = (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
            term = rho**(n - 2 * k)
            radial += (num / den) * term
        return radial

    def _calculate_complex_zernike_moments(self, mask, degree=8):
        h, w = mask.shape
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        rho = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)
        disk_mask = rho <= 1.0
        moments = []
        for n in range(degree + 1):
            for m in range(n + 1):
                if (n - m) % 2 == 0:
                    R_nm = self._radial_poly(rho, n, m)
                    Z_nm = R_nm * np.exp(1j * m * theta)
                    A_nm = (n + 1) / np.pi * np.sum(mask[disk_mask] * np.conj(Z_nm[disk_mask]))
                    moments.append(A_nm)
        return np.array(moments)

    def _recover_rotation_from_zms(self, zms_cad, zms_img):
        angles = []
        weights = []
        i = 0
        for n in range(11):
            for m in range(n + 1):
                if (n - m) % 2 == 0:
                    if m > 0:
                        phase_cad = np.angle(zms_cad[i])
                        phase_img = np.angle(zms_img[i])
                        angle_diff = phase_img - phase_cad
                        angle_estimate = -angle_diff / m
                        angles.append(angle_estimate)
                        weights.append(np.abs(zms_cad[i]) * np.abs(zms_img[i]))
                    i += 1
        if not angles:
            return 0.0
        angles = np.array(angles)
        weights = np.array(weights)
        x = np.sum(weights * np.cos(angles))
        y = np.sum(weights * np.sin(angles))
        return np.arctan2(y, x)

    # ===================================================================
    # END: ZERNIKE MOMENT ALIGNMENT LOGIC
    # ===================================================================

    def find_image_contour(self, image):
        if image is None: return None
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound_hsv, upper_bound_hsv = np.array([5, 50, 50]), np.array([30, 255, 255])
        color_mask = cv2.inRange(hsv_image, lower_bound_hsv, upper_bound_hsv)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea) if contours else None

    # --- Visualization and Debugging ---

    def run_visualization(self):
        print("--- Running Input Visualization ---")
        if self.cad_features['outline'].size > 0:
            all_points_list = [self.cad_features['outline']] + self.cad_features['holes'] + self.cad_features['creases']
            all_points = np.vstack(all_points_list)
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)
            CANVAS_W, CANVAS_H, padding = 800, 600, 50
            cad_w, cad_h = x_max - x_min, y_max - y_min
            if cad_w > 0 and cad_h > 0:
                scale = min((CANVAS_W - 2 * padding) / cad_w, (CANVAS_H - 2 * padding) / cad_h)
                cad_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype="uint8")
                def transform_pt(points):
                    return ((points - [x_min, y_min]) * [scale, -scale] + [padding, CANVAS_H - padding]).astype(np.int32)
                cv2.polylines(cad_canvas, [transform_pt(self.cad_features['outline'])], True, (255, 255, 255), 1)
                cv2.polylines(cad_canvas, [transform_pt(p) for p in self.cad_features['holes']], True, (0, 255, 255), 1)
                cv2.polylines(cad_canvas, [transform_pt(p) for p in self.cad_features['creases']], False, (255, 255, 0), 1)
                cv2.imshow("DEBUG: Parsed CAD", cad_canvas)

        if self.cv_image is not None:
            image_contour = self.find_image_contour(self.cv_image)
            if image_contour is not None:
                debug_image = self.original_cv_image.copy()
                cv2.drawContours(debug_image, [image_contour.astype(np.int32)], -1, (255, 0, 255), 3)
                cv2.imshow("DEBUG: Detected Image Contour", debug_image)
            else:
                messagebox.showinfo("Visualize", "No contour detected in image with current HSV settings.")
        
        messagebox.showinfo("Debug", "Debug windows opened. Press any key on them to close.")

    def run_alignment_debug(self):
        print("\n--- Running Zernike Alignment Debug Visualization ---")
        cad_contour = self.cad_features['outline'].reshape(-1, 1, 2)
        img_contour = self.find_image_contour(self.original_cv_image)
        if img_contour is None:
            return messagebox.showerror("Debug Error", "No contour found in image.")
        
        cad_mask = self._create_binary_mask_from_contour(cad_contour)
        img_mask = self._create_binary_mask_from_contour(img_contour)
        
        cad_mask_display = cv2.resize(cad_mask, (300, 300), interpolation=cv2.INTER_NEAREST)
        img_mask_display = cv2.resize(img_mask, (300, 300), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow("Debug 1: CAD Binary Mask", cad_mask_display)
        cv2.imshow("Debug 2: Image Binary Mask", img_mask_display)
        
        transform_matrix = self.align_shapes_with_zernike_moments()
        if transform_matrix is not None:
            debug_final_image = self.original_cv_image.copy()
            transformed_outline = cv2.transform(self.cad_features['outline'].reshape(-1, 1, 2), transform_matrix)
            cv2.polylines(debug_final_image, [transformed_outline.astype(np.int32)], True, (0, 255, 0), 3) # Bright green for debug
            cv2.imshow("Debug 3: Final Zernike Alignment", debug_final_image)
            messagebox.showinfo("Debug", "Debug windows opened. Examine them to diagnose the alignment process.")
        else:
            messagebox.showerror("Debug Error", "Alignment failed during debug run.")

if __name__ == "__main__":
    root = tk.Tk()
    app = InspectionApp(root)
    root.mainloop()
