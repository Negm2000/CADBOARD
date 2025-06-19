import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import ezdxf
import math
import numpy as np
import os
import traceback
from scipy.spatial import KDTree

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
        self.root.title("DIE-VIS: Visualizer & Inspector (Moments+ICP Pipeline Engine)")
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

    def capture_from_ids(self):
        h_cam = ueye.HIDS(0) 
        mem_ptr = ueye.c_mem_p()
        mem_id = ueye.int()
        try:
            ret = ueye.is_InitCamera(h_cam, None)
            if ret != ueye.IS_SUCCESS:
                messagebox.showerror("IDS Camera Error", f"Could not initialize camera. Error code: {ret}")
                return
            sensor_info = ueye.SENSORINFO()
            ueye.is_GetSensorInfo(h_cam, sensor_info)
            width, height = int(sensor_info.nMaxWidth), int(sensor_info.nMaxHeight)
            bits_per_pixel = 24
            ret = ueye.is_SetColorMode(h_cam, ueye.IS_CM_BGR8_PACKED)
            if ret != ueye.IS_SUCCESS:
                messagebox.showerror("IDS Camera Error", f"Could not set color mode. Is camera color?")
                return
            ueye.is_AllocImageMem(h_cam, width, height, bits_per_pixel, mem_ptr, mem_id)
            ueye.is_SetImageMem(h_cam, mem_ptr, mem_id)
            print("Capturing image from IDS camera...")
            ret = ueye.is_FreezeVideo(h_cam, ueye.IS_WAIT)
            if ret != ueye.IS_SUCCESS:
                messagebox.showerror("IDS Camera Error", "Could not capture image from camera.")
                return
            array = ueye.get_data(mem_ptr, width, height, bits_per_pixel, width * bits_per_pixel // 8, copy=True)
            frame_bgr = np.reshape(array, (height, width, 3))
            self.original_cv_image = frame_bgr
            self.cv_image = self.original_cv_image.copy()
            self.lbl_image_status.config(text="Image: From IDS Camera")
            self.image_path = "IDS_Capture"
            self.update_image_display(self.cv_image)
            self._check_files_loaded()
            messagebox.showinfo("Success", "Image captured successfully from IDS camera.")
        except Exception as e:
            error_details = traceback.format_exc()
            messagebox.showerror("IDS Camera Error", f"An unexpected error occurred: {e}\n\n{error_details}\n\n- Is the pyueye library installed?\n- Is the camera connected?\n- Are the IDS drivers installed?")
        finally:
            if mem_ptr: ueye.is_FreeImageMem(h_cam, mem_ptr, mem_id)
            if h_cam: ueye.is_ExitCamera(h_cam)
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
            holes = process_layer("HOLES")
            creases = process_layer("CREASES")

            ## FIX: Correct the CAD coordinate system to match the image's system (Y-axis pointing down).
            ## This is done once at load time to ensure all subsequent operations are consistent.
            all_points_for_transform = []
            if outlines: all_points_for_transform.extend(outlines)
            if holes: all_points_for_transform.extend(holes)
            if creases: all_points_for_transform.extend(creases)

            if all_points_for_transform:
                master_point_cloud = np.vstack([p for p in all_points_for_transform])
                y_max_cad = np.max(master_point_cloud[:, 1])

                def flip_y_axis(points):
                    points[:, 1] = y_max_cad - points[:, 1]
                    return points

                outlines = [flip_y_axis(p) for p in outlines]
                holes = [flip_y_axis(p) for p in holes]
                creases = [flip_y_axis(p) for p in creases]
                print("   Info: Flipped CAD Y-axis to match image coordinate system.")

            if outlines:
                self.cad_features['outline'] = max(outlines, key=lambda p: cv2.arcLength(p.reshape(-1, 1, 2), False))
            
            self.cad_features['holes'] = holes
            self.cad_features['creases'] = creases

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

    # --- CORE ALIGNMENT LOGIC (MOMENTS+ICP PIPELINE) ---
    
    def run_alignment_and_inspection(self):
        print("\n--- Starting Alignment & Inspection (Moments+ICP Pipeline) ---")
        if self.cv_image is None or self.cad_features['outline'].size == 0:
            messagebox.showerror("Input Error", "Please load both an image and a DXF file with an 'OUTLINE' layer.")
            return

        # The new pipeline returns a 2x3 rigid transformation matrix
        transform_matrix = self.align_with_moments_icp_pipeline()
        
        if transform_matrix is None:
            messagebox.showerror("Alignment Error", "Could not compute the transformation matrix. Check console for details.")
            return
        
        print("5. Final transformation matrix calculated successfully.")
        print("6. Transforming all DXF features to image space and drawing results...")

        result_image = self.original_cv_image.copy()
        
        def apply_rigid_transform(points, M):
            # cv2.transform requires shape (N, 1, 2) or (1, N, 2)
            points_reshaped = points.reshape(-1, 1, 2)
            transformed_points = cv2.transform(points_reshaped, M)
            return transformed_points

        transformed_outline = apply_rigid_transform(self.cad_features['outline'], transform_matrix)
        cv2.polylines(result_image, [transformed_outline.astype(np.int32)], True, (0, 255, 255), 2)

        for hole in self.cad_features['holes']:
            transformed_hole = apply_rigid_transform(hole, transform_matrix)
            cv2.polylines(result_image, [transformed_hole.astype(np.int32)], True, (255, 0, 255), 2)
        for crease in self.cad_features['creases']:
            transformed_crease = apply_rigid_transform(crease, transform_matrix)
            cv2.polylines(result_image, [transformed_crease.astype(np.int32)], False, (255, 255, 0), 2)
        
        self.update_image_display(result_image)
        messagebox.showinfo("Inspection Complete", "The DXF features have been aligned using the Moments+ICP pipeline.")
        print("--- Inspection Complete ---")

    def align_with_moments_icp_pipeline(self):
        """Orchestrates the Moments+ICP registration pipeline."""
        # --- Stage 0: Preprocessing ---
        print("1. Preprocessing: Extracting and preparing contours...")
        img_contour = self.find_image_contour(self.original_cv_image)
        if img_contour is None:
            print("   Error: No contour found in the image.")
            return None
        
        # CAD contour is already in the correct (Y-down) coordinate system from the load_dxf step
        cad_contour = self.cad_features['outline'].reshape(-1, 2).astype(np.float32)
        
        # --- Stage 1: Coarse Global Alignment (Moments) ---
        print("2. Coarse Alignment: Using Shape Moments to find initial pose...")
        coarse_transform = self._align_with_moments(cad_contour, img_contour)
        if coarse_transform is None:
             print("   Error: Coarse alignment with Moments failed.")
             return None
        print("3. Coarse alignment successful.")

        # --- Stage 2: Fine Iterative Refinement (Iterative Closest Point) ---
        print("4. Fine Refinement: Using ICP to finalize alignment...")
        final_transform = self._iterative_closest_point(
            source_points=cad_contour,
            target_points=img_contour,
            initial_matrix=coarse_transform
        )

        return final_transform

    def _resample_contour(self, contour, num_points):
        """
        Resamples a contour to have a specific number of uniformly spaced points.
        This version uses interpolation for robustness.
        """
        contour = contour.reshape(-1, 2).astype(np.float32)
        
        # Calculate the cumulative arc length along the contour, ensuring no negative inputs to sqrt
        distances = np.cumsum(np.sqrt(np.maximum(0, np.sum(np.diff(contour, axis=0, append=contour[0:1])**2, axis=1))))
        arc_length = distances[-1]
        
        # Handle degenerate case where contour has no length
        if arc_length == 0:
            return np.array([contour[0]] * num_points, dtype=np.float32)
            
        # Create the new, evenly spaced distances
        fx = distances / arc_length
        fy = np.linspace(0, 1, num_points)
        
        # Interpolate the x and y coordinates
        x_new = np.interp(fy, fx, contour[:, 0])
        y_new = np.interp(fy, fx, contour[:, 1])
        
        return np.array([x_new, y_new]).T.astype(np.float32)

    def _align_with_moments(self, cad_contour, img_contour):
        """
        Estimates a coarse transformation using shape moments.
        Calculates translation, scale, and rotation.
        """
        try:
            # --- Calculate moments for both contours ---
            M_cad = cv2.moments(cad_contour)
            M_img = cv2.moments(img_contour)
            
            if M_cad["m00"] == 0 or M_img["m00"] == 0:
                print("   Error: Contour with zero area found.")
                return None
            
            # --- 1. Calculate Centroids for Translation ---
            cad_cx = M_cad["m10"] / M_cad["m00"]
            cad_cy = M_cad["m01"] / M_cad["m00"]
            img_cx = M_img["m10"] / M_img["m00"]
            img_cy = M_img["m01"] / M_img["m00"]

            # --- 2. Calculate Scale from contour areas ---
            scale = math.sqrt(M_img["m00"] / M_cad["m00"])
            
            # --- 3. Calculate Rotation from orientation ---
            def get_orientation(M):
                mu11 = M['mu11']
                mu20 = M['mu20']
                mu02 = M['mu02']
                return 0.5 * math.atan2(2 * mu11, mu20 - mu02)

            angle_cad = get_orientation(M_cad)
            angle_img = get_orientation(M_img)
            rotation_angle_rad = angle_img - angle_cad

            # --- 4. Build the 2x3 Affine Transformation Matrix ---
            cos_a = math.cos(rotation_angle_rad)
            sin_a = math.sin(rotation_angle_rad)
            
            tx = img_cx - (scale * (cad_cx * cos_a - cad_cy * sin_a))
            ty = img_cy - (scale * (cad_cx * sin_a + cad_cy * cos_a))

            transform = np.array([
                [scale * cos_a, -scale * sin_a, tx],
                [scale * sin_a,  scale * cos_a, ty]
            ], dtype=np.float32)

            return transform
        except Exception as e:
            print(f"   Error during moments-based alignment: {e}")
            return None

    def _iterative_closest_point(self, source_points, target_points, initial_matrix, max_iterations=100, tolerance=1e-5):
        """
        Custom implementation of the Iterative Closest Point algorithm.
        Refines the alignment of source_points to target_points.
        """
        source_pts = source_points.reshape(-1, 2)
        target_pts = target_points.reshape(-1, 2)

        target_kdtree = KDTree(target_pts)
        
        current_transform = initial_matrix.copy()
        
        prev_error = float('inf')

        for i in range(max_iterations):
            source_transformed = cv2.transform(source_pts.reshape(-1, 1, 2), current_transform).reshape(-1, 2)
            
            distances, indices = target_kdtree.query(source_transformed)
            correspondences = target_pts[indices]
            
            # Use LMEDS for robustness against some outlier correspondences
            new_transform, _ = cv2.estimateAffine2D(source_pts, correspondences, method=cv2.LMEDS)

            if new_transform is None:
                print("   ICP Warning: Could not estimate transform in iteration. Using previous.")
                break # Exit if estimation fails

            current_transform = new_transform
            
            mean_error = np.mean(distances**2)
            if abs(prev_error - mean_error) < tolerance:
                print(f"   ICP converged in {i+1} iterations. Final MSE: {mean_error:.4f}")
                break
            prev_error = mean_error
        else:
             print(f"   ICP reached max iterations. Final MSE: {prev_error:.4f}")

        return current_transform

    def find_image_contour(self, image):
        """Finds the largest contour in the image using HSV color masking."""
        if image is None: return None
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # These values may need tuning for different lighting/object colors
        lower_bound_hsv, upper_bound_hsv = np.array([5, 50, 50]), np.array([30, 255, 255])
        color_mask = cv2.inRange(hsv_image, lower_bound_hsv, upper_bound_hsv)
        
        kernel = np.ones((5,5),np.uint8)
        mask_cleaned = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, hierarchy = cv2.findContours(mask_cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_area = 0
        largest_contour = None
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == -1: # It's an outer contour
                area = cv2.contourArea(contour)
                if area > largest_area:
                    largest_area = area
                    largest_contour = contour
        
        return largest_contour.astype(np.float32) if largest_contour is not None else None

    # --- VISUALIZATION AND DEBUGGING ---

    def _resize_for_display(self, window_name, image, max_dim=900):
        """Resizes an image to a maximum dimension for display while preserving aspect ratio."""
        h, w = image.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            display_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imshow(window_name, display_image)
        else:
            cv2.imshow(window_name, image)

    def run_visualization(self):
        print("--- Running Input Visualization ---")
        if self.cad_features['outline'].size > 0:
            all_points_list = [self.cad_features['outline']] + self.cad_features['holes'] + self.cad_features['creases']
            all_points = np.vstack([p.reshape(-1, 2) for p in all_points_list])
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)
            CANVAS_W, CANVAS_H, padding = 800, 600, 50
            cad_w, cad_h = x_max - x_min, y_max - y_min
            if cad_w > 0 and cad_h > 0:
                scale = min((CANVAS_W - 2 * padding) / cad_w, (CANVAS_H - 2 * padding) / cad_h)
                cad_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype="uint8")

                ## FIX: The transformation for visualization is now simpler because the coordinate systems
                ## already match. We no longer need to manually flip the Y-axis here.
                def transform_pt(points):
                    return ((points - [x_min, y_min]) * scale + padding).astype(np.int32)
                
                cv2.polylines(cad_canvas, [transform_pt(self.cad_features['outline'])], True, (255, 255, 255), 1)
                cv2.polylines(cad_canvas, [transform_pt(p) for p in self.cad_features['holes']], True, (0, 255, 255), 1)
                cv2.polylines(cad_canvas, [transform_pt(p) for p in self.cad_features['creases']], False, (255, 255, 0), 1)
                self._resize_for_display("DEBUG: Parsed CAD", cad_canvas)

        if self.cv_image is not None:
            image_contour = self.find_image_contour(self.cv_image)
            if image_contour is not None:
                debug_image = self.original_cv_image.copy()
                cv2.drawContours(debug_image, [image_contour.astype(np.int32)], -1, (255, 0, 255), 3)
                self._resize_for_display("DEBUG: Detected Image Contour", debug_image)
            else:
                messagebox.showinfo("Visualize", "No contour detected in image with current HSV settings.")
        
        messagebox.showinfo("Debug", "Debug windows opened. Press any key on them to close.")

    def run_alignment_debug(self):
        """Debug function to visualize the new Moments+ICP pipeline."""
        print("\n--- Running Moments+ICP Pipeline Debug Visualization ---")
        
        # --- Stage 0: Inputs ---
        img_contour = self.find_image_contour(self.original_cv_image)
        if img_contour is None: return messagebox.showerror("Debug Error", "No contour found in image.")
        cad_contour = self.cad_features['outline'].reshape(-1, 2).astype(np.float32)
        
        debug_img_inputs = self.original_cv_image.copy()
        cv2.drawContours(debug_img_inputs, [img_contour.astype(np.int32)], -1, (255, 0, 255), 2, lineType=cv2.LINE_AA)
        self._resize_for_display("Debug 0: Detected Image Contour", debug_img_inputs)
        
        # --- Stage 1: Coarse Alignment ---
        coarse_transform = self._align_with_moments(cad_contour, img_contour)
        
        if coarse_transform is None:
            return messagebox.showerror("Debug Error", "Coarse alignment (Moments) step failed.")

        debug_img_coarse = self.original_cv_image.copy()
        cv2.drawContours(debug_img_coarse, [img_contour.astype(np.int32)], -1, (255, 0, 255), 2)
        coarsely_aligned_cad = cv2.transform(cad_contour.reshape(-1, 1, 2), coarse_transform)
        cv2.polylines(debug_img_coarse, [coarsely_aligned_cad.astype(np.int32)], True, (0, 255, 255), 2)
        self._resize_for_display("Debug 1: Coarse (Moments) Alignment", debug_img_coarse)

        # --- Stage 2: Fine Alignment ---
        final_transform = self._iterative_closest_point(cad_contour, img_contour, coarse_transform)
        if final_transform is None:
            return messagebox.showerror("Debug Error", "Fine alignment (ICP) step failed.")

        debug_img_final = self.original_cv_image.copy()
        cv2.drawContours(debug_img_final, [img_contour.astype(np.int32)], -1, (255, 0, 255), 2)
        final_aligned_cad = cv2.transform(cad_contour.reshape(-1, 1, 2), final_transform)
        cv2.polylines(debug_img_final, [final_aligned_cad.astype(np.int32)], True, (0, 255, 0), 2) # Green for final
        self._resize_for_display("Debug 2: Final (ICP) Alignment", debug_img_final)

        messagebox.showinfo("Debug", "Debug windows opened. Examine the stages of alignment.")


if __name__ == "__main__":
    root = tk.Tk()
    app = InspectionApp(root)
    root.mainloop()
