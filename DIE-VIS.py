import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import ezdxf
import math
import numpy as np
import os
import traceback

class InspectionApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("DIE-VIS: Visualizer & Inspector")
        self.root.geometry("1200x850")

        # --- State Variables ---
        self.image_path = None
        self.dxf_path = None
        self.cv_image = None
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
        self.btn_load_image = ttk.Button(control_frame, text="1. Load Image", command=self.load_image)
        self.btn_load_image.pack(side=tk.LEFT, padx=5)

        self.btn_load_dxf = ttk.Button(control_frame, text="2. Load DXF", command=self.load_dxf)
        self.btn_load_dxf.pack(side=tk.LEFT, padx=5)
        
        self.btn_visualize = ttk.Button(control_frame, text="Visualize Inputs", state=tk.DISABLED, command=self.run_visualization)
        self.btn_visualize.pack(side=tk.LEFT, padx=(20, 5))
        
        # NEW: Button to debug the feature matching process specifically
        self.btn_debug_matching = ttk.Button(control_frame, text="Debug Matching", state=tk.DISABLED, command=self.run_feature_matching_debug)
        self.btn_debug_matching.pack(side=tk.LEFT, padx=5)
        
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
        style.configure("TButton", padding=6, relief="flat", background=self.colors["btn"], foreground=self.colors["fg"], font=('Helvetica', 10, 'bold'))
        style.map("TButton", background=[('active', self.colors["btn_active"]), ('disabled', '#3D3D3D')])

    def load_image(self):
        filepath = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if not filepath: return
        self.image_path = filepath
        try:
            self.cv_image = cv2.imread(self.image_path)
            if self.cv_image is None: raise ValueError("OpenCV could not read the image file.")
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
            outline_segments = []
            for entity in msp.query('*[layer=="OUTLINE"]'):
                points = self._extract_entity_points(entity)
                for i in range(len(points) - 1): outline_segments.append((points[i], points[i+1]))
            if outline_segments:
                assembled_outlines = self._assemble_paths(outline_segments)
                if assembled_outlines:
                    self.cad_features['outline'] = np.array(max(assembled_outlines, key=len), dtype=np.float32)
            all_hole_segments = []
            for entity in msp.query('*[layer=="HOLES"]'):
                points = self._extract_entity_points(entity)
                for i in range(len(points) - 1): all_hole_segments.append((points[i], points[i+1]))
            if all_hole_segments:
                assembled_hole_paths = self._assemble_paths(all_hole_segments)
                for path in assembled_hole_paths: self.cad_features['holes'].append(np.array(path, dtype=np.float32))
            all_crease_segments = []
            for entity in msp.query('*[layer=="CREASES"]'):
                points = self._extract_entity_points(entity)
                for i in range(len(points) - 1): all_crease_segments.append((points[i], points[i+1]))
            if all_crease_segments:
                assembled_crease_paths = self._assemble_paths(all_crease_segments)
                for path in assembled_crease_paths: self.cad_features['creases'].append(np.array(path, dtype=np.float32))
            self.lbl_dxf_status.config(text=f"DXF: {os.path.basename(self.dxf_path)}")
            self._check_files_loaded()
            print("--- DXF Parse Complete ---")
        except Exception as e:
            error_details = traceback.format_exc()
            messagebox.showerror("DXF Load Error", f"An error occurred during DXF processing:\n\n{e}\n\nDetails:\n{error_details}")
            self.reset_dxf_state()

    def _check_files_loaded(self):
        if self.image_path and self.dxf_path:
            self.btn_visualize.config(state=tk.NORMAL)
            self.btn_inspect.config(state=tk.NORMAL)
            self.btn_debug_matching.config(state=tk.NORMAL)
        else:
            self.btn_visualize.config(state=tk.DISABLED)
            self.btn_inspect.config(state=tk.DISABLED)
            self.btn_debug_matching.config(state=tk.DISABLED)

    def run_alignment_and_inspection(self):
        print("\n--- Starting Alignment & Inspection (DIE-VIS Methodology) ---")
        image_contour = self.find_image_contour(self.cv_image)
        if image_contour is None:
            messagebox.showerror("Alignment Error", "No contour found in the image.")
            return
        image_mask = np.zeros(self.cv_image.shape[:2], dtype="uint8")
        cv2.drawContours(image_mask, [image_contour.astype(int)], -1, 255, -1)
        dxf_outline = self.cad_features['outline']
        if dxf_outline.size == 0:
            messagebox.showerror("Alignment Error", "No outline found in the DXF file.")
            return
        dxf_mask, _ = self._render_cad_to_image(dxf_outline)
        print("1. Finding matching features between CAD and Image...")
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(dxf_mask, None)
        kp2, des2 = orb.detectAndCompute(image_mask, None)
        if des1 is None or des2 is None:
            messagebox.showerror("Alignment Error", "Could not find features in one of the inputs. Alignment failed.")
            return
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]
        if len(good_matches) < 4:
            messagebox.showerror("Alignment Error", f"Not enough good matches found ({len(good_matches)}/4). Cannot compute homography.")
            return
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        print(f"   Found {len(good_matches)} good feature matches.")
        print("2. Calculating Homography matrix...")
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography_matrix is None:
            messagebox.showerror("Alignment Error", "Could not compute the transformation matrix from the matched features.")
            return
        print("   Homography calculated successfully.")
        print("3. Transforming all DXF features to image space...")
        transformed_outline = cv2.perspectiveTransform(dxf_outline.reshape(-1, 1, 2), homography_matrix)
        transformed_holes = [cv2.perspectiveTransform(h.reshape(-1, 1, 2), homography_matrix) for h in self.cad_features['holes']]
        transformed_creases = [cv2.perspectiveTransform(c.reshape(-1, 1, 2), homography_matrix) for c in self.cad_features['creases']]
        print("4. Drawing final results...")
        result_image = self.cv_image.copy()
        cv2.polylines(result_image, [transformed_outline.astype(np.int32)], True, (255, 255, 0), 2)
        cv2.polylines(result_image, [h.astype(np.int32) for h in transformed_holes], True, (0, 255, 255), 2)
        cv2.polylines(result_image, [c.astype(np.int32) for c in transformed_creases], False, (255, 255, 0), 2)
        self.update_image_display(result_image)
        messagebox.showinfo("Inspection Complete", "The DXF features have been aligned and drawn on the image using feature matching.")
        print("--- Inspection Complete ---")

    # --- UPDATED: Enhanced debugging function ---
    def run_feature_matching_debug(self):
        """
        Generates and displays intermediate images from the feature
        matching process to help diagnose alignment failures.
        """
        print("\n--- Running Feature Matching Debug Visualization ---")

        # 1. Generate and show the Image Mask
        image_contour = self.find_image_contour(self.cv_image)
        if image_contour is None: return messagebox.showerror("Debug Error", "No contour found in the image.")
        image_mask = np.zeros(self.cv_image.shape[:2], dtype="uint8")
        cv2.drawContours(image_mask, [image_contour.astype(int)], -1, 255, -1)
        cv2.imshow("Debug 1: Image Mask", image_mask)

        # 2. Generate and show the CAD Mask
        dxf_outline = self.cad_features['outline']
        if dxf_outline.size == 0: return messagebox.showerror("Debug Error", "No outline found in the DXF file.")
        dxf_mask, _ = self._render_cad_to_image(dxf_outline)
        cv2.imshow("Debug 2: Rendered CAD Mask", dxf_mask)

        # 3. Detect features and descriptors
        orb = cv2.ORB_create(nfeatures=2000)
        kp_cad, des_cad = orb.detectAndCompute(dxf_mask, None)
        kp_img, des_img = orb.detectAndCompute(image_mask, None)
        
        # --- NEW: More detailed console output and visualizations ---
        print(f"   CAD Mask Shape: {dxf_mask.shape}, Keypoints Found: {len(kp_cad)}")
        print(f"   Image Mask Shape: {image_mask.shape}, Keypoints Found: {len(kp_img)}")
        
        # Visualize keypoints on each mask individually
        cad_kp_img = cv2.drawKeypoints(dxf_mask, kp_cad, None, color=(0, 255, 0), flags=0)
        img_kp_img = cv2.drawKeypoints(image_mask, kp_img, None, color=(0, 255, 0), flags=0)
        cv2.imshow("Debug 2a: Keypoints on CAD Mask", cad_kp_img)
        cv2.imshow("Debug 2b: Keypoints on Image Mask", img_kp_img)

        if des_cad is None or des_img is None:
            return messagebox.showerror("Debug Error", "Could not find features in one of the masks.")

        # 4. Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_cad, des_img)
        matches = sorted(matches, key=lambda x: x.distance)
        print(f"   Found {len(matches)} total feature matches.")
        
        # 5. Visualize the matches
        num_matches_to_show = min(len(matches), 50)
        # Be explicit with variable names to avoid any confusion
        match_visualization = cv2.drawMatches(
            dxf_mask, kp_cad, 
            image_mask, kp_img, 
            matches[:num_matches_to_show], None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imshow("Debug 3: Feature Matches", match_visualization)
        
        messagebox.showinfo("Debug", "Debug windows opened. Examine them to diagnose the matching process.")

    def _render_cad_to_image(self, outline, holes=[], creases=[], padding=50):
        """
        Renders CAD features onto a black image (mask), intelligently scaling
        the drawing to a reasonable size for feature detection.
        """
        all_points = [outline] + holes + creases
        if not any(p.size > 0 for p in all_points):
            return np.zeros((100, 100), dtype="uint8"), None

        v_stack = np.vstack([p for p in all_points if p.size > 0])
        x_min, y_min = np.min(v_stack, axis=0)
        x_max, y_max = np.max(v_stack, axis=0)

        cad_w = x_max - x_min
        cad_h = y_max - y_min

        if cad_w == 0 or cad_h == 0:
            return np.zeros((100, 100), dtype="uint8"), None

        # --- NEW: Intelligent Scaling Logic ---
        # Define a target size for the rendered mask for consistency.
        TARGET_WIDTH = 800
        
        # Calculate scale factor to fit the drawing to the target width
        scale = TARGET_WIDTH / cad_w
        
        # Calculate the new canvas dimensions while maintaining aspect ratio
        canvas_w = int(TARGET_WIDTH + padding * 2)
        canvas_h = int(cad_h * scale + padding * 2)
        
        canvas = np.zeros((canvas_h, canvas_w), dtype="uint8")

        # The new transform function now includes scaling
        def transform(points):
            # First, translate points relative to the top-left corner
            translated_pts = points - [x_min, y_min]
            # Next, scale the points
            scaled_pts = translated_pts * scale
            # Finally, add padding
            final_pts = scaled_pts + [padding, padding]
            return final_pts.astype(int)

        # Draw the features as a filled shape for a solid mask
        cv2.fillPoly(canvas, [transform(outline)], 255)
        # You could optionally draw holes here if needed for more complex masks
        # For now, we only need the outer boundary as per the paper.
             
        return canvas, None # Returning None for the second value as it's not used

    def on_resize(self, event=None):
        if self.cv_image is not None: self.update_image_display(self.cv_image)

    def update_image_display(self, image_to_show):
        if image_to_show is None: return
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

    def find_image_contour(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound_hsv, upper_bound_hsv = np.array([5, 50, 50]), np.array([30, 255, 255])
        color_mask = cv2.inRange(hsv_image, lower_bound_hsv, upper_bound_hsv)
        kernel = np.ones((5,5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea) if contours else None

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
                points = [(p.x, p.y) for p in entity.flattening(sagitta=0.001)]
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
        self.image_path, self.cv_image = None, None
        self.lbl_image_status.config(text="Image: None")
        self.image_label.config(image='')
        self._check_files_loaded()

    def reset_dxf_state(self):
        self.dxf_path = None
        self.cad_features = {'outline': np.array([]), 'holes': [], 'creases': []}
        self.lbl_dxf_status.config(text="DXF: None")
        self._check_files_loaded()
        
    def run_visualization(self):
        print("--- Running Input Visualization ---")
        if not (self.cad_features['outline'].size > 0 or self.cad_features['holes'] or self.cad_features['creases']):
            messagebox.showinfo("Visualize", "No CAD features were found in the loaded DXF file.")
            return
        all_points_list = []
        if self.cad_features['outline'].size > 0: all_points_list.append(self.cad_features['outline'])
        if self.cad_features['holes']: all_points_list.extend(self.cad_features['holes'])
        if self.cad_features['creases']: all_points_list.extend(self.cad_features['creases'])
        all_points = np.vstack(all_points_list)
        x_min, y_min, x_max, y_max = *np.min(all_points, axis=0), *np.max(all_points, axis=0)
        CANVAS_W, CANVAS_H, padding = 800, 600, 50
        cad_w, cad_h = x_max - x_min, y_max - y_min
        if cad_w > 0 and cad_h > 0:
            scale = min((CANVAS_W - 2 * padding) / cad_w, (CANVAS_H - 2 * padding) / cad_h)
            cad_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype="uint8")
            def transform_pt(points):
                return ((points - [x_min, y_min]) * [scale, -scale] + [padding, CANVAS_H - padding]).astype(np.int32)
            if self.cad_features['outline'].size > 0:
                cv2.polylines(cad_canvas, [transform_pt(self.cad_features['outline'])], True, (255, 255, 255), 1)
            cv2.polylines(cad_canvas, [transform_pt(p) for p in self.cad_features['holes']], True, (0, 255, 255), 1)
            cv2.polylines(cad_canvas, [transform_pt(p) for p in self.cad_features['creases']], False, (255, 255, 0), 1)
            cv2.imshow("DEBUG: Parsed CAD", cad_canvas)
        image_contour = self.find_image_contour(self.cv_image)
        if image_contour is not None:
            debug_image = self.cv_image.copy()
            cv2.drawContours(debug_image, [image_contour.astype(np.int32)], -1, (255, 0, 255), 3)
            cv2.imshow("DEBUG: Detected Image Contour", debug_image)
        messagebox.showinfo("Debug", "Debug windows opened. Press any key on them to close.")

if __name__ == "__main__":
    root = tk.Tk()
    app = InspectionApp(root)
    root.mainloop()