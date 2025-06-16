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
        # This will hold the image currently displayed in the GUI
        self.display_image = None
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
            self.display_image = self.cv_image.copy()
            self.lbl_image_status.config(text=f"Image: {os.path.basename(self.image_path)}")
            self.update_image_display()
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

    # --- REWRITTEN: Main alignment function using Hu Moments ---
    def run_alignment_and_inspection(self):
        print("\n--- Starting Alignment & Inspection (Hu Moments Methodology) ---")
        
        image_contour = self.find_image_contour(self.cv_image)
        if image_contour is None:
            return messagebox.showerror("Alignment Error", "No contour found in the image.")
        image_contour = image_contour.astype(np.int32)

        dxf_outline = self.cad_features['outline']
        if dxf_outline.size == 0:
            return messagebox.showerror("Alignment Error", "No outline found in the DXF file.")
        
        print("1. Finding keypoints (corners) on both outlines...")
        cad_mask, cad_transform = self._render_cad_to_image(dxf_outline, thickness=2)
        img_mask = np.zeros(self.cv_image.shape[:2], dtype="uint8")
        cv2.drawContours(img_mask, [image_contour], -1, 255, 1)

        cad_keypoints = cv2.goodFeaturesToTrack(cad_mask, maxCorners=100, qualityLevel=0.01, minDistance=20)
        img_keypoints = cv2.goodFeaturesToTrack(img_mask, maxCorners=100, qualityLevel=0.01, minDistance=20)

        if cad_keypoints is None or img_keypoints is None:
            return messagebox.showerror("Alignment Error", "Could not detect corners on one of the shapes.")

        # Convert keypoints from rendered CAD space back to original DXF space
        cad_kp_original_space = []
        if cad_keypoints is not None:
            for kp in cad_keypoints:
                x, y = kp.ravel()
                # Reverse the transformation from the render function
                orig_x = (x - cad_transform['padding']) / cad_transform['scale'] + cad_transform['x_min']
                orig_y = (y - cad_transform['padding']) / cad_transform['scale'] + cad_transform['y_min']
                cad_kp_original_space.append([orig_x, orig_y])
        cad_kp_original_space = np.array(cad_kp_original_space)

        print("2. Matching keypoints using local shape (Hu Moments)...")
        good_matches = []
        for i, kp1_orig in enumerate(cad_kp_original_space):
            hu_moments1 = self._get_hu_moments_for_keypoint(kp1_orig, dxf_outline)

            best_match_idx = -1
            min_dist = float('inf')
            for j, kp2 in enumerate(img_keypoints):
                hu_moments2 = self._get_hu_moments_for_keypoint(kp2.ravel(), image_contour.reshape(-1, 2))
                
                dist = cv2.matchShapes(hu_moments1, hu_moments2, cv2.CONTOURS_MATCH_I1, 0.0)
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = j
            
            good_matches.append((kp1_orig, img_keypoints[best_match_idx].ravel()))
        
        if len(good_matches) < 4:
            return messagebox.showerror("Alignment Error", f"Not enough good matches found ({len(good_matches)}/4).")

        src_pts = np.float32([match[0] for match in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([match[1] for match in good_matches]).reshape(-1, 1, 2)
        
        print("3. Calculating Homography matrix...")
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography_matrix is None:
            return messagebox.showerror("Alignment Error", "Could not compute the transformation matrix.")
        print("   Homography calculated successfully.")

        print("4. Creating blended result image...")
        transformed_outline = cv2.perspectiveTransform(dxf_outline.reshape(-1, 1, 2), homography_matrix)
        transformed_holes = [cv2.perspectiveTransform(h.reshape(-1, 1, 2), homography_matrix) for h in self.cad_features['holes']]
        transformed_creases = [cv2.perspectiveTransform(c.reshape(-1, 1, 2), homography_matrix) for c in self.cad_features['creases']]

        overlay = np.zeros_like(self.cv_image)
        cv2.polylines(overlay, [transformed_outline.astype(np.int32)], True, (255, 0, 0), 2)
        cv2.polylines(overlay, [h.astype(np.int32) for h in transformed_holes], True, (0, 255, 255), 2)
        cv2.polylines(overlay, [c.astype(np.int32) for c in transformed_creases], False, (255, 255, 0), 2)
        
        blended_image = cv2.addWeighted(self.cv_image, 0.7, overlay, 0.4, 0)
        
        self.display_image = blended_image
        self.update_image_display()
        self.root.update_idletasks()
        
        messagebox.showinfo("Inspection Complete", "The DXF features have been aligned and drawn on the image.")
        print("--- Inspection Complete ---")

    # --- REWRITTEN: Debug function for the new Hu Moments approach ---
    def run_feature_matching_debug(self):
        print("\n--- Running Hu Moments Matching Debug Visualization ---")
        
        # 1. Get contours
        image_contour = self.find_image_contour(self.cv_image)
        if image_contour is None: return messagebox.showerror("Debug Error", "No contour found in the image.")
        image_contour = image_contour.astype(np.int32)
        
        dxf_outline = self.cad_features['outline']
        if dxf_outline.size == 0: return messagebox.showerror("Debug Error", "No outline found in the DXF file.")
        
        # 2. Get keypoints (corners)
        cad_mask, _ = self._render_cad_to_image(dxf_outline, thickness=2)
        img_mask = np.zeros(self.cv_image.shape[:2], dtype="uint8")
        cv2.drawContours(img_mask, [image_contour], -1, 255, 1)
        
        cad_keypoints = cv2.goodFeaturesToTrack(cad_mask, maxCorners=100, qualityLevel=0.01, minDistance=20)
        img_keypoints = cv2.goodFeaturesToTrack(img_mask, maxCorners=100, qualityLevel=0.01, minDistance=20)

        # 3. Visualize the detected keypoints
        cad_kp_img = cv2.cvtColor(cad_mask, cv2.COLOR_GRAY2BGR)
        img_kp_img = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
        
        if cad_keypoints is not None:
            for kp in cad_keypoints:
                x, y = kp.ravel()
                cv2.circle(cad_kp_img, (int(x), int(y)), 5, (0, 255, 0), -1)
        if img_keypoints is not None:
             for kp in img_keypoints:
                x, y = kp.ravel()
                cv2.circle(img_kp_img, (int(x), int(y)), 5, (0, 255, 0), -1)

        cv2.imshow("Debug 1: Corners on CAD", cad_kp_img)
        cv2.imshow("Debug 2: Corners on Image", img_kp_img)
        messagebox.showinfo("Debug", "Showing detected corners. Close these windows to continue.")

        # 4. Perform matching and prepare for visualization
        # ... (This section duplicates logic from the main function for visualization)
        # ... You can expand this to draw the final match lines if needed.
        print("Debug finished. Check the corner detection images.")


    # --- REWRITTEN: Renders CAD and returns transform info ---
    def _render_cad_to_image(self, outline, holes=[], creases=[], padding=50, thickness=-1):
        all_points = [outline] + holes + creases
        if not any(p.size > 0 for p in all_points):
             return np.zeros((100, 100), dtype="uint8"), None
        v_stack = np.vstack([p for p in all_points if p.size > 0])
        x_min, y_min = np.min(v_stack, axis=0)
        x_max, y_max = np.max(v_stack, axis=0)
        cad_w, cad_h = x_max - x_min, y_max - y_min
        if cad_w == 0 or cad_h == 0: return np.zeros((100, 100), dtype="uint8"), None

        TARGET_WIDTH = 800
        scale = TARGET_WIDTH / cad_w
        canvas_w = int(TARGET_WIDTH + padding * 2)
        canvas_h = int(cad_h * scale + padding * 2)
        canvas = np.zeros((canvas_h, canvas_w), dtype="uint8")

        def transform(points):
            return ((points - [x_min, y_min]) * scale + padding).astype(int)

        cv2.polylines(canvas, [transform(outline)], True, 255, thickness)
        
        # Return the canvas AND the transformation info needed to go backwards
        transform_info = {'x_min': x_min, 'y_min': y_min, 'scale': scale, 'padding': padding}
        return canvas, transform_info

    # --- NEW: Helper function for Hu Moments matching ---
    def _get_hu_moments_for_keypoint(self, keypoint, contour, neighborhood_size=50):
        distances = np.linalg.norm(contour - keypoint, axis=1)
        closest_idx = np.argmin(distances)
        num_contour_pts = len(contour)
        half_size = neighborhood_size // 2
        
        indices = [(closest_idx - half_size + i + num_contour_pts) % num_contour_pts for i in range(neighborhood_size)]
        local_neighborhood = contour[indices]
        
        moments = cv2.moments(local_neighborhood)
        hu_moments = cv2.HuMoments(moments)
        
        # Log scale transform to make moments more comparable
        with np.errstate(divide='ignore', invalid='ignore'):
            log_hu = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
            log_hu[np.isneginf(log_hu)] = 0 # Handle log(0) case
        return log_hu

    def on_resize(self, event=None):
        if self.display_image is not None: self.update_image_display()

    def update_image_display(self):
        if self.display_image is None: return
        image_to_show = self.display_image
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
        paths, tolerance = [], 1e-6
        while segments:
            current_path = list(segments.pop(0))
            while True:
                extended = False
                for i, segment in enumerate(segments):
                    p_start, p_end = segment
                    if math.dist(current_path[-1], p_start) < tolerance:
                        current_path.append(p_end)
                        segments.pop(i)
                        extended = True
                        break
                    elif math.dist(current_path[-1], p_end) < tolerance:
                        current_path.append(p_start)
                        segments.pop(i)
                        extended = True
                        break
                if not extended: break
            paths.append(current_path)
        return paths

    def reset_image_state(self):
        self.image_path, self.cv_image, self.display_image = None, None, None
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
            return messagebox.showinfo("Visualize", "No CAD features were found in the loaded DXF file.")
        all_points_list = [p for p in [self.cad_features['outline']] + self.cad_features['holes'] + self.cad_features['creases'] if p.size > 0]
        if not all_points_list: return
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