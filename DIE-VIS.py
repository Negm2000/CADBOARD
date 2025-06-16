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
        self.original_cv_image = None # Store original for resets
        self.cad_features = {'outline': np.array([]), 'holes': [], 'creases': []}
        
        # --- UI Colors and Styles ---
        self.colors = {
            "bg": "#2E2E2E", "fg": "#2C2C2C", "btn": "#4A4A4A",
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
        
        # MODIFIED: Debug button now targets the new geometric alignment logic
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
        style.configure("TButton", padding=6, relief="flat", background=self.colors["btn"], foreground=self.colors["fg"], font=('Helvetica', 10, 'bold'))
        style.map("TButton", background=[('active', self.colors["btn_active"]), ('disabled', '#3D3D3D')])

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
        all_loaded = self.image_path and self.dxf_path
        state = tk.NORMAL if all_loaded else tk.DISABLED
        self.btn_visualize.config(state=state)
        self.btn_inspect.config(state=state)
        self.btn_debug_alignment.config(state=state)

    def run_alignment_and_inspection(self):
        """
        Aligns the CAD model to the image object using a robust vertex matching
        strategy and overlays the result.
        """
        print("\n--- Starting Alignment & Inspection (Vertex Matching Methodology) ---")
        if self.cv_image is None or self.cad_features['outline'].size == 0:
            messagebox.showerror("Input Error", "Please load both an image and a DXF file with an 'OUTLINE' layer.")
            return

        homography_matrix, _, _ = self.align_shapes_by_vertices()
        if homography_matrix is None:
            messagebox.showerror("Alignment Error", "Could not compute the transformation matrix. Check debug output for details.")
            return
        
        print("1. Homography calculated successfully from vertex correspondences.")
        print("2. Transforming all DXF features to image space...")

        result_image = self.original_cv_image.copy()
        
        # Transform and draw the main outline
        transformed_outline = cv2.perspectiveTransform(self.cad_features['outline'].reshape(-1, 1, 2), homography_matrix)
        cv2.polylines(result_image, [transformed_outline.astype(np.int32)], True, (0, 255, 255), 2)

        # Transform and draw holes and creases
        for hole in self.cad_features['holes']:
            transformed_hole = cv2.perspectiveTransform(hole.reshape(-1, 1, 2), homography_matrix)
            cv2.polylines(result_image, [transformed_hole.astype(np.int32)], True, (255, 0, 255), 2)
        for crease in self.cad_features['creases']:
            transformed_crease = cv2.perspectiveTransform(crease.reshape(-1, 1, 2), homography_matrix)
            cv2.polylines(result_image, [transformed_crease.astype(np.int32)], False, (255, 255, 0), 2)
        
        print("3. Drawing final results...")
        self.update_image_display(result_image)
        messagebox.showinfo("Inspection Complete", "The DXF features have been aligned and drawn on the image using vertex matching.")
        print("--- Inspection Complete ---")

    def align_shapes_by_vertices(self):
        """
        Core alignment logic. Simplifies contours to vertices, finds the best
        correspondence, and computes the homography.
        """
        cad_contour = self.cad_features['outline'].reshape(-1, 1, 2)
        img_contour = self.find_image_contour(self.cv_image)
        if img_contour is None:
            print("Error: No contour found in the image using HSV masking.")
            return None, None, None

        epsilon_cad = 0.01 * cv2.arcLength(cad_contour, True)
        cad_vertices = cv2.approxPolyDP(cad_contour, epsilon_cad, True)
        epsilon_img = 0.01 * cv2.arcLength(img_contour, True)
        img_vertices = cv2.approxPolyDP(img_contour, epsilon_img, True)
        
        print(f"Simplified CAD to {len(cad_vertices)} vertices and Image to {len(img_vertices)} vertices.")

        if len(cad_vertices) < 4 or len(img_vertices) < 4 or len(cad_vertices) != len(img_vertices):
            messagebox.showwarning("Alignment Warning", f"Could not simplify shapes to an equal number of vertices (CAD: {len(cad_vertices)}, Image: {len(img_vertices)}). Alignment may fail.")
            return None, None, None

        img_vertices_ordered = self._find_best_vertex_correspondence(cad_vertices, img_vertices)
        if img_vertices_ordered is None:
            print("Error: Could not determine vertex correspondence.")
            return None, None, None

        homography_matrix, _ = cv2.findHomography(cad_vertices, img_vertices_ordered, cv2.RANSAC, 5.0)
        return homography_matrix, cad_vertices, img_vertices_ordered

    def _find_best_vertex_correspondence(self, cad_verts, img_verts):
        """
        Finds the optimal ordering of image vertices to match the CAD vertices
        by testing all circular shifts.
        """
        min_dist = float('inf')
        best_shift = -1
        for i in range(len(img_verts)):
            shifted_img_verts = np.roll(img_verts, i, axis=0)
            dist = np.sum((cad_verts - shifted_img_verts)**2)
            if dist < min_dist:
                min_dist = dist
                best_shift = i
        return np.roll(img_verts, best_shift, axis=0) if best_shift != -1 else None

    def run_alignment_debug(self):
        """
        Generates and displays intermediate images from the vertex matching
        process to help diagnose alignment failures.
        """
        print("\n--- Running Alignment Debug Visualization ---")


        # 1. Visualize the detected and simplified image contour
        img_contour = self.find_image_contour(self.cv_image)
        if img_contour is None: return messagebox.showerror("Debug Error", "No contour found in image.")
        
        epsilon_img = 0.01 * cv2.arcLength(img_contour, True)
        img_vertices = cv2.approxPolyDP(img_contour, epsilon_img, True)

        debug_img_contours = self.original_cv_image.copy()
        cv2.drawContours(debug_img_contours, [img_contour], -1, (255, 0, 0), 2)
        for pt in img_vertices:
            cv2.circle(debug_img_contours, tuple(pt[0]), 7, (0, 0, 255), -1)
        cv2.imshow("Debug 2: Image Contour and Vertices", debug_img_contours)
        
        # 2. Visualize final vertex correspondences
        _, _, img_v_final = self.align_shapes_by_vertices()
        if img_v_final is None: return

        correspondence_img = self.original_cv_image.copy()
        for i, pt_array in enumerate(img_v_final):
            pt_img = tuple(pt_array.ravel().astype(int))
            cv2.circle(correspondence_img, pt_img, 8, (0, 255, 0), -1)
            cv2.putText(correspondence_img, str(i), (pt_img[0] + 5, pt_img[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("Debug 3: Matched Image Vertices (Numbered)", correspondence_img)
        messagebox.showinfo("Debug", "Debug windows opened. Examine them to diagnose the alignment process.")

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

    # --- UNCHANGED AS REQUESTED: Using original HSV-based contour detection ---
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
        
    def run_visualization(self):
        """
        Displays visualizations of both the parsed CAD features and the
        detected image contour in separate windows for comparison.
        """
        print("--- Running Input Visualization ---")

        # Visualize Parsed CAD Data
        cad_features_found = self.cad_features['outline'].size > 0 or \
                             any(self.cad_features['holes']) or \
                             any(self.cad_features['creases'])

        if not cad_features_found:
            messagebox.showinfo("Visualize", "No CAD features were found in the loaded DXF file.")
        else:
            all_points_list = []
            if self.cad_features['outline'].size > 0: all_points_list.append(self.cad_features['outline'])
            if self.cad_features['holes']: all_points_list.extend(self.cad_features['holes'])
            if self.cad_features['creases']: all_points_list.extend(self.cad_features['creases'])

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
                
                if self.cad_features['outline'].size > 0:
                    cv2.polylines(cad_canvas, [transform_pt(self.cad_features['outline'])], True, (255, 255, 255), 1)
                cv2.polylines(cad_canvas, [transform_pt(p) for p in self.cad_features['holes']], True, (0, 255, 255), 1)
                cv2.polylines(cad_canvas, [transform_pt(p) for p in self.cad_features['creases']], False, (255, 255, 0), 1)
                cv2.imshow("DEBUG: Parsed CAD", cad_canvas)

        # Visualize Detected Image Contour
        image_contour = self.find_image_contour(self.cv_image)
        if image_contour is not None:
            debug_image = self.original_cv_image.copy()
            cv2.drawContours(debug_image, [image_contour.astype(np.int32)], -1, (255, 0, 255), 3)
            cv2.imshow("DEBUG: Detected Image Contour", debug_image)
        else:
            messagebox.showinfo("Visualize", "No contour detected in image with current HSV settings.")
        
        messagebox.showinfo("Debug", "Debug windows opened. Press any key on them to close.")


if __name__ == "__main__":
    root = tk.Tk()
    app = InspectionApp(root)
    root.mainloop()
