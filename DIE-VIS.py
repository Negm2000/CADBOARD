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
    """
    A simplified GUI application to visualize a DXF file and an image contour.
    """
    def __init__(self, root):

        self.root = root
        self.root.title("DIE-VIS: Visualizer")
        self.root.geometry("1200x850")

        # --- State Variables ---
        self.image_path = None
        self.dxf_path = None
        self.cv_image = None
        # Simplified feature storage
        self.cad_features = {'outline': np.array([]), 'holes': []}
        
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
        
        self.btn_visualize = ttk.Button(control_frame, text="Visualize", state=tk.DISABLED, command=self.run_visualization)
        self.btn_visualize.pack(side=tk.LEFT, padx=(20, 5))
        
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
        """
        Loads a DXF, extracts geometry as segments, and assembles them into
        continuous, ordered paths using the path assembly algorithm.
        """
        filepath = filedialog.askopenfilename(title="Select a DXF File", filetypes=[("DXF Files", "*.dxf"), ("All files", "*.*")])
        if not filepath: return
        self.dxf_path = filepath
        try:
            print("--- Starting DXF Parsing and Path Assembly ---")
            doc = ezdxf.readfile(self.dxf_path)
            msp = doc.modelspace()
            
            # Reset data-holding variables
            self.cad_features = {'outline': np.array([]), 'holes': []}

            # --- Process OUTLINE layer ---
            print("1. Processing OUTLINE layer segments...")
            outline_segments = []
            for entity in msp.query('*[layer=="OUTLINE"]'):
                points = self._extract_entity_points(entity)
                # Convert list of points [p1, p2, p3] to segments [(p1,p2), (p2,p3)]
                for i in range(len(points) - 1):
                    outline_segments.append((points[i], points[i+1]))
            
            if outline_segments:
                print(f"   Found {len(outline_segments)} raw outline segments. Assembling...")
                assembled_outlines = self._assemble_paths(outline_segments)
                if assembled_outlines:
                    # Assume the longest assembled path is the main outline
                    main_outline = max(assembled_outlines, key=len)
                    self.cad_features['outline'] = np.array(main_outline, dtype=np.float32)
                    print(f"   Assembled outline into a single path with {len(main_outline)} points.")

            # --- Process HOLES layer ---
            print("2. Processing HOLES layer segments...")
            # This logic assumes each entity on the HOLES layer is a separate hole.
            # For complex holes made of multiple entities, a grouping step would be needed here.
            for entity in msp.query('*[layer=="HOLES"]'):
                hole_segments = []
                points = self._extract_entity_points(entity)
                for i in range(len(points) - 1):
                    hole_segments.append((points[i], points[i+1]))
                
                if hole_segments:
                    assembled_holes = self._assemble_paths(hole_segments)
                    for path in assembled_holes:
                        self.cad_features['holes'].append(np.array(path, dtype=np.float32))

            print(f"   Found and assembled {len(self.cad_features['holes'])} hole entities.")
            
            self.lbl_dxf_status.config(text=f"DXF: {os.path.basename(self.dxf_path)}")
            self._check_files_loaded()
            print("--- DXF Parse Complete ---")

        except Exception as e:
            error_details = traceback.format_exc()
            messagebox.showerror("DXF Load Error", f"An error occurred during DXF processing:\n\n{e}\n\nDetails:\n{error_details}")
            self.reset_dxf_state()

    def _assemble_paths(self, segments):
        """
        Assembles a list of unordered segments into one or more continuous paths.
        This implements the "Path Assembly and Vertex Ordering" algorithm.
        """
        if not segments:
            return []

        # Create a copy to modify
        segments = list(segments)
        paths = []
        tolerance = 1e-6

        # While there are still segments to process
        while segments:
            # 1. Start a New Path
            current_path = list(segments.pop(0))

            # 2. Extend the Path
            while True:
                extended = False
                # Search for a segment that connects to the end of our path
                for i, segment in enumerate(segments):
                    p_start, p_end = segment
                    last_point_in_path = current_path[-1]

                    # Check for a connection (forward or reverse)
                    if math.dist(last_point_in_path, p_start) < tolerance:
                        current_path.append(p_end)
                        segments.pop(i)
                        extended = True
                        break # Restart search with the new end point
                    elif math.dist(last_point_in_path, p_end) < tolerance:
                        current_path.append(p_start)
                        segments.pop(i)
                        extended = True
                        break # Restart search with the new end point
                
                # If we went through all segments and found no connection, the path is done
                if not extended:
                    break
            
            # 3. Store the Completed Path
            paths.append(current_path)
            
        return paths

    def _check_files_loaded(self):

        if self.image_path and self.dxf_path:
            self.btn_visualize.config(state=tk.NORMAL)
        else:
            self.btn_visualize.config(state=tk.DISABLED)

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

    def run_visualization(self):

        print("--- Running Visualization ---")
        
        # --- 1. Visualize Parsed CAD Data ---
        if self.cad_features['outline'].size > 0 or self.cad_features['holes']:
            all_points_list = []
            if self.cad_features['outline'].size > 0:
                all_points_list.append(self.cad_features['outline'])
            if self.cad_features['holes']:
                all_points_list.extend(self.cad_features['holes'])
            
            all_points = np.vstack(all_points_list)
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)
            
            CANVAS_W, CANVAS_H = 800, 600
            padding = 50
            cad_w, cad_h = x_max - x_min, y_max - y_min
            
            if cad_w > 0 and cad_h > 0:
                scale = min((CANVAS_W - 2 * padding) / cad_w, (CANVAS_H - 2 * padding) / cad_h)
                cad_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
                
                def transform_pt(points):
                    transformed = (points - [x_min, y_min]) * [scale, -scale]
                    transformed += [padding, (CANVAS_H - padding)]
                    return transformed.astype(np.int32)

                outline_shifted = transform_pt(self.cad_features['outline'])
                cv2.polylines(cad_canvas, [outline_shifted], isClosed=False, color=(255, 255, 255), thickness=1)
                
                for hole_points in self.cad_features['holes']:
                    hole_shifted = transform_pt(hole_points)
                    cv2.polylines(cad_canvas, [hole_shifted], isClosed=True, color=(0, 255, 255), thickness=1)
                    
                cv2.imshow("DEBUG: Parsed CAD", cad_canvas)

        # --- 2. Visualize Image Contour ---
        image_contour = self.find_image_contour(self.cv_image)
        if image_contour is not None:
            debug_image = self.cv_image.copy()
            cv2.drawContours(debug_image, [image_contour.astype(np.int32)], -1, (255, 0, 255), 3)
            cv2.imshow("DEBUG: Detected Image Contour", debug_image)
            
        messagebox.showinfo("Debug", "Debug windows opened. Press any key on them to close.")

    def find_image_contour(self, image):

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound_hsv = np.array([5, 50, 50])
        upper_bound_hsv = np.array([30, 255, 255])
        color_mask = cv2.inRange(hsv_image, lower_bound_hsv, upper_bound_hsv)
        
        kernel = np.ones((5,5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea).astype(np.float32) if contours else None

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

            else:
                print(f"   Info: Skipping unhandled entity type '{dxf_type}'")

        except Exception as e:
            print(f"   Warning: Could not process entity {dxf_type}: {e}")
        
        return points

    def _remove_consecutive_duplicates(self, points, tolerance=1e-6):

        if not points:
            return []
        filtered = [points[0]]
        for point in points[1:]:
            last_point = filtered[-1]
            dist_sq = (point[0] - last_point[0])**2 + (point[1] - last_point[1])**2
            if dist_sq > tolerance**2:
                filtered.append(point)
        return filtered

    def reset_image_state(self):

        self.image_path, self.cv_image = None, None
        self.lbl_image_status.config(text="Image: None")
        self.image_label.config(image='')
        self._check_files_loaded()

    def reset_dxf_state(self):

        self.dxf_path = None
        self.cad_features = {'outline': np.array([]), 'holes': []}
        self.lbl_dxf_status.config(text="DXF: None")
        self._check_files_loaded()


if __name__ == "__main__":
    root = tk.Tk()
    app = InspectionApp(root)
    root.mainloop()