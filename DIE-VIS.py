import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import ezdxf
from ezdxf import edgeminer, edgesmith # Correct, necessary imports
from ezdxf.math import Vec2 # Import Vec2 for type checking
import numpy as np
import os
import traceback

class InspectionApp:
    """
    A GUI application for inspecting cardboard features against a DXF file,
    based on the DIE-VIS paper.
    This version includes improved HSV contour detection and auto-scaling debug visualization.
    """
    def __init__(self, root):
        """
        Initializes the main application window and its widgets.
        
        Args:
            root: The root Tkinter window.
        """
        self.root = root
        self.root.title("DIE-VIS: Cardboard Inspection System - Corrected")
        self.root.geometry("1200x850")

        # --- State Variables ---
        self.image_path = None
        self.dxf_path = None
        self.cv_image = None
        self.result_image = None
        self.cad_features = {'outline': None, 'holes': [], 'creases': []}
        self.transformed_features = {'outline': None, 'holes': [], 'creases': []}
        self.homography_matrix = None
        
        # --- UI Colors and Styles ---
        self.colors = {
            "bg": "#2E2E2E",
            "fg": "#FFFFFF",
            "btn": "#4A4A4A",
            "btn_active": "#5A5A5A",
            "accent": "#007ACC",
            "success": "#28A745", # Green for 'Found'
            "fail": "#DC3545",    # Red for 'Missing'
            "info": "#17A2B8",     # Cyan for 'Expected Crease'
            "debug": "#FFC107"    # Amber for Debug
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
        self.btn_load_image = ttk.Button(control_frame, text="1. Load Cardboard Image", command=self.load_image)
        self.btn_load_image.pack(side=tk.LEFT, padx=5)

        self.btn_load_dxf = ttk.Button(control_frame, text="2. Load DXF File", command=self.load_dxf)
        self.btn_load_dxf.pack(side=tk.LEFT, padx=5)
        
        self.btn_debug_alignment = ttk.Button(control_frame, text="Debug Alignment", state=tk.DISABLED, command=self.run_debug_visualization)
        self.btn_debug_alignment.pack(side=tk.LEFT, padx=(20, 5))

        self.btn_run_inspection = ttk.Button(control_frame, text="Align & Inspect", state=tk.DISABLED, command=self.run_inspection)
        self.btn_run_inspection.pack(side=tk.LEFT, padx=5)
        
        self.lbl_image_status = tk.Label(control_frame, text="Image: None", bg=self.colors["bg"], fg=self.colors["fg"], padx=10)
        self.lbl_image_status.pack(side=tk.LEFT)
        
        self.lbl_dxf_status = tk.Label(control_frame, text="DXF: None", bg=self.colors["bg"], fg=self.colors["fg"], padx=10)
        self.lbl_dxf_status.pack(side=tk.LEFT)

        # --- Image Display ---
        self.image_label = tk.Label(self.image_frame, bg="#000000")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label.bind("<Configure>", self.on_resize)

    def setup_styles(self):
        """Configures the visual style for ttk widgets."""
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat",
                        background=self.colors["btn"], foreground=self.colors["fg"],
                        font=('Helvetica', 10, 'bold'))
        style.map("TButton",
                  background=[('active', self.colors["btn_active"]),
                              ('disabled', '#3D3D3D')])

    def load_image(self):
        """Opens a file dialog to load the cardboard image."""
        filepath = filedialog.askopenfilename(title="Select a Cardboard Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if not filepath: return
        self.image_path = filepath
        try:
            self.cv_image = cv2.imread(self.image_path)
            if self.cv_image is None: raise ValueError("OpenCV could not read the image file.")
            self.result_image = self.cv_image.copy()
            self.lbl_image_status.config(text=f"Image: {os.path.basename(self.image_path)}")
            self.update_image_display(self.cv_image)
            self._check_files_loaded()
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Failed to load image: {e}")
            self.reset_image_state()

    def load_dxf(self):
        """
        Opens a file dialog to load the DXF and processes it, ensuring all coordinates are 2D.
        """
        filepath = filedialog.askopenfilename(title="Select a DXF File", filetypes=[("DXF Files", "*.dxf"), ("All files", "*.*")])
        if not filepath: return
        self.dxf_path = filepath
        try:
            doc = ezdxf.readfile(self.dxf_path)
            msp = doc.modelspace()
            self.cad_features = {'outline': None, 'holes': [], 'creases': []}
            
            outline_entities = msp.query('*[layer=="OUTLINE"]')
            if outline_entities:
                edges = list(edgesmith.edges_from_entities_2d(outline_entities))
                if edges:
                    chain = edgeminer.find_sequential_chain(edges)
                    if chain:
                        contour_points = [v.vec2 for v in (edge.start for edge in chain)]
                        self.cad_features['outline'] = np.array(contour_points, dtype=np.float32)

            hole_entities = msp.query('*[layer=="HOLES"]')
            if hole_entities:
                remaining_edges = list(edgesmith.edges_from_entities_2d(hole_entities))
                
                while remaining_edges:
                    chain = edgeminer.find_sequential_chain(remaining_edges)
                    if not chain: break 
                    
                    if chain[0].start.isclose(chain[-1].end):
                        contour_points = [v.vec2 for v in (edge.start for edge in chain)]
                        self.cad_features['holes'].append({'points': np.array(contour_points, dtype=np.float32)})
                    
                    used_edges_set = set(chain)
                    remaining_edges = [edge for edge in remaining_edges if edge not in used_edges_set]

            crease_entities = msp.query('LINE[layer=="CREASES"]')
            for entity in crease_entities:
                start, end = entity.dxf.start, entity.dxf.end
                self.cad_features['creases'].append({'start': (start.x, start.y), 'end': (end.x, end.y)})

            if self.cad_features['outline'] is None:
                messagebox.showwarning("DXF Warning", "Could not find a connected outline on the 'OUTLINE' layer. Alignment will fail.")
            
            self.lbl_dxf_status.config(text=f"DXF: {os.path.basename(self.dxf_path)}")
            self._check_files_loaded()
            print("--- DXF Parse Complete ---")
            print(f"Features Loaded:\n- Outline Found: {self.cad_features['outline'] is not None}\n- Holes: {len(self.cad_features['holes'])}\n- Creases: {len(self.cad_features['creases'])}")

        except Exception as e:
            error_details = traceback.format_exc()
            print("---!!! DXF PARSING FAILED !!!---")
            print(error_details)
            messagebox.showerror("DXF Load Error", f"An unexpected error occurred: {e}")
            self.reset_dxf_state()

    def _check_files_loaded(self):
        """Enables the inspection and debug buttons only if both files are loaded."""
        if self.image_path and self.dxf_path and self.cad_features['outline'] is not None:
            self.btn_run_inspection.config(state=tk.NORMAL)
            self.btn_debug_alignment.config(state=tk.NORMAL)
        else:
            self.btn_run_inspection.config(state=tk.DISABLED)
            self.btn_debug_alignment.config(state=tk.DISABLED)

    def on_resize(self, event=None):
        """Handles window resizing to keep the image display scaled properly."""
        if self.result_image is not None: self.update_image_display(self.result_image)

    def update_image_display(self, image_to_show):
        """Resizes and displays the given OpenCV image in the Tkinter label."""
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

    # --- NEW: ROBUST CONTOUR FINDING WITH HSV --- 
    def find_image_contour(self, image):
        """
        Finds the main external contour of the cardboard in the image using HSV color segmentation.
        This is more robust to lighting and background variations.
        """
        # Convert the image from BGR to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define the HSV range for typical cardboard color. These values may need tuning.
        lower_bound_hsv = np.array([5, 40, 40])   # Lower Hue, Saturation, Value
        upper_bound_hsv = np.array([35, 255, 240])   # Upper H, S, V
        
        # Create a binary mask where pixels within the HSV range are white, and others are black.
        color_mask = cv2.inRange(hsv_image, lower_bound_hsv, upper_bound_hsv)

      
        
        # Find contours on the cleaned, final mask. RETR_EXTERNAL gets only the outer boundaries.
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Return the largest contour by area, which should be the cardboard box.
        return max(contours, key=cv2.contourArea).astype(np.float32) if contours else None

    # --- NEW: AUTO-SCALING CAD VISUALIZATION ---
    def run_debug_visualization(self):
        """
        Generates and displays auto-scaled visualizations for the CAD features and the
        detected image contour in separate windows for debugging.
        """
        print("--- Running Alignment Debug Visualization ---")
        
        # 1. Visualize the CAD data with auto-scaling
        if not self.cad_features['holes']:
             all_points = self.cad_features['outline']
        else:
             all_points = np.vstack([self.cad_features['outline']] + [h['points'] for h in self.cad_features['holes']])
        
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        
        # Define a fixed size for the debug canvas
        CANVAS_W, CANVAS_H = 800, 600
        padding = 50
        
        # Calculate scale factor to fit the drawing
        cad_w = x_max - x_min
        cad_h = y_max - y_min
        if cad_w == 0 or cad_h == 0: return # Avoid division by zero
        
        scale_x = (CANVAS_W - 2 * padding) / cad_w
        scale_y = (CANVAS_H - 2 * padding) / cad_h
        scale = min(scale_x, scale_y) # Use the smaller scale to maintain aspect ratio

        # Create a black canvas
        cad_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        
        # Function to transform and scale points
        def transform_pt(points):
            # Translate so top-left is at (0,0), then scale, then add padding
            return (((points - [x_min, y_min]) * scale) + padding).astype(np.int32)

        # Draw the outline in white
        outline_shifted = transform_pt(self.cad_features['outline'])
        cv2.polylines(cad_canvas, [outline_shifted], isClosed=True, color=(255, 255, 255), thickness=2)
        
        # Draw the holes in yellow
        for hole in self.cad_features['holes']:
            hole_shifted = transform_pt(hole['points'])
            cv2.polylines(cad_canvas, [hole_shifted], isClosed=True, color=(0, 255, 255), thickness=2)
        
        cv2.imshow("DEBUG: Parsed CAD Features (Auto-Scaled)", cad_canvas)

        # 2. Visualize the detected image contour
        image_contour = self.find_image_contour(self.cv_image)
        if image_contour is not None:
            debug_image = self.cv_image.copy()
            cv2.drawContours(debug_image, [image_contour.astype(np.int32)], -1, (255, 0, 255), 3)
            cv2.imshow("DEBUG: Detected Image Contour (HSV)", debug_image)
        else:
            messagebox.showwarning("Debug Warning", "Could not find any contour in the image using HSV method.")
            
        messagebox.showinfo("Debug", "Debug visualization windows have been opened. Press any key on those windows to close them.")

    def run_inspection(self):
        """Main function to run the full alignment and inspection pipeline."""
        print("--- Starting Inspection ---")
        self.result_image = self.cv_image.copy()

        image_contour = self.find_image_contour(self.cv_image)
        if image_contour is None:
            messagebox.showerror("Inspection Error", "Could not find the cardboard's outline in the image.")
            return

        self.homography_matrix, _ = cv2.findHomography(self.cad_features['outline'], image_contour, cv2.RANSAC, 5.0)
        if self.homography_matrix is None:
            messagebox.showerror("Alignment Error", "Could not calculate the transformation (homography). Use the 'Debug Alignment' button to compare the CAD shape and the detected image contour.")
            return
        
        self.transform_cad_features()
        print("Alignment successful.")

        self.detect_holes()
        self.detect_creases()

        self.update_image_display(self.result_image)
        messagebox.showinfo("Inspection Complete", "Inspection finished. Results are shown in the image.")

    def transform_cad_features(self):
        """Applies the calculated homography to all CAD features to map them into image space."""
        if self.homography_matrix is None: return
        self.transformed_features = {'outline': None, 'holes': [], 'creases': []}
        
        outline_pts = self.cad_features['outline'].reshape(-1, 1, 2)
        self.transformed_features['outline'] = cv2.perspectiveTransform(outline_pts, self.homography_matrix).reshape(-1, 2)
        
        for hole in self.cad_features['holes']:
            hole_pts = hole['points'].reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(hole_pts, self.homography_matrix)
            self.transformed_features['holes'].append({'points': transformed_points.reshape(-1, 2)})

        for crease in self.cad_features['creases']:
            crease_pts = np.array([crease['start'], crease['end']], dtype=np.float32).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(crease_pts, self.homography_matrix).reshape(-1, 2)
            self.transformed_features['creases'].append({'start': transformed_points[0], 'end': transformed_points[1]})

    def detect_holes(self):
        """Detects missing holes by comparing expected CAD holes to found image contours."""
        print("--- Detecting Holes ---")
        if not self.transformed_features['holes']: return
        
        mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.transformed_features['outline'].astype(np.int32)], 255)

        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

        detected_contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        found_holes_flags = [False] * len(self.transformed_features['holes'])

        for contour in detected_contours:
            M = cv2.moments(contour)
            if M["m00"] == 0: continue
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            for i, expected_hole in enumerate(self.transformed_features['holes']):
                if found_holes_flags[i]: continue
                if cv2.pointPolygonTest(expected_hole['points'].astype(np.int32), center, False) >= 0:
                    found_holes_flags[i] = True
                    break

        for i, was_found in enumerate(found_holes_flags):
            hole_poly = self.transformed_features['holes'][i]['points'].astype(np.int32)
            if was_found:
                cv2.polylines(self.result_image, [hole_poly], True, self.hex_to_bgr(self.colors["success"]), 3)
            else:
                center = tuple(np.mean(hole_poly, axis=0).astype(int))
                cv2.line(self.result_image, (center[0]-15, center[1]-15), (center[0]+15, center[1]+15), self.hex_to_bgr(self.colors["fail"]), 3)
                cv2.line(self.result_image, (center[0]-15, center[1]+15), (center[0]+15, center[1]-15), self.hex_to_bgr(self.colors["fail"]), 3)

    def detect_creases(self):
        """Detects missing creases using the custom pixel transition algorithm from the DIE-VIS paper."""
        print("--- Detecting Creases ---")
        if not self.transformed_features['creases']: return

        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        crease_evidence_map = self.find_pixel_transitions(gray, low_thresh=15, high_thresh=50)

        for crease in self.transformed_features['creases']:
            p1 = crease['start'].astype(int)
            p2 = crease['end'].astype(int)
            
            mask = np.zeros_like(gray)
            cv2.line(mask, tuple(p1), tuple(p2), 255, 10) 
            
            intersection = cv2.bitwise_and(crease_evidence_map, crease_evidence_map, mask=mask)
            
            evidence_pixels = np.count_nonzero(intersection)
            crease_length = np.linalg.norm(p1 - p2)
            density = evidence_pixels / crease_length if crease_length > 0 else 0
            
            if density > 0.4:
                 overlay = self.result_image.copy()
                 cv2.line(overlay, tuple(p1), tuple(p2), self.hex_to_bgr(self.colors["success"]), 4)
                 self.result_image = cv2.addWeighted(overlay, 0.6, self.result_image, 0.4, 0)
            else:
                cv2.line(self.result_image, tuple(p1), tuple(p2), self.hex_to_bgr(self.colors["fail"]), 3)

    def find_pixel_transitions(self, image, low_thresh, high_thresh):
        """
        Implementation of Algorithm 1 from the DIE-VIS paper. It scans rows and
        columns to find the characteristic shadow-highlight pattern of a crease.
        """
        h, w = image.shape
        crease_map = np.zeros((h, w), dtype=np.uint8)
        img_float = image.astype(float)

        for r in range(h):
            start_transition, is_down, is_up = False, False, False
            prev_val, prev_coord = 0, 0
            for c in range(1, w):
                p1, p2 = img_float[r, c-1], img_float[r, c]
                if p1 > p2: 
                    if not start_transition:
                        start_transition, is_down, is_up = True, True, False
                        prev_val, prev_coord = p1, c
                    elif is_up:
                        delta = abs(prev_val - p1)
                        if low_thresh <= delta <= high_thresh:
                            crease_map[r, (c + prev_coord) // 2] = 255
                        is_down, is_up, prev_val, prev_coord = True, False, p1, c
                elif p1 < p2:
                    if not start_transition:
                        start_transition, is_up, is_down = True, True, False
                        prev_val, prev_coord = p1, c
                    elif is_down:
                        delta = abs(prev_val - p1)
                        if low_thresh <= delta <= high_thresh:
                            crease_map[r, (c + prev_coord) // 2] = 255
                        is_up, is_down, prev_val, prev_coord = True, False, p1, c
        return crease_map


    def hex_to_bgr(self, hex_color):
        """Converts a hex color string like '#FFFFFF' to a BGR tuple."""
        h = hex_color.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))

    def reset_image_state(self):
        """Resets all variables related to the loaded image."""
        self.image_path, self.cv_image, self.result_image = None, None, None
        self.lbl_image_status.config(text="Image: None")
        self.update_image_display(None)
        self._check_files_loaded()

    def reset_dxf_state(self):
        """Resets all variables related to the loaded DXF file."""
        self.dxf_path = None
        self.cad_features = {'outline': None, 'holes': [], 'creases': []}
        self.lbl_dxf_status.config(text="DXF: None")
        self._check_files_loaded()

if __name__ == "__main__":
    root = tk.Tk()
    app = InspectionApp(root)
    root.mainloop()
