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
    A GUI application for inspecting cardboard features against a DXF file.
    """
    def __init__(self, root):
        """
        Initializes the main application window and its widgets.
        
        Args:
            root: The root Tkinter window.
        """
        self.root = root
        self.root.title("DIE-VIS: Cardboard Inspection System")
        self.root.geometry("1200x850")

        # --- State Variables ---
        self.image_path = None
        self.dxf_path = None
        self.cv_image = None
        self.result_image = None
        # This dictionary will store all candidate chains found for debugging purposes
        self.candidate_chains = []
        self.cad_features = {'outline': None, 'holes': [], 'creases': []}
        self.transformed_features = {'outline': None, 'holes': [], 'creases': []}
        self.homography_matrix = None
        
        # --- UI Colors and Styles ---
        self.colors = {
            "bg": "#2E2E2E", "fg": "#FFFFFF", "btn": "#4A4A4A",
            "btn_active": "#5A5A5A", "accent": "#007ACC", "success": "#28A745",
            "fail": "#DC3545", "info": "#17A2B8", "debug": "#FFC107"
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
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background=self.colors["btn"], foreground=self.colors["fg"], font=('Helvetica', 10, 'bold'))
        style.map("TButton", background=[('active', self.colors["btn_active"]), ('disabled', '#3D3D3D')])

    def load_image(self):
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
        Loads and processes a DXF file by extracting all entities from each layer
        and converting them to point sequences. This approach correctly handles
        complex entities like arcs, splines, and polylines.
        """
        filepath = filedialog.askopenfilename(
            title="Select a DXF File", 
            filetypes=[("DXF Files", "*.dxf"), ("All files", "*.*")]
        )
        if not filepath: 
            return
        
        self.dxf_path = filepath
        
        try:
            print("--- Starting DXF Parsing (Direct Entity Processing) ---")
            doc = ezdxf.readfile(self.dxf_path)
            msp = doc.modelspace()
            
            # Reset data-holding variables
            self.candidate_chains = []
            self.cad_features = {'outline': None, 'holes': [], 'creases': []}

            # --- PROCESS OUTLINE LAYER ---
            print("1. Processing OUTLINE layer...")
            outline_entities = msp.query('*[layer=="OUTLINE"]')
            if not outline_entities:
                raise ValueError("No entities found on the 'OUTLINE' layer.")
            
            outline_points = []
            
            for entity in outline_entities:
                entity_points = self._extract_entity_points(entity)
                if entity_points:
                    outline_points.extend(entity_points)
            
            if not outline_points:
                raise ValueError("Could not extract any points from OUTLINE layer entities.")
            
            # Remove duplicate consecutive points
            outline_points = self._remove_consecutive_duplicates(outline_points)
            
            if len(outline_points) < 3:
                raise ValueError(f"OUTLINE layer produced only {len(outline_points)} unique points (minimum 3 required).")
            
            self.cad_features['outline'] = np.array(outline_points, dtype=np.float32)
            print(f"   Extracted {len(outline_points)} points from OUTLINE layer")

            # --- PROCESS HOLES LAYER ---
            print("2. Processing HOLES layer...")
            hole_entities = msp.query('*[layer=="HOLES"]')
            if hole_entities:
                # Group hole entities by proximity or connectivity
                hole_groups = self._group_hole_entities(hole_entities)
                
                for group in hole_groups:
                    hole_points = []
                    for entity in group:
                        entity_points = self._extract_entity_points(entity)
                        if entity_points:
                            hole_points.extend(entity_points)
                    
                    if hole_points:
                        hole_points = self._remove_consecutive_duplicates(hole_points)
                        if len(hole_points) >= 3:
                            self.cad_features['holes'].append({
                                'points': np.array(hole_points, dtype=np.float32)
                            })
                
                print(f"   Found {len(self.cad_features['holes'])} hole(s)")

            # --- PROCESS CREASES LAYER ---
            print("3. Processing CREASES layer...")
            crease_entities = msp.query('*[layer=="CREASES"]')
            for entity in crease_entities:
                if entity.dxftype() == 'LINE':
                    start, end = entity.dxf.start, entity.dxf.end
                    self.cad_features['creases'].append({
                        'start': (start.x, start.y), 
                        'end': (end.x, end.y)
                    })
                else:
                    # Handle other crease entity types if needed
                    entity_points = self._extract_entity_points(entity)
                    if len(entity_points) >= 2:
                        self.cad_features['creases'].append({
                            'start': entity_points[0], 
                            'end': entity_points[-1]
                        })
            
            print(f"   Found {len(self.cad_features['creases'])} crease(s)")

            # Update UI
            self.lbl_dxf_status.config(text=f"DXF: {os.path.basename(self.dxf_path)}")
            self._check_files_loaded()
            print("--- DXF Parse Complete ---")

        except Exception as e:
            error_details = traceback.format_exc()
            messagebox.showerror("DXF Load Error", f"An error occurred during DXF processing:\n\n{e}\n\nDetails:\n{error_details}")
            self.reset_dxf_state()

    def _extract_entity_points(self, entity):
        """
        Extract points from a DXF entity, handling different entity types.
        Returns a list of (x, y) tuples.
        """
        points = []
        
        try:
            if entity.dxftype() == 'LINE':
                start, end = entity.dxf.start, entity.dxf.end
                points = [(start.x, start.y), (end.x, end.y)]
                
            elif entity.dxftype() == 'POLYLINE':
                points = [(vertex.dxf.location.x, vertex.dxf.location.y) for vertex in entity.vertices]
                
            elif entity.dxftype() == 'LWPOLYLINE':
                points = [(point[0], point[1]) for point in entity.get_points('xy')]
                
            elif entity.dxftype() == 'ARC':
                # Convert arc to polyline approximation
                start_angle = math.radians(entity.dxf.start_angle)
                end_angle = math.radians(entity.dxf.end_angle)
                center = entity.dxf.center
                radius = entity.dxf.radius
                
                # Handle angle wrap-around
                if end_angle < start_angle:
                    end_angle += 2 * math.pi
                
                # Create arc approximation with sufficient resolution
                num_segments = max(8, int(abs(end_angle - start_angle) * radius / 2))
                angle_step = (end_angle - start_angle) / num_segments
                
                for i in range(num_segments + 1):
                    angle = start_angle + i * angle_step
                    x = center.x + radius * math.cos(angle)
                    y = center.y + radius * math.sin(angle)
                    points.append((x, y))
                    
            elif entity.dxftype() == 'CIRCLE':
                # Convert circle to polyline approximation
                center = entity.dxf.center
                radius = entity.dxf.radius
                num_segments = max(16, int(2 * math.pi * radius / 4))  # Adaptive resolution
                
                for i in range(num_segments):
                    angle = 2 * math.pi * i / num_segments
                    x = center.x + radius * math.cos(angle)
                    y = center.y + radius * math.sin(angle)
                    points.append((x, y))
                # Close the circle
                if points:
                    points.append(points[0])
                    
            elif entity.dxftype() == 'SPLINE':
                # Use ezdxf's built-in spline flattening
                try:
                    flattened = list(entity.flattening(0.1))  # 0.1 is the distance tolerance
                    points = [(point.x, point.y) for point in flattened]
                except:
                    # Fallback: use control points
                    points = [(point.x, point.y) for point in entity.control_points]
                    
            elif entity.dxftype() == 'ELLIPSE':
                # Convert ellipse to polyline approximation
                center = entity.dxf.center
                major_axis = entity.dxf.major_axis
                ratio = entity.dxf.ratio
                start_param = entity.dxf.start_param
                end_param = entity.dxf.end_param
                
                # Create ellipse approximation
                if end_param < start_param:
                    end_param += 2 * math.pi
                
                param_range = end_param - start_param
                num_segments = max(16, int(param_range * 8))
                param_step = param_range / num_segments
                
                major_length = major_axis.magnitude
                minor_length = major_length * ratio
                
                # Get major axis angle
                major_angle = math.atan2(major_axis.y, major_axis.x)
                
                for i in range(num_segments + 1):
                    param = start_param + i * param_step
                    # Parametric ellipse point
                    local_x = major_length * math.cos(param)
                    local_y = minor_length * math.sin(param)
                    
                    # Rotate by major axis angle and translate to center
                    x = center.x + local_x * math.cos(major_angle) - local_y * math.sin(major_angle)
                    y = center.y + local_x * math.sin(major_angle) + local_y * math.cos(major_angle)
                    points.append((x, y))
                    
            else:
                print(f"   Warning: Unhandled entity type '{entity.dxftype()}' - skipping")
                
        except Exception as e:
            print(f"   Warning: Error processing {entity.dxftype()} entity: {e}")
        
        return points

    def _remove_consecutive_duplicates(self, points, tolerance=1e-6):
        """Remove consecutive duplicate points within tolerance."""
        if not points:
            return points
        
        filtered = [points[0]]
        for point in points[1:]:
            last_point = filtered[-1]
            distance = math.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
            if distance > tolerance:
                filtered.append(point)
        
        return filtered

    def _group_hole_entities(self, hole_entities):
        """
        Group hole entities that belong to the same hole.
        For now, return each entity as its own group.
        More sophisticated grouping could be added based on proximity.
        """
        # Simple approach: each entity is its own group
        # This works well when each hole is a single entity (circle, polyline, etc.)
        return [[entity] for entity in hole_entities]

    def _check_files_loaded(self):
        # Use .get() for safer dictionary access
        if self.image_path and self.dxf_path and self.cad_features.get('outline') is not None:
            self.btn_run_inspection.config(state=tk.NORMAL)
            self.btn_debug_alignment.config(state=tk.NORMAL)
        else:
            self.btn_run_inspection.config(state=tk.DISABLED)
            self.btn_debug_alignment.config(state=tk.DISABLED)

    def on_resize(self, event=None):
        if self.result_image is not None: self.update_image_display(self.result_image)

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
        lower_bound_hsv = np.array([5, 50, 50])
        upper_bound_hsv = np.array([30, 255, 255])
        color_mask = cv2.inRange(hsv_image, lower_bound_hsv, upper_bound_hsv)
        
        kernel = np.ones((5,5),np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea).astype(np.float32) if contours else None

    def run_debug_visualization(self):
        print("--- Running Alignment Debug Visualization ---")
        if self.cad_features.get('outline') is None: return

        # --- Visualize Parsed CAD Data ---
        # Collect all points from all features for auto-scaling
        all_points_list = [self.cad_features['outline']]
        if self.cad_features['holes']:
             all_points_list.extend([h['points'] for h in self.cad_features['holes']])
        all_points = np.vstack(all_points_list)

        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        
        CANVAS_W, CANVAS_H = 800, 600
        padding = 50
        cad_w, cad_h = x_max - x_min, y_max - y_min
        if cad_w == 0 or cad_h == 0: return
        
        scale = min((CANVAS_W - 2 * padding) / cad_w, (CANVAS_H - 2 * padding) / cad_h)
        cad_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        
        def transform_pt(points):
            transformed = (points - [x_min, y_min]) * [scale, -scale]
            transformed += [padding, (CANVAS_H - padding)]
            return transformed.astype(np.int32)

        outline_shifted = transform_pt(self.cad_features['outline'])
        # Since it's an open path, use isClosed=False
        cv2.polylines(cad_canvas, [outline_shifted], isClosed=False, color=(255, 255, 255), thickness=1)
        
        # Draw holes in yellow
        for hole in self.cad_features['holes']:
            hole_shifted = transform_pt(hole['points'])
            cv2.polylines(cad_canvas, [hole_shifted], isClosed=False, color=(0, 255, 255), thickness=1)
            
        cv2.imshow("DEBUG: Parsed CAD", cad_canvas)

        # --- Visualize Image Contour ---
        image_contour = self.find_image_contour(self.cv_image)
        if image_contour is not None:
            debug_image = self.cv_image.copy()
            cv2.drawContours(debug_image, [image_contour.astype(np.int32)], -1, (255, 0, 255), 3)
            cv2.imshow("DEBUG: Detected Image Contour", debug_image)
            
        messagebox.showinfo("Debug", "Debug windows opened. Press any key on them to close.")

    def run_inspection(self):
        print("--- Starting Inspection ---")
        self.result_image = self.cv_image.copy()

        image_contour = self.find_image_contour(self.cv_image)
        if image_contour is None:
            messagebox.showerror("Inspection Error", "Could not find the cardboard's outline in the image.")
            return

        cad_outline_for_homography = self.cad_features['outline']
        self.homography_matrix, _ = cv2.findHomography(cad_outline_for_homography, image_contour, cv2.RANSAC, 5.0)
        
        if self.homography_matrix is None:
            messagebox.showerror("Alignment Error", "Could not calculate transformation (homography). Use 'Debug Alignment' to check CAD and image contours.")
            return
        
        print("Alignment successful.")
        self.transform_cad_features()
        self.detect_holes()
        self.detect_creases()
        
        if self.transformed_features.get('outline') is not None:
            aligned_outline = self.transformed_features['outline'].astype(np.int32)
            # Draw the aligned path. It may or may not be closed.
            cv2.polylines(self.result_image, [aligned_outline], False, self.hex_to_bgr(self.colors["accent"]), 2)

        self.update_image_display(self.result_image)
        messagebox.showinfo("Inspection Complete", "Inspection finished.")

    def transform_cad_features(self):
        if self.homography_matrix is None: return
        self.transformed_features = {'outline': None, 'holes': [], 'creases': []}
        
        outline_pts = self.cad_features['outline'].reshape(-1, 1, 2)
        transformed_outline = cv2.perspectiveTransform(outline_pts, self.homography_matrix)
        if transformed_outline is not None:
             self.transformed_features['outline'] = transformed_outline.reshape(-1, 2)
        
        for hole in self.cad_features['holes']:
            hole_pts = hole['points'].reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(hole_pts, self.homography_matrix)
            if transformed_points is not None:
                self.transformed_features['holes'].append({'points': transformed_points.reshape(-1, 2)})

        for crease in self.cad_features['creases']:
            crease_pts = np.array([crease['start'], crease['end']], dtype=np.float32).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(crease_pts, self.homography_matrix)
            if transformed_points is not None:
                start_pt, end_pt = transformed_points[0][0], transformed_points[1][0]
                self.transformed_features['creases'].append({'start': start_pt, 'end': end_pt})

    def detect_holes(self):
        # This function requires a closed outline to create a mask. If our outline is open,
        # we can attempt to close it for masking purposes or search the whole image.
        print("--- Detecting Holes ---")
        if not self.transformed_features.get('holes'): return
        
        mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
        if self.transformed_features.get('outline') is not None and len(self.transformed_features['outline']) > 2:
            # fillPoly can handle open contours by implicitly closing them for the fill.
            cv2.fillPoly(mask, [self.transformed_features['outline'].astype(np.int32)], 255)
        else:
            mask.fill(255) # Fallback to searching the whole image

        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

        detected_contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        found_holes_flags = [False] * len(self.transformed_features['holes'])

        for contour in detected_contours:
            if cv2.contourArea(contour) < 20: continue
            M = cv2.moments(contour)
            if M["m00"] == 0: continue
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            for i, expected_hole in enumerate(self.transformed_features['holes']):
                if found_holes_flags[i] or not expected_hole.get('points').any(): continue
                if cv2.pointPolygonTest(expected_hole['points'].astype(np.int32), center, False) >= 0:
                    found_holes_flags[i] = True
                    break

        for i, was_found in enumerate(found_holes_flags):
            hole_poly = self.transformed_features['holes'][i]['points'].astype(np.int32)
            color = self.hex_to_bgr(self.colors["success"] if was_found else self.colors["fail"])
            cv2.polylines(self.result_image, [hole_poly], True, color, 2)
            if not was_found:
                center = tuple(np.mean(hole_poly, axis=0).astype(int))
                cv2.line(self.result_image, (center[0]-10, center[1]-10), (center[0]+10, center[1]+10), color, 2)
                cv2.line(self.result_image, (center[0]-10, center[1]+10), (center[0]+10, center[1]-10), color, 2)

    def detect_creases(self):
        print("--- Detecting Creases ---")
        if not self.transformed_features.get('creases'): return

        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny_edges = cv2.Canny(blurred, 50, 150)

        for crease in self.transformed_features['creases']:
            p1, p2 = crease['start'].astype(int), crease['end'].astype(int)
            mask = np.zeros_like(gray)
            cv2.line(mask, tuple(p1), tuple(p2), 255, 10) # 10px search radius
            
            intersection = cv2.bitwise_and(canny_edges, canny_edges, mask=mask)
            evidence_pixels, crease_length = np.count_nonzero(intersection), np.linalg.norm(p1 - p2)
            density = evidence_pixels / crease_length if crease_length > 0 else 0
            
            crease_found = density > 0.3
            color = self.hex_to_bgr(self.colors["success"] if crease_found else self.colors["fail"])
            cv2.line(self.result_image, tuple(p1), tuple(p2), color, 2)
            
            if not crease_found:
                center = (p1 + p2) // 2
                cv2.line(self.result_image, (center[0]-8, center[1]-8), (center[0]+8, center[1]+8), color, 2)
                cv2.line(self.result_image, (center[0]-8, center[1]+8), (center[0]+8, center[1]-8), color, 2)

    def hex_to_bgr(self, hex_color):
        h = hex_color.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))

    def reset_image_state(self):
        self.image_path, self.cv_image, self.result_image = None, None, None
        self.lbl_image_status.config(text="Image: None")
        self.image_label.config(image='')
        self._check_files_loaded()

    def reset_dxf_state(self):
        self.dxf_path = None
        self.candidate_chains = []
        self.cad_features = {'outline': None, 'holes': [], 'creases': []}
        self.lbl_dxf_status.config(text="DXF: None")
        self._check_files_loaded()

if __name__ == "__main__":
    root = tk.Tk()
    app = InspectionApp(root)
    root.mainloop()
