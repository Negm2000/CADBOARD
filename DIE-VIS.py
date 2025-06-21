import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import ezdxf
import math
import numpy as np
import os
import traceback
import json
from scipy.spatial import KDTree

try:
    from pyueye import ueye
except ImportError:
    print("WARNING: pyueye library not found. IDS Camera functionality will be disabled.")
    class MockUeye:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                if name == 'is_InitCamera': return -1
                return 0
            return method
    ueye = MockUeye()


class InspectionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("DIE-VIS: Visualizer & Inspector (Hybrid Geometric Engine)")
        self.root.geometry("1300x950") # Increased height for new controls

        # --- State Variables ---
        self.image_path = None
        self.dxf_path = None
        self.cv_image = None
        self.original_cv_image = None 
        self.cad_features = {'outline': np.array([]), 'holes': [], 'creases': []}
        self.last_transform_matrix = None
        self.last_transform_type = None
        self.settings_file = "settings.json"
        
        # --- UI Colors and Styles ---
        self.colors = {
            "bg": "#2E2E2E", "fg": "#FFFFFF", "btn": "#4A4A4A",
            "btn_active": "#5A5A5A", "accent": "#00AACC", "accent_fg": "#FFFFFF"
        }
        self.root.configure(bg=self.colors["bg"])
        self.setup_styles()

        # --- UI Layout ---
        main_frame = tk.Frame(root, bg=self.colors["bg"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_controls_frame = tk.Frame(main_frame, bg=self.colors["bg"])
        top_controls_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))
        
        control_frame_1 = tk.Frame(top_controls_frame, bg=self.colors["bg"])
        control_frame_1.pack(fill=tk.X, pady=(0, 5))
        control_frame_2 = tk.Frame(top_controls_frame, bg=self.colors["bg"])
        control_frame_2.pack(fill=tk.X, pady=(0, 5))
        control_frame_3 = tk.Frame(top_controls_frame, bg=self.colors["bg"])
        control_frame_3.pack(fill=tk.X, pady=(0, 5))
        control_frame_crease = tk.Frame(top_controls_frame, bg=self.colors["bg"])
        control_frame_crease.pack(fill=tk.X, pady=(0,5))
        control_frame_4 = tk.Frame(top_controls_frame, bg=self.colors["bg"])
        control_frame_4.pack(fill=tk.X, pady=(5,10))
        
        self.image_frame = tk.Frame(main_frame, bg="#000000", relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # --- Populate Control Widgets ---
        # Frame 1: Buttons
        self.btn_load_image = ttk.Button(control_frame_1, text="1a. Load Image", command=self.load_image)
        self.btn_load_image.pack(side=tk.LEFT, padx=5)
        self.btn_capture_ids = ttk.Button(control_frame_1, text="1b. Capture IDS", command=self.capture_from_ids)
        self.btn_capture_ids.pack(side=tk.LEFT, padx=5)
        self.btn_load_dxf = ttk.Button(control_frame_1, text="2. Load DXF", command=self.load_dxf)
        self.btn_load_dxf.pack(side=tk.LEFT, padx=(15,5))
        self.btn_visualize = ttk.Button(control_frame_1, text="Visualize", state=tk.DISABLED, command=self.run_visualization)
        self.btn_visualize.pack(side=tk.LEFT, padx=(20, 5))
        self.btn_inspect = ttk.Button(control_frame_1, text="Align (Affine)", state=tk.DISABLED, command=self.run_alignment_and_inspection)
        self.btn_inspect.pack(side=tk.LEFT, padx=5)
        self.btn_inspect_homography = ttk.Button(control_frame_1, text="Align (Homography)", state=tk.DISABLED, command=self.run_homography_inspection)
        self.btn_inspect_homography.pack(side=tk.LEFT, padx=5)
        self.btn_find_anomalies = ttk.Button(control_frame_1, text="Find Anomalies", state=tk.DISABLED, command=self.run_feature_specific_anomaly_detection, style="Accent.TButton")
        self.btn_find_anomalies.pack(side=tk.LEFT, padx=(15, 5))
        
        # Frame 2: Hole and Missing Material Tolerances
        self.lbl_hole_tolerance = tk.Label(control_frame_2, text="Hole Occlusion Tol (%):", bg=self.colors["bg"], fg=self.colors["fg"])
        self.lbl_hole_tolerance.pack(side=tk.LEFT, padx=(15, 5))
        self.hole_occlusion_tolerance_var = tk.DoubleVar(value=10.0)
        self.hole_tolerance_slider = ttk.Scale(control_frame_2, from_=1.0, to=99.0, orient=tk.HORIZONTAL, length=150, variable=self.hole_occlusion_tolerance_var, command=self._update_slider_labels)
        self.hole_tolerance_slider.pack(side=tk.LEFT, padx=5)
        self.lbl_hole_occlusion_val = tk.Label(control_frame_2, text="", bg=self.colors["bg"], fg=self.colors["accent"], width=6, anchor='w')
        self.lbl_hole_occlusion_val.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_missing_tolerance = tk.Label(control_frame_2, text="Missing Mat. Tol (% Diag):", bg=self.colors["bg"], fg=self.colors["fg"])
        self.lbl_missing_tolerance.pack(side=tk.LEFT, padx=(15, 5))
        self.missing_material_tolerance_var = tk.DoubleVar(value=0.5)
        self.missing_tolerance_slider = ttk.Scale(control_frame_2, from_=0.1, to=5.0, orient=tk.HORIZONTAL, length=150, variable=self.missing_material_tolerance_var, command=self._update_slider_labels)
        self.missing_tolerance_slider.pack(side=tk.LEFT, padx=5)
        self.lbl_missing_tol_val = tk.Label(control_frame_2, text="", bg=self.colors["bg"], fg=self.colors["accent"], width=6, anchor='w')
        self.lbl_missing_tol_val.pack(side=tk.LEFT, padx=5)

        # Frame 3: Extra Material Tolerance and Debug
        self.lbl_extra_tolerance = tk.Label(control_frame_3, text="Extra Mat. Tol (% Diag):  ", bg=self.colors["bg"], fg=self.colors["fg"])
        self.lbl_extra_tolerance.pack(side=tk.LEFT, padx=(15, 5))
        self.extra_material_tolerance_var = tk.DoubleVar(value=0.5)
        self.extra_tolerance_slider = ttk.Scale(control_frame_3, from_=0.1, to=5.0, orient=tk.HORIZONTAL, length=150, variable=self.extra_material_tolerance_var, command=self._update_slider_labels)
        self.extra_tolerance_slider.pack(side=tk.LEFT, padx=5)
        self.lbl_extra_tol_val = tk.Label(control_frame_3, text="", bg=self.colors["bg"], fg=self.colors["accent"], width=6, anchor='w')
        self.lbl_extra_tol_val.pack(side=tk.LEFT, padx=(0, 20))
        
        self.debug_mode = tk.BooleanVar()
        self.chk_debug = tk.Checkbutton(control_frame_3, text="Debug Mode", variable=self.debug_mode, 
                                        bg=self.colors["bg"], fg=self.colors["fg"], 
                                        selectcolor=self.colors["btn"], activebackground=self.colors["bg"],
                                        activeforeground=self.colors["fg"], highlightthickness=0)
        self.chk_debug.pack(side=tk.LEFT, padx=(142, 5))

        # Crease Controls Frame
        self.lbl_crease_low = tk.Label(control_frame_crease, text="Crease Low Thresh:", bg=self.colors["bg"], fg=self.colors["fg"])
        self.lbl_crease_low.pack(side=tk.LEFT, padx=(15,5))
        self.crease_low_thresh_var = tk.IntVar(value=10)
        self.crease_low_thresh_slider = ttk.Scale(control_frame_crease, from_=1, to=254, orient=tk.HORIZONTAL, length=100, variable=self.crease_low_thresh_var, command=self._update_slider_labels)
        self.crease_low_thresh_slider.pack(side=tk.LEFT, padx=5)
        self.lbl_crease_low_val = tk.Label(control_frame_crease, text="", bg=self.colors["bg"], fg=self.colors["accent"], width=4, anchor='w')
        self.lbl_crease_low_val.pack(side=tk.LEFT, padx=(0,10))
        
        self.lbl_crease_high = tk.Label(control_frame_crease, text="Crease High Thresh:", bg=self.colors["bg"], fg=self.colors["fg"])
        self.lbl_crease_high.pack(side=tk.LEFT, padx=(10,5))
        self.crease_high_thresh_var = tk.IntVar(value=100)
        self.crease_high_thresh_slider = ttk.Scale(control_frame_crease, from_=1, to=254, orient=tk.HORIZONTAL, length=100, variable=self.crease_high_thresh_var, command=self._update_slider_labels)
        self.crease_high_thresh_slider.pack(side=tk.LEFT, padx=5)
        self.lbl_crease_high_val = tk.Label(control_frame_crease, text="", bg=self.colors["bg"], fg=self.colors["accent"], width=4, anchor='w')
        self.lbl_crease_high_val.pack(side=tk.LEFT, padx=(0,10))

        self.lbl_crease_presence = tk.Label(control_frame_crease, text="Min Crease Presence (%):", bg=self.colors["bg"], fg=self.colors["fg"])
        self.lbl_crease_presence.pack(side=tk.LEFT, padx=(10,5))
        self.min_crease_presence_var = tk.DoubleVar(value=50.0)
        self.min_crease_presence_slider = ttk.Scale(control_frame_crease, from_=1, to=99, orient=tk.HORIZONTAL, length=100, variable=self.min_crease_presence_var, command=self._update_slider_labels)
        self.min_crease_presence_slider.pack(side=tk.LEFT, padx=5)
        self.lbl_crease_presence_val = tk.Label(control_frame_crease, text="", bg=self.colors["bg"], fg=self.colors["accent"], width=6, anchor='w')
        self.lbl_crease_presence_val.pack(side=tk.LEFT, padx=0)
        
        # Frame 4: Status Labels
        self.lbl_image_status = tk.Label(control_frame_4, text="Image: None", bg=self.colors["bg"], fg=self.colors["fg"], padx=10)
        self.lbl_image_status.pack(side=tk.LEFT)
        self.lbl_dxf_status = tk.Label(control_frame_4, text="DXF: None", bg=self.colors["bg"], fg=self.colors["fg"], padx=10)
        self.lbl_dxf_status.pack(side=tk.LEFT)

        # Image Display Setup
        self.image_label = tk.Label(self.image_frame, bg="#000000")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label.bind("<Configure>", self.on_resize)
        
        self.load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        self.save_settings()
        cv2.destroyAllWindows()
        self.root.destroy()

    def save_settings(self):
        settings = {
            "hole_occlusion_tolerance": self.hole_occlusion_tolerance_var.get(),
            "missing_material_tolerance": self.missing_material_tolerance_var.get(),
            "extra_material_tolerance": self.extra_material_tolerance_var.get(),
            "crease_low_thresh": self.crease_low_thresh_var.get(),
            "crease_high_thresh": self.crease_high_thresh_var.get(),
            "min_crease_presence": self.min_crease_presence_var.get(),
            "debug_mode": self.debug_mode.get()
        }
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            print("Settings saved successfully.")
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.hole_occlusion_tolerance_var.set(settings.get("hole_occlusion_tolerance", 10.0))
                    self.missing_material_tolerance_var.set(settings.get("missing_material_tolerance", 0.5))
                    self.extra_material_tolerance_var.set(settings.get("extra_material_tolerance", 0.5))
                    self.crease_low_thresh_var.set(settings.get("crease_low_thresh", 10))
                    self.crease_high_thresh_var.set(settings.get("crease_high_thresh", 100))
                    self.min_crease_presence_var.set(settings.get("min_crease_presence", 50.0))
                    self.debug_mode.set(settings.get("debug_mode", False))
                    print("Settings loaded successfully.")
        except Exception as e:
            print(f"Error loading settings, using defaults. Error: {e}")
        finally:
            self._update_slider_labels()

    def _update_slider_labels(self, *args):
        self.lbl_hole_occlusion_val.config(text=f"{self.hole_occlusion_tolerance_var.get():.1f}%")
        self.lbl_missing_tol_val.config(text=f"{self.missing_material_tolerance_var.get():.1f}%")
        self.lbl_extra_tol_val.config(text=f"{self.extra_material_tolerance_var.get():.1f}%")
        self.lbl_crease_low_val.config(text=f"{self.crease_low_thresh_var.get()}")
        self.lbl_crease_high_val.config(text=f"{self.crease_high_thresh_var.get()}")
        self.lbl_crease_presence_val.config(text=f"{self.min_crease_presence_var.get():.1f}%")
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat", background=self.colors["btn"], foreground=self.colors["fg"], font=('Helvetica', 10, 'bold'), borderwidth=0)
        style.map("TButton", background=[('active', self.colors["btn_active"]), ('disabled', '#3D3D3D')], foreground=[('disabled', '#888888')])
        style.configure("Accent.TButton", background=self.colors["accent"], foreground=self.colors["accent_fg"], font=('Helvetica', 10, 'bold'))
        style.map("Accent.TButton", background=[('active', '#005f9e'), ('disabled', '#3D3D3D')])
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["fg"], font=('Helvetica', 10))
        style.configure("Horizontal.TScale", background=self.colors["bg"], troughcolor=self.colors["btn"])

    def _detect_crease_transitions(self, image, low_thresh, high_thresh, scan_type='rows'):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if scan_type == 'columns':
            gray_image = cv2.transpose(gray_image)

        h, w = gray_image.shape
        crease_mask = np.zeros((h, w), dtype=np.uint8)

        for i in range(h):
            start, down, up = False, False, False
            pre_val, pre_coord = 0, 0
            for j in range(w - 1):
                p1 = int(gray_image[i, j])
                p2 = int(gray_image[i, j + 1])

                if p1 > p2: # Downward transition
                    if not start:
                        start, down, pre_val, pre_coord = True, True, p1, j
                    elif up:
                        delta = abs(pre_val - p1)
                        if low_thresh <= delta <= high_thresh:
                            crease_mask[i, (j + pre_coord) // 2] = 255
                        up, down, pre_val, pre_coord = False, True, p1, j
                elif p1 < p2: # Upward transition
                    if not start:
                        start, up, pre_val, pre_coord = True, True, p1, j
                    elif down:
                        delta = abs(pre_val - p1)
                        if low_thresh <= delta <= high_thresh:
                            crease_mask[i, (j + pre_coord) // 2] = 255
                        down, up, pre_val, pre_coord = False, True, p1, j
        
        return cv2.transpose(crease_mask) if scan_type == 'columns' else crease_mask

    def run_feature_specific_anomaly_detection(self):
        print("\n--- Starting Full Anomaly Detection ---")
        if self.last_transform_matrix is None or self.original_cv_image is None:
            messagebox.showerror("Prerequisite Error", "Image must be loaded and aligned first.")
            return

        h, w = self.original_cv_image.shape[:2]
        transform_func = cv2.transform if self.last_transform_type == 'affine' else cv2.perspectiveTransform
        visualization_img = self.original_cv_image.copy()
        found_anomalies = 0
        
        # --- 1. SETUP MASKS AND ALIGNED FEATURES ---
        raw_image_mask = np.zeros((h, w), dtype=np.uint8)
        image_contours_raw, _ = self.find_image_contours_with_holes(self.original_cv_image)
        if not image_contours_raw:
            messagebox.showerror("Detection Error", "Could not find any object contours in the image."); return
        cv2.drawContours(raw_image_mask, image_contours_raw, -1, 255, cv2.FILLED)

        aligned_cad_outline = transform_func(self.cad_features['outline'].reshape(-1, 1, 2), self.last_transform_matrix)
        aligned_cad_holes = [transform_func(h.reshape(-1, 1, 2), self.last_transform_matrix) for h in self.cad_features['holes']]
        aligned_cad_outline_int = aligned_cad_outline.astype(np.int32)

        cv2.polylines(visualization_img, [aligned_cad_outline_int], True, (0, 255, 255), 1)
        for hole in aligned_cad_holes:
            cv2.polylines(visualization_img, [hole.astype(np.int32)], True, (100, 100, 100), 1)

        outline_area_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(outline_area_mask, [aligned_cad_outline_int], -1, 255, -1)
        
        # --- 2. OUTLINE AND HOLE DEFECTS ---
        cleaned_physical_mask = cv2.bitwise_and(raw_image_mask, outline_area_mask)
        
        object_diagonal = math.sqrt(w**2 + h**2)
        extra_tolerance_px = max(1, int(round(object_diagonal * (self.extra_material_tolerance_var.get() / 100.0))))
        kernel_dilate = np.ones((extra_tolerance_px, extra_tolerance_px), np.uint8)
        upper_bound_mask = cv2.dilate(outline_area_mask, kernel_dilate)

        missing_tolerance_px = max(1, int(round(object_diagonal * (self.missing_material_tolerance_var.get() / 100.0))))
        kernel_erode = np.ones((missing_tolerance_px, missing_tolerance_px), np.uint8)
        lower_bound_mask = cv2.erode(outline_area_mask, kernel_erode)

        extra_material_mask = cv2.subtract(cleaned_physical_mask, upper_bound_mask)
        contours, _ = cv2.findContours(extra_material_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for contour in contours:
                if cv2.contourArea(contour) < 4: continue
                found_anomalies += 1
                cv2.drawContours(visualization_img, [contour], -1, (0, 165, 255), -1)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
                    cv2.putText(visualization_img, "Extra Material", (cx - 40, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        missing_material_mask = cv2.subtract(lower_bound_mask, cleaned_physical_mask)
        all_missing_contours, _ = cv2.findContours(missing_material_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if all_missing_contours:
            processed_contour_indices = set()
            for expected_hole in aligned_cad_holes:
                total_detected_area_in_hole = 0
                contours_in_this_hole = []
                for i, contour in enumerate(all_missing_contours):
                    M = cv2.moments(contour)
                    if M["m00"] == 0: continue
                    cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
                    if cv2.pointPolygonTest(expected_hole, (float(cx), float(cy)), False) >= 0:
                        total_detected_area_in_hole += cv2.contourArea(contour)
                        contours_in_this_hole.append(contour)
                        processed_contour_indices.add(i)

                expected_hole_area = cv2.contourArea(expected_hole)
                presence_ratio = min(1.0, total_detected_area_in_hole / expected_hole_area if expected_hole_area > 0 else 0.0)
                occlusion_ratio = 1.0 - presence_ratio
                max_allowed_occlusion = self.hole_occlusion_tolerance_var.get() / 100.0

                if occlusion_ratio > max_allowed_occlusion:
                    found_anomalies += 1
                    cv2.drawContours(visualization_img, contours_in_this_hole, -1, (255, 100, 0), -1)
                    M_hole = cv2.moments(expected_hole)
                    label_cx = int(M_hole["m10"] / M_hole["m00"]) if M_hole["m00"] > 0 else 0
                    label_cy = int(M_hole["m01"] / M_hole["m00"]) if M_hole["m00"] > 0 else 0
                    label = f"Hole Occluded ({occlusion_ratio*100:.0f}%)"
                    cv2.putText(visualization_img, label, (label_cx - 50, label_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            for i, contour in enumerate(all_missing_contours):
                if i in processed_contour_indices: continue
                if cv2.contourArea(contour) < 5: continue
                found_anomalies += 1
                cv2.drawContours(visualization_img, [contour], -1, (0, 0, 255), -1)
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
                    cv2.putText(visualization_img, "Missing Material", (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # --- 3. CREASE DEFECT DETECTION ---
        print("\n--- Starting Crease Detection ---")
        low_t = self.crease_low_thresh_var.get()
        high_t = self.crease_high_thresh_var.get()
        min_presence = self.min_crease_presence_var.get() / 100.0

        print(" - Running row scan for vertical creases...")
        vertical_crease_mask = self._detect_crease_transitions(visualization_img, low_t, high_t, 'rows')
        print(" - Running column scan for horizontal creases...")
        horizontal_crease_mask = self._detect_crease_transitions(visualization_img, low_t, high_t, 'columns')

        if self.debug_mode.get():
            cv2.imshow("DEBUG: Vertical Crease Scan", vertical_crease_mask)
            cv2.imshow("DEBUG: Horizontal Crease Scan", horizontal_crease_mask)

        for crease in self.cad_features['creases']:
            if crease.size < 4: continue
            transformed_crease = transform_func(crease.reshape(-1, 1, 2), self.last_transform_matrix).astype(np.int32)
            
            p1 = transformed_crease[0, 0]; p2 = transformed_crease[-1, 0]
            angle = abs(math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / math.pi)

            scan_mask_to_use = vertical_crease_mask if 45 < angle < 135 else horizontal_crease_mask
            
            crease_roi_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.polylines(crease_roi_mask, [transformed_crease], False, 255, 5) # Use thickness for ROI
            
            detected_pixels_mask = cv2.bitwise_and(scan_mask_to_use, crease_roi_mask)
            detected_pixel_count = cv2.countNonZero(detected_pixels_mask)
            
            expected_pixel_count = cv2.arcLength(transformed_crease.astype(np.float32), False)
            if expected_pixel_count < 1: continue

            presence_ratio = detected_pixel_count / expected_pixel_count
            
            if presence_ratio < min_presence:
                found_anomalies += 1
                cv2.polylines(visualization_img, [transformed_crease], False, (0, 255, 0), 2)
                label_pos = tuple(transformed_crease[len(transformed_crease) // 2, 0])
                label = f"Crease Defect ({presence_ratio*100:.0f}%)"
                cv2.putText(visualization_img, label, (label_pos[0] - 50, label_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        # --- FINALIZATION ---
        self.update_image_display(visualization_img)
        messagebox.showinfo("Inspection Complete", f"Found {found_anomalies} total potential anomalies.")
        print(f"--- Anomaly Detection Complete: {found_anomalies} issues found. ---")
    
    # ... Rest of the unchanged methods from previous versions
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
            self.last_transform_matrix = None 
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Failed to load image: {e}")
            self.reset_image_state()

    def capture_from_ids(self):
        h_cam = ueye.HIDS(0) 
        mem_ptr = ueye.c_mem_p()
        mem_id = ueye.int()
        try:
            ret = ueye.is_InitCamera(h_cam, None)
            if ret != ueye.IS_SUCCESS: messagebox.showerror("IDS Camera Error", f"Could not initialize camera. Error code: {ret}"); return
            sensor_info = ueye.SENSORINFO(); ueye.is_GetSensorInfo(h_cam, sensor_info)
            width, height = int(sensor_info.nMaxWidth), int(sensor_info.nMaxHeight)
            bits_per_pixel = 24
            ret = ueye.is_SetColorMode(h_cam, ueye.IS_CM_BGR8_PACKED)
            if ret != ueye.IS_SUCCESS: messagebox.showerror("IDS Camera Error", "Could not set color mode. Is camera color?"); return
            ueye.is_AllocImageMem(h_cam, width, height, bits_per_pixel, mem_ptr, mem_id)
            ueye.is_SetImageMem(h_cam, mem_ptr, mem_id)
            ret = ueye.is_FreezeVideo(h_cam, ueye.IS_WAIT)
            if ret != ueye.IS_SUCCESS: messagebox.showerror("IDS Camera Error", "Could not capture image from camera."); return
            array = ueye.get_data(mem_ptr, width, height, bits_per_pixel, width * bits_per_pixel // 8, copy=True)
            self.original_cv_image = np.reshape(array, (height, width, 3))
            self.cv_image = self.original_cv_image.copy()
            self.lbl_image_status.config(text="Image: From IDS Camera"); self.image_path = "IDS_Capture"
            self.update_image_display(self.cv_image); self._check_files_loaded(); self.last_transform_matrix = None 
            messagebox.showinfo("Success", "Image captured successfully from IDS camera.")
        except Exception as e:
            messagebox.showerror("IDS Camera Error", f"An unexpected error occurred: {traceback.format_exc()}")
        finally:
            if mem_ptr: ueye.is_FreeImageMem(h_cam, mem_ptr, mem_id)
            if h_cam: ueye.is_ExitCamera(h_cam)

    def load_dxf(self):
        filepath = filedialog.askopenfilename(title="Select a DXF File", filetypes=[("DXF Files", "*.dxf"), ("All files", "*.*")])
        if not filepath: return
        self.dxf_path = filepath
        try:
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
            if outlines:
                self.cad_features['outline'] = max(outlines, key=lambda p: cv2.arcLength(p.reshape(-1, 1, 2), False))
            self.cad_features['holes'] = holes
            self.cad_features['creases'] = creases
            self.lbl_dxf_status.config(text=f"DXF: {os.path.basename(self.dxf_path)}")
            self._check_files_loaded()
            self.last_transform_matrix = None 
        except Exception as e:
            messagebox.showerror("DXF Load Error", f"An error occurred during DXF processing:\n\n{traceback.format_exc()}")
            self.reset_dxf_state()

    def run_alignment_and_inspection(self):
        print("\n--- Starting Alignment & Inspection (Affine/Moments+ICP Pipeline) ---")
        transform_matrix = self.align_with_moments_icp_pipeline()
        if transform_matrix is None:
            messagebox.showerror("Alignment Error", "Could not compute the transformation matrix.")
            self.last_transform_matrix = None
            return
        
        self.last_transform_matrix = transform_matrix
        self.last_transform_type = 'affine'
        
        result_image = self.original_cv_image.copy()
        transformed_outline = cv2.transform(self.cad_features['outline'].reshape(-1, 1, 2), transform_matrix)
        cv2.polylines(result_image, [transformed_outline.astype(np.int32)], True, (0, 255, 255), 2)
        for hole in self.cad_features['holes']:
            if hole.size > 0:
                transformed_hole = cv2.transform(hole.reshape(-1,1,2), transform_matrix)
                cv2.polylines(result_image, [transformed_hole.astype(np.int32)], True, (255, 0, 255), 2)
        for crease in self.cad_features['creases']:
            if crease.size > 0:
                transformed_crease = cv2.transform(crease.reshape(-1,1,2), transform_matrix)
                cv2.polylines(result_image, [transformed_crease.astype(np.int32)], False, (255, 255, 0), 2)
        
        self.update_image_display(result_image)
        messagebox.showinfo("Inspection Complete", "DXF features aligned using the Affine (Moments+ICP) pipeline.")

    def run_homography_inspection(self):
        print("\n--- Starting Smart Homography Alignment ---")
        homography_matrix, method_used = self.align_with_homography_pipeline()
        
        if homography_matrix is None:
            messagebox.showerror("Alignment Error", "Could not compute the homography matrix.")
            self.last_transform_matrix = None
            return

        self.last_transform_matrix = homography_matrix
        self.last_transform_type = 'homography'

        result_image = self.original_cv_image.copy()
        transformed_outline = cv2.perspectiveTransform(self.cad_features['outline'].reshape(-1, 1, 2), homography_matrix)
        cv2.polylines(result_image, [transformed_outline.astype(np.int32)], True, (0, 255, 255), 2)
        for hole in self.cad_features['holes']:
            if hole.size > 0:
                transformed_hole = cv2.perspectiveTransform(hole.reshape(-1,1,2), homography_matrix)
                cv2.polylines(result_image, [transformed_hole.astype(np.int32)], True, (255, 0, 255), 2)
        for crease in self.cad_features['creases']:
            if crease.size > 0:
                transformed_crease = cv2.perspectiveTransform(crease.reshape(-1,1,2), homography_matrix)
                cv2.polylines(result_image, [transformed_crease.astype(np.int32)], False, (255, 255, 0), 2)

        self.update_image_display(result_image)
        messagebox.showinfo("Inspection Complete", f"DXF features aligned using the {method_used} method.")
        
    def _check_files_loaded(self):
        all_loaded = self.image_path is not None and self.dxf_path is not None and self.cad_features['outline'].size > 0
        state = tk.NORMAL if all_loaded else tk.DISABLED
        self.btn_visualize.config(state=state)
        self.btn_inspect.config(state=state)
        self.btn_inspect_homography.config(state=state)
        self.btn_find_anomalies.config(state=state)

    def on_resize(self, event=None):
        self.update_image_display(self.cv_image)

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
                if dxf_type == 'LWPOLYLINE':
                    points = [(p[0], p[1]) for p in entity.get_points('xy')]
                else: 
                    points = [(p.dxf.location.x, p.dxf.location.y) for p in entity.vertices]
            elif dxf_type in ('CIRCLE', 'ARC', 'ELLIPSE', 'SPLINE'):
                points = [(p.x, p.y) for p in entity.flattening(sagitta=0.01)]
        except Exception as e:
            print(f"               Warning: Could not process entity {dxf_type}: {e}")
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
        self.last_transform_matrix = None
        self._check_files_loaded()

    def reset_dxf_state(self):
        self.dxf_path = None
        self.cad_features = {'outline': np.array([]), 'holes': [], 'creases': []}
        self.lbl_dxf_status.config(text="DXF: None")
        self.last_transform_matrix = None
        self._check_files_loaded()

    def align_with_moments_icp_pipeline(self):
        print("1. Preprocessing: Extracting and preparing contours...")
        img_contour = self.find_image_contour(self.original_cv_image)
        if img_contour is None:
            print("   Error: No contour found in the image.")
            return None
        
        cad_contour_original = self.cad_features['outline'].reshape(-1, 2).astype(np.float32)
        
        print("\n--- Starting Pass 1: Y-Flipped (Image Coordinate System) ---")
        y_max = np.max(cad_contour_original[:, 1])
        cad_contour_yflipped = cad_contour_original.copy()
        cad_contour_yflipped[:, 1] = y_max - cad_contour_yflipped[:, 1]
        
        coarse_transform_1 = self._align_with_moments(cad_contour_yflipped, img_contour)
        if coarse_transform_1 is None:
                print("   Error: Coarse alignment failed on Pass 1.")
                return None
        
        final_transform_1, final_mse_1 = self._iterative_closest_point(source_points=cad_contour_yflipped, target_points=img_contour, initial_matrix=coarse_transform_1)
        if final_transform_1 is None:
            print("   Error: ICP failed on Pass 1.")
            return None
        print(f"   Pass 1 (Y-Flipped) Complete. MSE: {final_mse_1:.4f}")

        print("\n--- Starting Pass 2: Original (CAD Coordinate System) ---")
        coarse_transform_2 = self._align_with_moments(cad_contour_original, img_contour)
        if coarse_transform_2 is None:
            print("   Error: Coarse alignment failed on Pass 2.")
            return None

        final_transform_2, final_mse_2 = self._iterative_closest_point(source_points=cad_contour_original, target_points=img_contour, initial_matrix=coarse_transform_2)
        if final_transform_2 is None:
            print("   Error: ICP failed on Pass 2.")
            return None
        print(f"   Pass 2 (Original) Complete. MSE: {final_mse_2:.4f}")
        
        if final_mse_1 < final_mse_2:
            print(f"\n--- Y-Flipped alignment is better (MSE {final_mse_1:.4f} < {final_mse_2:.4f}). ---")
            y_flip_matrix = np.array([[1, 0, 0], [0, -1, y_max]], dtype=np.float32)
            T_flip_3x3 = np.vstack([y_flip_matrix, [0, 0, 1]])
            T_align_3x3 = np.vstack([final_transform_1, [0, 0, 1]])
            combined_transform_3x3 = T_align_3x3 @ T_flip_3x3
            return combined_transform_3x3[:2, :]
        else:
            print(f"\n--- Original alignment is better (MSE {final_mse_2:.4f} <= {final_mse_1:.4f}). ---")
            return final_transform_2

    def align_with_homography_pipeline(self):
        print("--- Starting Smart Homography Pipeline (with MSE convergence) ---")
        print("1. Calculating baseline affine transformation...")
        M_affine = self.align_with_moments_icp_pipeline()
        if M_affine is None:
            return None, "Error"
        img_contour = self.find_image_contour(self.original_cv_image)
        if img_contour is None: return None, "Error"
        cad_contour = self.cad_features['outline'].reshape(-1, 2).astype(np.float32)
        affine_mse = self._calculate_alignment_error(cad_contour, img_contour, M_affine)
        if affine_mse is None: return None, "Error"
        print(f"   Benchmark Affine MSE: {affine_mse:.4f}")
        print("4. Starting iterative homography refinement...")
        image_corners = self._extract_corners(img_contour, epsilon_factor=0.0015)
        cad_corners = self._extract_corners(cad_contour, epsilon_factor=0.0015)
        if image_corners is None or cad_corners is None or len(image_corners) < 4 or len(cad_corners) < 4:
            print("   Error: Insufficient corners for homography. Falling back to affine.")
            final_matrix = np.vstack([M_affine, [0, 0, 1]])
            return final_matrix, "Affine Fallback (Not enough corners)"
        current_H = np.vstack([M_affine, [0, 0, 1]])
        image_corner_kdtree = KDTree(image_corners)
        previous_mse = affine_mse 
        for i in range(10):
            cad_corners_transformed = cv2.perspectiveTransform(cad_corners.reshape(-1, 1, 2), current_H).reshape(-1, 2)
            _, indices = image_corner_kdtree.query(cad_corners_transformed)
            src_pts, dst_pts = cad_corners, image_corners[indices]
            next_H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if next_H is None:
                break
            current_mse = self._calculate_alignment_error(cad_contour, img_contour, next_H)
            if current_mse is None:
                break 
            if current_mse >= previous_mse - 1e-6:
                break 
            current_H = next_H
            previous_mse = current_mse
        homography_mse = previous_mse
        print(f"   Final Homography MSE: {homography_mse:.4f}")
        if homography_mse < affine_mse:
            print("   SUCCESS: Homography improved alignment. Using perspective transform.")
            return current_H, "Iterative Homography"
        else:
            print("   INFO: Homography did not improve alignment. Falling back to affine transform.")
            final_matrix = np.vstack([M_affine, [0, 0, 1]])
            return final_matrix, "Affine Fallback (MSE did not improve)"

    def _calculate_alignment_error(self, source_contour, target_contour, transform_matrix):
        if source_contour is None or target_contour is None or transform_matrix is None: return None
        if transform_matrix.shape == (2, 3):
            transformed_source = cv2.transform(source_contour.reshape(-1, 1, 2), transform_matrix)
        elif transform_matrix.shape == (3, 3):
            transformed_source = cv2.perspectiveTransform(source_contour.reshape(-1, 1, 2), transform_matrix)
        else: return None
        transformed_source = transformed_source.reshape(-1, 2)
        target_contour_points = target_contour.reshape(-1, 2)
        try:
            kdtree = KDTree(target_contour_points)
            distances, _ = kdtree.query(transformed_source)
            return np.mean(np.square(distances))
        except Exception as e:
            print(f"   Error during MSE calculation: {e}")
            return None

    def _extract_corners(self, contour, epsilon_factor=0.001):
        if contour is None or len(contour) < 3: return None
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, epsilon, True)
        return corners.reshape(-1, 2) if corners is not None else None

    def _align_with_moments(self, cad_contour, img_contour):
        try:
            M_cad = cv2.moments(cad_contour)
            M_img = cv2.moments(img_contour)
            if M_cad["m00"] == 0 or M_img["m00"] == 0: return None
            cad_cx, cad_cy = M_cad["m10"] / M_cad["m00"], M_cad["m01"] / M_cad["m00"]
            img_cx, img_cy = M_img["m10"] / M_img["m00"], M_img["m01"] / M_img["m00"]
            scale = math.sqrt(M_img["m00"] / M_cad["m00"])
            get_orientation = lambda M: 0.5 * math.atan2(2 * M['mu11'], M['mu20'] - M['mu02'])
            angle_cad, angle_img = get_orientation(M_cad), get_orientation(M_img)
            rotation_angle_rad = angle_img - angle_cad
            cos_a, sin_a = math.cos(rotation_angle_rad), math.sin(rotation_angle_rad)
            tx = img_cx - (scale * (cad_cx * cos_a - cad_cy * sin_a))
            ty = img_cy - (scale * (cad_cx * sin_a + cad_cy * cos_a))
            return np.array([[scale * cos_a, -scale * sin_a, tx], [scale * sin_a,  scale * cos_a, ty]], dtype=np.float32)
        except Exception as e:
            print(f"   Error during moments-based alignment: {e}")
            return None

    def _iterative_closest_point(self, source_points, target_points, initial_matrix, max_iterations=100, tolerance=1e-5):
        if initial_matrix is None: return None, float('inf')
        source_pts, target_pts = source_points.reshape(-1, 2), target_points.reshape(-1, 2)
        if len(source_pts) == 0 or len(target_pts) == 0: return None, float('inf')
        target_kdtree = KDTree(target_pts)
        current_transform = initial_matrix.copy()
        prev_error = float('inf')
        for i in range(max_iterations):
            source_transformed = cv2.transform(source_pts.reshape(-1, 1, 2), current_transform).reshape(-1, 2)
            distances, indices = target_kdtree.query(source_transformed)
            correspondences = target_pts[indices]
            if len(source_pts) < 3 or len(correspondences) < 3: return current_transform, prev_error
            new_transform, _ = cv2.estimateAffinePartial2D(source_pts, correspondences, method=cv2.LMEDS)
            if new_transform is None: return current_transform, prev_error
            current_transform = new_transform
            mean_error = np.mean(distances**2)
            if abs(prev_error - mean_error) < tolerance: return current_transform, mean_error
            prev_error = mean_error
        return current_transform, prev_error

    def find_image_contour(self, image):
        if image is None: return None
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound_hsv, upper_bound_hsv = np.array([5, 50, 50]), np.array([50, 255, 255])
        color_mask = cv2.inRange(hsv_image, lower_bound_hsv, upper_bound_hsv)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour.astype(np.float32) if largest_contour is not None else None

    def find_image_contours_with_holes(self, image):
        if image is None: return None, None
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound_hsv, upper_bound_hsv = np.array([5, 50, 50]), np.array([50, 255, 255])
        color_mask = cv2.inRange(hsv_image, lower_bound_hsv, upper_bound_hsv)
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy

    def run_visualization(self):
        print("--- Running Input Visualization ---")
        if self.cad_features['outline'].size > 0:
            all_points_list = [self.cad_features['outline']] + self.cad_features['holes'] + self.cad_features['creases']
            all_points = np.vstack([p.reshape(-1, 2) for p in all_points_list if p.size > 0])
            if all_points.size == 0:
                messagebox.showerror("Visualize Error", "No points found in DXF features to visualize.")
                return
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
                if self.cad_features['holes']:
                    cv2.polylines(cad_canvas, [transform_pt(p) for p in self.cad_features['holes']], True, (0, 255, 255), 1)
                if self.cad_features['creases']:
                    cv2.polylines(cad_canvas, [transform_pt(p) for p in self.cad_features['creases']], False, (255, 255, 0), 1)
                self._resize_for_display("DEBUG: Parsed CAD", cad_canvas)
        if self.original_cv_image is not None:
            image_contour = self.find_image_contour(self.original_cv_image)
            if image_contour is not None:
                debug_image = self.original_cv_image.copy()
                cv2.drawContours(debug_image, [image_contour.astype(np.int32)], -1, (255, 0, 255), 3)
                self._resize_for_display("DEBUG: Detected Image Contour", debug_image)
            else:
                messagebox.showinfo("Visualize", "No contour detected in image with current HSV settings.")
        messagebox.showinfo("Debug", "Debug windows opened.")

    def _resize_for_display(self, window_name, image, max_dim=900):
        h, w = image.shape[:2]
        if max_dim <= 0: max_dim=900
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            display_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            display_image = image
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_name, display_image)
        cv2.waitKey(1)

if __name__ == "__main__":
    root = tk.Tk()
    app = InspectionApp(root)
    root.mainloop()