import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class BackgroundSubtractionApp:
    """
    A simple GUI application to demonstrate robust contour detection
    using background subtraction.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Background Subtraction Test")
        
        self.background_frame = None
        self.image_with_object = None
        
        # --- UI Elements ---
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.btn_load_bg = tk.Button(btn_frame, text="1. Load Background Image", command=self.load_background)
        self.btn_load_bg.pack(side=tk.LEFT, padx=5)

        self.btn_load_obj = tk.Button(btn_frame, text="2. Load Image with Object", command=self.load_object_image, state=tk.DISABLED)
        self.btn_load_obj.pack(side=tk.LEFT, padx=5)

        self.lbl_status = tk.Label(main_frame, text="Please load a background image to begin.", relief=tk.SUNKEN, anchor=tk.W)
        self.lbl_status.pack(fill=tk.X, pady=5)
        
        # --- Image Display Panes ---
        display_frame = tk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.bg_pane = tk.LabelFrame(display_frame, text="Background", padx=5, pady=5)
        self.bg_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.bg_label = tk.Label(self.bg_pane)
        self.bg_label.pack()

        self.obj_pane = tk.LabelFrame(display_frame, text="Image with Object", padx=5, pady=5)
        self.obj_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.obj_label = tk.Label(self.obj_pane)
        self.obj_label.pack()

    def load_background(self):
        """Loads the reference image of the empty scene."""
        filepath = filedialog.askopenfilename(title="Select Background Image")
        if not filepath: return
        
        self.background_frame = cv2.imread(filepath)
        if self.background_frame is None:
            messagebox.showerror("Error", f"Failed to load background image from {filepath}")
            return
            
        self.lbl_status.config(text="Background loaded. Now load the image with the object.")
        self.display_image(self.background_frame, self.bg_label)
        self.btn_load_obj.config(state=tk.NORMAL)

    def load_object_image(self):
        """Loads the image with the object and performs subtraction."""
        filepath = filedialog.askopenfilename(title="Select Image with Object")
        if not filepath: return
        
        self.image_with_object = cv2.imread(filepath)
        if self.image_with_object is None:
            messagebox.showerror("Error", f"Failed to load object image from {filepath}")
            return
        
        if self.image_with_object.shape != self.background_frame.shape:
            messagebox.showerror("Error", "Object image and background image must have the same dimensions.")
            return

        self.lbl_status.config(text="Object image loaded. Performing subtraction...")
        self.display_image(self.image_with_object, self.obj_label)
        
        self.find_contour_with_subtraction()

    def find_contour_with_subtraction(self):
        """
        Performs background subtraction to find the object's contour.
        """
        # 1. Convert both images to grayscale and blur to reduce noise
        bg_gray = cv2.cvtColor(self.background_frame, cv2.COLOR_BGR2GRAY)
        bg_blur = cv2.GaussianBlur(bg_gray, (5, 5), 0)

        obj_gray = cv2.cvtColor(self.image_with_object, cv2.COLOR_BGR2GRAY)
        obj_blur = cv2.GaussianBlur(obj_gray, (5, 5), 0)

        # 2. Compute the absolute difference between the two images
        diff = cv2.absdiff(bg_blur, obj_blur)
        cv2.imshow("Debug 1: Absolute Difference", diff)
        cv2.waitKey(0)

        # 3. Threshold the difference image to get a binary mask
        # Any pixel difference greater than the threshold is considered part of the object.
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        cv2.imshow("Debug 2: Thresholded Difference", thresh)
        cv2.waitKey(0)

        # 4. Clean up the mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cv2.imshow("Debug 3: Cleaned Mask", closed)
        cv2.waitKey(0)

        # 5. Find the largest contour in the clean mask
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            messagebox.showerror("Error", "Could not find any contours after subtraction.")
            return

        largest_contour = max(contours, key=cv2.contourArea)

        # --- Draw result and show final image ---
        result_image = self.image_with_object.copy()
        cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 3)
        
        cv2.imshow("Final Result - Background Subtraction", result_image)
        cv2.waitKey(0)
        
        self.lbl_status.config(text="Processing complete. Ready for new images.")
        cv2.destroyAllWindows()


    def display_image(self, cv_image, label_widget):
        """Helper function to display an OpenCV image in a Tkinter label."""
        if cv_image is None: return
        
        # Resize for display
        h, w = cv_image.shape[:2]
        max_h = 400
        if h > max_h:
            scale = max_h / h
            new_w, new_h = int(w * scale), int(h * scale)
            display_image = cv2.resize(cv_image, (new_w, new_h))
        else:
            display_image = cv_image
            
        # Convert for Tkinter
        rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        
        label_widget.config(image=tk_image)
        label_widget.image = tk_image # Keep a reference


if __name__ == '__main__':
    root = tk.Tk()
    app = BackgroundSubtractionApp(root)
    root.mainloop()

