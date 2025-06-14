import cv2
import numpy as np
from matplotlib import pyplot as plt

def color_segment_cardboard(image_path='cardboard.png'):
    # Load the image
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert BGR to HSV
    hsv_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Define the HSV range for cardboard color
    lower_bound_hsv = np.array([5, 40, 40])    # Lower H, S, V
    upper_bound_hsv = np.array([35, 255, 240])   # Upper H, S, V
    
    # Create the mask
    color_mask = cv2.inRange(hsv_image, lower_bound_hsv, upper_bound_hsv)

    # # --- Optional: Refine the mask using morphological operations ---
    # # Opening: Remove small noise/dots (erosion followed by dilation)
    # kernel_open = np.ones((5, 5), np.uint8)
    # mask_opened = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # # Closing: Fill small holes within the main object (dilation followed by erosion)
    # kernel_close = np.ones((11, 11), np.uint8) # Larger kernel for closing
    # mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    final_mask = color_mask # Use the cleaned mask

    # --- Find contours on the final mask ---
    contours, hierarchy = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contour_drawn = img_rgb.copy()
    contour_status = "No Significant Cardboard Color Contours Found"

    if contours:
        # Filter out very small contours that might be noise
        # Or simply take the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        min_contour_area = 500 # Adjust as needed

        if cv2.contourArea(largest_contour) > min_contour_area:
            cv2.drawContours(img_contour_drawn, [largest_contour], -1, (0, 255, 0), 3)
            contour_status = f"Largest Color Contour (Area: {cv2.contourArea(largest_contour):.0f})"
        else:
            contour_status = (f"Largest color contour area ({cv2.contourArea(largest_contour):.0f}) "
                              f"below threshold {min_contour_area}.")
    else:
        print("No contours found from color mask.")


    # --- Display the results ---
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(hsv_image) # HSV image itself can look weird when plotted with RGB interpretation
    plt.title('HSV Image (for reference)')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(final_mask, cmap='gray')
    plt.title('Final Color Mask')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(img_contour_drawn)
    plt.title(contour_status)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print(f"Processed '{image_path}' with HSV range: L-{lower_bound_hsv}, U-{upper_bound_hsv}")

# Run the segmentation
# Make sure 'cardboard.png' exists or change the path.
color_segment_cardboard(image_path='cardboard.png')
