import cv2
import numpy as np
from matplotlib import pyplot as plt

img_bgr = cv2.imread('cardboard.png')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.title('Original Image (RGB for display)')
plt.imshow(img_rgb)


gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(gray, cmap='gray'), plt.title('Grayscale Image')
plt.subplot(122), plt.imshow(blur, cmap='gray'), plt.title('Blurred Image')
ret, threshold_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(threshold_otsu, cmap='gray'), plt.title('Binary Image (Otsu)')
plt.show()
current_threshold_img = threshold_otsu

contours, hierarchy = cv2.findContours(current_threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_copy_rgb = img_rgb.copy()
largest_contour = max(contours, key=cv2.contourArea)

cv2.drawContours(img_copy_rgb, [largest_contour], -1, (0, 255, 0), 3)

plt.imshow(img_copy_rgb), plt.title('Largest Contour on RGB Image')
plt.show()
