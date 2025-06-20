import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 1. CONFIGURATION ---
# Set the path to your image
image_path = "Multi-Prespective-Test-Cases\\cardboard.png" # <-- CHANGE THIS if needed
model_type = "facebook/sam2.1-hiera-base-plus"

# --- 2. LOAD IMAGE ---
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- 3. INTERACTIVE POINT SELECTION ---
print("--- Interactive Point Selection ---")
print("An interactive window will open.")
print(" - LEFT-CLICK to add a GREEN foreground point.")
print(" - RIGHT-CLICK to add a RED background point.")
print(" - CLOSE THE WINDOW when you are finished selecting points.")

# Global list to store clicked points and their labels
prompt_points = []

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image_rgb)
ax.set_title("Click to add points. Close window to segment.")
ax.axis('off')

# Define the function to be called on mouse click
def onclick(event):
    # Ignore clicks outside the plot axes
    if event.inaxes != ax:
        return
    
    # Get click coordinates
    x, y = int(event.xdata), int(event.ydata)
    
    # Determine label: 1 for left-click (foreground), 0 for right-click (background)
    label = 1 if event.button == 1 else 0
    color = 'green' if label == 1 else 'red'
    
    # Add the point to our list
    prompt_points.append(((x, y), label))
    
    # Draw the point on the plot for visual feedback
    ax.scatter(x, y, color=color, marker='*', s=150, edgecolor='white', linewidth=1.25)
    fig.canvas.draw() # Update the plot

# Connect the click event to our function
fig.canvas.mpl_connect('button_press_event', onclick)

# Display the plot. This is a blocking call; code will pause here until the window is closed.
plt.show()

if not prompt_points:
    print("No points selected. Exiting.")
    exit()

# Convert our list of points to the NumPy arrays SAM expects
input_points = np.array([p[0] for p in prompt_points])
input_labels = np.array([p[1] for p in prompt_points])

print(f"\n{len(prompt_points)} point(s) selected. Running model...")

# --- 4. MODEL INFERENCE ---
print("Loading model...")
predictor = SAM2ImagePredictor.from_pretrained(model_type)
print("Model loaded.")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image_rgb)
    
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )

# --- 5. PROCESS AND VISUALIZE THE RESULTS ---
best_mask_index = np.argmax(scores)
best_mask = masks[best_mask_index]

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6]) # Dodger blue
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

print("Prediction complete. Displaying final result.")
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
show_mask(best_mask, plt.gca())
show_points(input_points, input_labels, plt.gca())
plt.axis('off')
plt.title(f"SAM 2.1 Output (Score: {scores[best_mask_index]:.3f})")
plt.show()