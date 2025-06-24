# CADBOARD: Visualizer & Inspector

CADBOARD is a desktop application developed with Python and Tkinter for visual inspection of physical parts against their CAD designs. It allows a user to load a `.dxf` file and an image of the corresponding physical object, align them, and perform an automated inspection to find manufacturing defects.

The application uses computer vision techniques to detect features in the image and compares them against the geometric data extracted from the DXF file. It can identify several types of anomalies:
-   **Extra Material:** Unexpected material found on the part.
-   **Missing Material:** Portions of the part's outline that are missing.
-   **Hole Occlusion:** Holes that are partially or fully blocked.
-   **Crease Defects:** Deformations or missing creases on the part's surface.

## Key Features

-   **Image and DXF Loading:** Load part images and DXF design files.
-   **Live Camera Capture:** Capture images directly from IDS uEye industrial cameras or standard IP/web cameras (e.g., a phone).
-   **MOG2 Background Subtraction:** You could use the MOG2 option for a more flexible object segmentation.
-   **Geometric Alignment:** Aligns the DXF drawing with the image using both Affine and Homography transformations.
-   **Anomaly Detection:** Automatically identifies and highlights discrepancies between the image and the CAD model.
-   **Adjustable Tolerances:** Fine-tune the sensitivity of the inspection for different types of defects via an intuitive UI.
-   **Debug Mode:** Visualize intermediate steps of the computer vision pipeline for troubleshooting and analysis.
-   **Persistent Settings:** Saves your tolerance and configuration settings between sessions.

## Dependencies

To run this application, you will need Python 3 and the following libraries:

-   `opencv-python`: For all computer vision tasks.
-   `ezdxf`: For reading and parsing `.dxf` files.
-   `numpy`: For numerical operations, especially with coordinates and image data.
-   `scipy`: Used for its `KDTree` -   `scipy`: Used for its `KDTree` implementation for efficient nearest-neighbor searches.
-   `Pillow` (PIL Fork): For handling and displaying images within the Tkinter UI.
-   `pyueye` (Optional): Required **only** if you intend to use an IDS uEye industrial camera. The application will run without it, but IDS camera features will be disabled.

## Installation

1.  **Clone or download the repository.**
    Place the `cadboard.py` file in a directory of your choice.

2.  **Install the required Python libraries using pip:**
    ```bash
    pip install opencv-python ezdxf numpy scipy Pillow
    ```

3.  **(Optional) Install IDS Camera Library:**
    If you need to connect to an IDS uEye camera, download and install the `pyueye` library from the official IDS Imaging website. Follow their instructions for installation. If this library is not found, the application will still run but the IDS-specific buttons will be disabled.

## How to Run

Navigate to the directory containing the script and run it from your terminal:

```bash
python cadboard.py
```
Basic Usage Workflow
 * Load Input:
   * Click 1a. Load Image to select a static image file of the part.
   * OR, click 1b. Capture IDS or 1d. MOG2 (Phone) to capture an image from a connected camera. For the phone option, you'll be prompted to enter the stream URL.
   * Click 2. Load DXF to select the corresponding CAD design file.
   * Status labels at the bottom will confirm that the files have been loaded.
 * Align CAD with Image:
   * Once an image and a DXF are loaded, the alignment buttons become active.
   * Click Align (Affine) or Align (Homography) to automatically align the DXF drawing over the object in the image. Homography is generally better for flat objects with perspective distortion, while Affine works well for 2D translations, rotations, and scaling.
   * The aligned DXF outline will be overlaid on the image in blue.
 * Adjust Tolerances (Optional):
   * Use the sliders at the top to adjust the sensitivity of the defect detection. For example, increase the "Hole Occlusion Tol" to allow for smaller blockages before flagging a hole as a defect. The values are updated in real-time.
 * Run Inspection:
   * Click the Find Anomalies button.
   * The application will analyze the part based on the aligned CAD model and the current tolerance settings.
   * Defects will be highlighted directly on the image display:
     * Orange: Extra material.
     * Red: Missing material or hole defects.
     * Yellow: Crease defects.
 * Review Results:
   * Examine the highlighted areas on the image. The application provides a clear visual representation of all detected manufacturing flaws. You can re-run the inspection with different tolerance settings as needed.
