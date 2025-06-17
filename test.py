from pyueye import ueye
import numpy as np
import cv2
import time
import os


def main():
    width, height = 1936, 1216  # Native resolution
    bitspp = 24  # BGR8 = 8 bits * 3 channels

    hCam = ueye.HIDS(0)
    if ueye.is_InitCamera(hCam, None) != ueye.IS_SUCCESS:
        print("Failed to initialize camera")
        return

    # Set color mode to hardware-processed BGR
    ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED)

    # Allocate image memory
    mem_ptr = ueye.c_mem_p()
    mem_id = ueye.int()
    ueye.is_AllocImageMem(hCam, width, height, bitspp, mem_ptr, mem_id)
    ueye.is_SetImageMem(hCam, mem_ptr, mem_id)

    # Enable automatic white balance
    ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE, ueye.c_double(1), None)

    # Start video capture
    ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)

    print("Streaming... Press 'q' to quit.")

    while True:
        # Get raw image buffer and reshape
        array = ueye.get_data(mem_ptr, width, height, bitspp, width * 3, copy=False)
        frame = np.reshape(array, (height, width, 3)).astype(np.uint8)

        # Display
        cv2.imshow("Live Feed (uEye Processed)", frame)

        # Capture image manually, change this to a periodic capture
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            timestamp = int(time.time())
            filename = os.path.join("images", f"frame_{timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")

        if key == ord('q'):
            break

        if key == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cv2.destroyAllWindows()
    ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
    ueye.is_ExitCamera(hCam)


if __name__ == "__main__":
    main()