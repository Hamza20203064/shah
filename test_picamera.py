import cv2
import time


def test_camera():
    try:
        print("Initializing camera...")
        # Try different camera indices
        for camera_index in range(10):
            print(f"Trying camera index {camera_index}...")
            cap = cv2.VideoCapture(camera_index)

            if cap.isOpened():
                print(f"Successfully opened camera {camera_index}")
                # Read a frame
                ret, frame = cap.read()

                if ret:
                    print(
                        f"Successfully captured frame from camera {camera_index}")
                    print(f"Frame shape: {frame.shape}")

                    # Display the frame
                    cv2.imshow('Test Frame', frame)
                    print("Press any key to continue...")
                    cv2.waitKey(0)

                    # Clean up
                    cv2.destroyAllWindows()
                    cap.release()
                    return True
                else:
                    print(
                        f"Failed to capture frame from camera {camera_index}")

            cap.release()
            print(f"Camera {camera_index} not available")

        print("No working camera found")
        return False

    except Exception as e:
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    test_camera()
