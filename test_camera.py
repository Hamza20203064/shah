import cv2
import time


def test_camera():
    print("Testing camera...")

    # Try different camera devices
    camera_devices = [
        "/dev/video10",
        "/dev/video11",
        "/dev/video12",
        "/dev/video13",
        "/dev/video14",
        "/dev/video15",
        "/dev/video16",
        "/dev/video18",
        "/dev/video19",
        "/dev/video20",
        "/dev/video21",
        "/dev/video22",
        "/dev/video23",
        "/dev/video31"
    ]

    for device in camera_devices:
        try:
            print(f"\nTrying camera device: {device}")
            cap = cv2.VideoCapture(device)

            if not cap.isOpened():
                print(f"Failed to open {device}")
                continue

            # Try to read a frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Could not read frame from {device}")
                cap.release()
                continue

            print(f"Successfully opened {device}")
            print(f"Frame shape: {frame.shape}")

            # Show the frame
            cv2.imshow('Test Frame', frame)
            print("Press any key to continue...")
            cv2.waitKey(0)

            cap.release()
            cv2.destroyAllWindows()
            return True

        except Exception as e:
            print(f"Error with {device}: {str(e)}")
            if 'cap' in locals():
                cap.release()

    print("\nNo working camera found")
    return False


if __name__ == "__main__":
    test_camera()
