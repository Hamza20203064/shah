import cv2
import time


def test_camera():
    try:
        print("Initializing camera...")
        # List of video devices to try
        video_devices = [
            # Only try the devices that were detected
            '/dev/video14', '/dev/video15', '/dev/video21', '/dev/video22'
        ]

        for device in video_devices:
            print(f"\nTrying device {device}...")
            cap = cv2.VideoCapture(device)

            if cap.isOpened():
                print(f"Successfully opened {device}")

                # Try different formats
                formats = [
                    (640, 480),
                    (320, 240),
                    (1280, 720)
                ]

                for width, height in formats:
                    print(f"Trying format: {width}x{height}")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                    # Add a delay to let the camera initialize
                    time.sleep(2)

                    # Read a frame
                    ret, frame = cap.read()

                    if ret:
                        print(f"Successfully captured frame from {device}")
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
                            f"Failed to capture frame with format {width}x{height}")

            cap.release()
            print(f"Device {device} not available")

        print("\nNo working camera found")
        return False

    except Exception as e:
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    test_camera()
