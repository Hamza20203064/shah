import cv2
import time


def test_camera():
    try:
        print("Initializing camera...")
        # List of video devices to try
        video_devices = [
            '/dev/video10', '/dev/video11', '/dev/video12', '/dev/video13',
            '/dev/video14', '/dev/video15', '/dev/video16', '/dev/video18',
            '/dev/video19', '/dev/video20', '/dev/video21', '/dev/video22',
            '/dev/video23', '/dev/video31'
        ]

        for device in video_devices:
            print(f"Trying device {device}...")
            cap = cv2.VideoCapture(device)

            if cap.isOpened():
                print(f"Successfully opened {device}")
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
                    print(f"Failed to capture frame from {device}")

            cap.release()
            print(f"Device {device} not available")

        print("No working camera found")
        return False

    except Exception as e:
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    test_camera()
