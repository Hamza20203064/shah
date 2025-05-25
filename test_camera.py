import cv2
import time


def test_camera():
    print("Testing camera...")

    # Try different camera indices
    for i in range(10):  # Try first 10 indices
        try:
            print(f"\nTrying camera index: {i}")
            cap = cv2.VideoCapture(i)

            if not cap.isOpened():
                print(f"Failed to open camera index {i}")
                continue

            # Try to read a frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Could not read frame from camera index {i}")
                cap.release()
                continue

            print(f"Successfully opened camera index {i}")
            print(f"Frame shape: {frame.shape}")

            # Show the frame
            cv2.imshow('Test Frame', frame)
            print("Press any key to continue...")
            cv2.waitKey(0)

            cap.release()
            cv2.destroyAllWindows()
            return True

        except Exception as e:
            print(f"Error with camera index {i}: {str(e)}")
            if 'cap' in locals():
                cap.release()

    print("\nNo working camera found")
    return False


if __name__ == "__main__":
    test_camera()
