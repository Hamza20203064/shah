from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import cv2


def test_picamera():
    try:
        print("Initializing PiCamera...")
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 30
        raw_capture = PiRGBArray(camera, size=(640, 480))

        print("Camera initialized. Capturing frame...")
        # Allow the camera to warm up
        time.sleep(2)

        # Capture a frame
        camera.capture(raw_capture, format="rgb")
        frame = raw_capture.array

        print(f"Frame captured. Shape: {frame.shape}")

        # Convert to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow('Test Frame', frame)
        print("Press any key to continue...")
        cv2.waitKey(0)

        # Clean up
        cv2.destroyAllWindows()
        camera.close()
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    test_picamera()
