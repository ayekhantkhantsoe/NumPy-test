"""Run live heart detection from the default webcam."""

import cv2

from image_processing import process_frame


def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera could not be opened.")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to capture frame from camera.")
                break

            processed_frame = process_frame(frame)
            cv2.imshow("Webcam Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
