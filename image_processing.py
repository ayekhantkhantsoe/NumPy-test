"""Image processing shared by the webcam application and tests."""

import cv2
import numpy as np

from heart_detection import is_heart_contour


def process_frame(frame: np.ndarray) -> np.ndarray:
    """Return a 300 x 300 copy with boxes around red upright hearts."""
    resized_frame = cv2.resize(frame, (300, 300))
    result = resized_frame.copy()

    hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 50], dtype=np.uint8)
    upper_red = np.array([10, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        if not is_heart_contour(contour):
            continue

        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(
            result, "Heart", (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    return result
