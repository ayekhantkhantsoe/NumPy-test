import unittest
from unittest.mock import patch

import cv2
import numpy as np

from image_processing import process_frame


class WebcamProcessingTests(unittest.TestCase):
    def test_blank_frame_remains_blank(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)

        result = process_frame(frame)

        self.assertEqual(result.shape, (300, 300, 3))
        self.assertTrue(np.array_equal(result, frame))

    def test_red_heart_gets_a_heart_label(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        points = np.array([
            [150, 85], [120, 55], [85, 50], [60, 80], [55, 115],
            [75, 150], [150, 235], [225, 150], [245, 115],
            [240, 80], [215, 50], [180, 55],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [points], (0, 0, 255))
        labels = []
        real_put_text = cv2.putText

        def record_label(image, label, *args, **kwargs):
            labels.append(label)
            return real_put_text(image, label, *args, **kwargs)

        with patch("cv2.putText", side_effect=record_label):
            result = process_frame(frame)

        self.assertEqual(labels, ["Heart"])
        self.assertEqual(result.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
