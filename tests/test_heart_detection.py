"""Exercise the real HSV/contour pipeline with generated red shapes.

Run: python -m unittest discover -s tests -v
Only file access and GUI operations are replaced, so tests preserve your images.
"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import cv2
import numpy as np


MAIN_PATH = Path(__file__).resolve().parents[1] / "static_image_lesson.py"


def heart_points(size: int = 180, offset: tuple[int, int] = (60, 50)) -> np.ndarray:
    """Sample a standard heart curve, then fit it into a square."""
    t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t)
          - 2 * np.cos(3 * t) - np.cos(4 * t))
    points = np.column_stack((x, y))
    points = (points - points.min(axis=0)) / np.ptp(points, axis=0)
    return np.rint(points * size + offset).astype(np.int32)


def blank_image() -> np.ndarray:
    return np.zeros((300, 300, 3), dtype=np.uint8)


def run_pipeline(image: np.ndarray) -> list[str]:
    """Capture labels while allowing OpenCV to actually draw each label."""
    labels: list[str] = []
    real_put_text = cv2.putText

    def record_text(canvas, text, *args, **kwargs):
        labels.append(text)
        return real_put_text(canvas, text, *args, **kwargs)

    with patch("cv2.imread", return_value=image.copy()), \
            patch("cv2.imwrite", return_value=True), \
            patch("cv2.imshow"), patch("cv2.waitKey", return_value=0), \
            patch("cv2.destroyAllWindows"), \
            patch("cv2.putText", side_effect=record_text), \
            redirect_stdout(io.StringIO()):
        runpy.run_path(str(MAIN_PATH), run_name="__main__")
    return labels


class HeartDetectionTests(unittest.TestCase):
    def test_upright_heart_is_labeled_heart(self) -> None:
        image = blank_image()
        cv2.fillPoly(image, [heart_points()], (0, 0, 255))
        self.assertEqual(run_pipeline(image), ["Heart"])

    def test_heart_detection_handles_scale_and_translation(self) -> None:
        for size, offset in ((50, (10, 15)), (100, (175, 150)), (220, (35, 30))):
            with self.subTest(size=size, offset=offset):
                image = blank_image()
                cv2.fillPoly(image, [heart_points(size, offset)], (0, 0, 255))
                self.assertEqual(run_pipeline(image), ["Heart"])

    def test_hand_drawn_polygon_heart_is_recognized(self) -> None:
        image = blank_image()
        points = np.array([
            [150, 90], [120, 60], [85, 55], [60, 80], [55, 115],
            [75, 150], [150, 235], [225, 150], [245, 115],
            [240, 80], [215, 55], [180, 60],
        ], dtype=np.int32)
        cv2.fillPoly(image, [points], (0, 0, 255))
        self.assertEqual(run_pipeline(image), ["Heart"])

    def test_every_large_contour_gets_its_own_label(self) -> None:
        image = blank_image()
        cv2.fillPoly(image, [heart_points(100, (15, 30))], (0, 0, 255))
        cv2.rectangle(image, (180, 170), (270, 260), (0, 0, 255), -1)
        self.assertCountEqual(run_pipeline(image), ["Heart", "Rectangle"])

    def test_no_contours_does_not_crash_or_draw_a_label(self) -> None:
        self.assertEqual(run_pipeline(blank_image()), [])

    def test_small_noise_is_not_labeled(self) -> None:
        image = blank_image()
        cv2.rectangle(image, (120, 120), (124, 124), (0, 0, 255), -1)
        self.assertEqual(run_pipeline(image), [])

    def test_existing_triangle_and_rectangle_labels_are_preserved(self) -> None:
        for name, points in (
            ("Triangle", [[150, 40], [40, 250], [260, 250]]),
            ("Rectangle", [[50, 50], [250, 50], [250, 250], [50, 250]]),
        ):
            with self.subTest(shape=name):
                image = blank_image()
                cv2.fillPoly(image, [np.array(points, dtype=np.int32)], (0, 0, 255))
                self.assertEqual(run_pipeline(image), [name])

    def test_circle_star_and_u_shape_are_not_hearts(self) -> None:
        circle = blank_image()
        cv2.circle(circle, (150, 150), 90, (0, 0, 255), -1)
        star = blank_image()
        angles = np.arange(10) * np.pi / 5 - np.pi / 2
        radii = np.where(np.arange(10) % 2 == 0, 100, 40)
        points = np.column_stack((150 + radii * np.cos(angles),
                                  150 + radii * np.sin(angles))).astype(np.int32)
        cv2.fillPoly(star, [points], (0, 0, 255))
        u_shape = blank_image()
        cv2.rectangle(u_shape, (60, 60), (240, 240), (0, 0, 255), -1)
        cv2.rectangle(u_shape, (110, 50), (190, 180), (0, 0, 0), -1)
        for name, image in (("circle", circle), ("star", star), ("U", u_shape)):
            with self.subTest(shape=name):
                labels = run_pipeline(image)
                self.assertEqual(len(labels), 1)
                self.assertNotIn("Heart", labels)

    def test_sideways_and_upside_down_hearts_are_outside_upright_rules(self) -> None:
        upright = blank_image()
        cv2.fillPoly(upright, [heart_points()], (0, 0, 255))
        for rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180):
            with self.subTest(rotation=rotation):
                self.assertNotIn("Heart", run_pipeline(cv2.rotate(upright, rotation)))


if __name__ == "__main__":
    unittest.main()
