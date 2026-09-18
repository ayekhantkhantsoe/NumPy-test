"""Rules for deciding whether a red contour looks like an upright heart."""

import cv2
import numpy as np


def is_heart_contour(contour: np.ndarray) -> bool:
    if len(contour) < 7 or cv2.contourArea(contour) <= 0:
        return False

    x, y, width, height = cv2.boundingRect(contour)
    aspect_ratio = width / height
    if not 0.65 <= aspect_ratio <= 1.4:
        return False

    perimeter = cv2.arcLength(contour, True)
    outline = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    if len(outline) < 7:
        return False

    # Use the original contour. Approximation can introduce self-intersections.
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return False
    hull_indices = np.unique(hull_indices).astype(np.int32).reshape(-1, 1)
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return False
    if defects is None:
        return False

    deep_defects = [
        defect for defect in defects.reshape(-1, 4)
        if defect[3] / 256.0 >= 0.05 * height
    ]
    if len(deep_defects) != 1:
        return False

    start, end, farthest, depth = deep_defects[0]
    shoulders = sorted(
        (contour[start, 0], contour[end, 0]), key=lambda point: point[0]
    )
    left, right = shoulders
    notch = contour[farthest, 0]

    notch_x = (notch[0] - x) / width
    notch_y = (notch[1] - y) / height
    if not (0.30 <= notch_x <= 0.70 and 0.08 <= notch_y <= 0.50):
        return False
    if not 0.06 <= depth / 256.0 / height <= 0.35:
        return False
    if max(left[1], right[1]) > y + 0.35 * height:
        return False
    if abs(int(left[1]) - int(right[1])) > 0.20 * height:
        return False
    if not (left[0] + 0.10 * width < notch[0] < right[0] - 0.10 * width):
        return False
    if notch[1] < max(left[1], right[1]) + 0.04 * height:
        return False

    points = contour[:, 0, :]
    bottom_points = points[points[:, 1] >= y + 0.85 * (height - 1)]
    if len(bottom_points) == 0:
        return False

    bottom_width = np.ptp(bottom_points[:, 0])
    bottom_center = (
        float(bottom_points[:, 0].min()) + float(bottom_points[:, 0].max())
    ) / 2
    return bool(
        bottom_width < 0.45 * width
        and 0.35 <= (bottom_center - x) / width <= 0.65
    )
