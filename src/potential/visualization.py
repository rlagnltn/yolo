from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def colorize_potential(potential: Any, blocked_mask: Any | None = None) -> np.ndarray:
    values = np.asarray(potential, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("potential must be a finite 2D array.")
    blocked = np.zeros(values.shape, bool) if blocked_mask is None else np.asarray(blocked_mask, dtype=bool)
    if blocked.shape != values.shape:
        raise ValueError("blocked_mask must match potential shape.")
    valid = ~blocked
    scaled = np.zeros(values.shape, dtype=np.uint8)
    if valid.any():
        low, high = float(values[valid].min()), float(values[valid].max())
        if high > low:
            scaled[valid] = np.rint((values[valid] - low) * 255.0 / (high - low)).astype(np.uint8)
    image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    image[blocked] = (40, 40, 40)
    return image


def draw_goal_marker(image: Any, goal_cell: tuple[int, int]) -> np.ndarray:
    result = np.asarray(image).copy()
    row, col = goal_cell
    cv2.drawMarker(result, (int(col), int(row)), (0, 255, 0), cv2.MARKER_CROSS, 9, 1)
    return result


def draw_gradient_vectors(image: Any, gradient: dict[str, np.ndarray], step: int = 16) -> np.ndarray:
    result = np.asarray(image).copy()
    descent_x, descent_z = gradient["descent_x"], gradient["descent_z"]
    for row in range(0, result.shape[0], step):
        for col in range(0, result.shape[1], step):
            cv2.arrowedLine(result, (col, row), (int(col + descent_x[row, col]), int(row - descent_z[row, col])), (255, 255, 255), 1, tipLength=0.25)
    return result
