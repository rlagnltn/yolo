"""OpenCV BGR visualizations for mapping grids."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .config import MappingConfig


def colorize_occupancy_grid(grid: Any, config: MappingConfig, *, grayscale: bool = False) -> np.ndarray:
    values = np.asarray(grid)
    if values.ndim != 2:
        raise ValueError("occupancy grid must be 2D.")
    image = np.full(values.shape, 127, dtype=np.uint8)
    image[values == config.free_value] = 255
    image[values == config.occupied_value] = 0
    known = np.isin(values, [config.unknown_value, config.free_value, config.occupied_value])
    if not known.all():
        raise ValueError("occupancy grid contains an unsupported state value.")
    return image if grayscale else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def colorize_cost_grid(cost_grid: Any) -> np.ndarray:
    costs = np.asarray(cost_grid, dtype=np.float32)
    if costs.ndim != 2:
        raise ValueError("cost grid must be 2D.")
    finite = np.isfinite(costs)
    if np.any((costs[finite] < 0.0) | (costs[finite] > 1.0)):
        raise ValueError("Finite costs must be within [0, 1].")
    scaled = np.zeros(costs.shape, dtype=np.uint8)
    scaled[finite] = np.rint(np.clip(costs[finite], 0.0, 1.0) * 255.0).astype(np.uint8)
    image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    image[~finite] = (127, 127, 127)
    return image
