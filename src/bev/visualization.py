"""Minimal OpenCV BGR BEV visualizations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import BEVConfig

UNKNOWN_BGR = (48, 48, 48)
GRID_BGR = (80, 80, 80)
ORIGIN_BGR = (0, 255, 255)


def colorize_bev_class_grid(class_grid: Any, observed_mask: Any, config: BEVConfig) -> np.ndarray:
    from src.scene_segmentation.class_mapping import CITYSCAPES_BGR_PALETTE

    config.validate()
    grid = np.asarray(class_grid)
    observed = np.asarray(observed_mask, dtype=bool)
    if grid.shape != config.shape or observed.shape != config.shape:
        raise ValueError("BEV grid and observed mask must match config shape.")
    image = np.zeros((config.height_cells, config.width_cells, 3), dtype=np.uint8)
    image[:] = UNKNOWN_BGR
    for class_id, color in CITYSCAPES_BGR_PALETTE.items():
        image[observed & (grid == class_id)] = color
    return image


def draw_bev_grid_lines(image: Any, spacing_cells: int = 10) -> np.ndarray:
    canvas = np.asarray(image).copy()
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("BEV visualization image must have shape (H, W, 3).")
    if spacing_cells <= 0:
        raise ValueError("spacing_cells must be positive.")
    canvas[::spacing_cells, :, :] = GRID_BGR
    canvas[:, ::spacing_cells, :] = GRID_BGR
    return canvas


def mark_camera_origin(image: Any) -> np.ndarray:
    canvas = np.asarray(image).copy()
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("BEV visualization image must have shape (H, W, 3).")
    row = canvas.shape[0] - 1
    col = canvas.shape[1] // 2
    row_min, row_max = max(0, row - 2), row + 1
    col_min, col_max = max(0, col - 2), min(canvas.shape[1], col + 3)
    canvas[row_min:row_max, col_min:col_max] = ORIGIN_BGR
    return canvas
