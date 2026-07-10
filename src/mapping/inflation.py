"""Metric obstacle inflation for traversability costs."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def inflate_cost_grid(
    cost_grid: Any,
    occupied_mask: Any,
    resolution_m: float,
    inflation_radius_m: float,
    decay: str = "linear",
) -> np.ndarray:
    costs = np.asarray(cost_grid, dtype=np.float32)
    occupied = np.asarray(occupied_mask, dtype=bool)
    if costs.ndim != 2 or occupied.shape != costs.shape:
        raise ValueError("cost_grid and occupied_mask must be matching 2D arrays.")
    if not np.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError("resolution_m must be finite and greater than zero.")
    if not np.isfinite(inflation_radius_m) or inflation_radius_m < 0.0:
        raise ValueError("inflation_radius_m must be finite and non-negative.")
    if decay != "linear":
        raise ValueError("Only linear inflation decay is supported.")
    result = costs.copy()
    if inflation_radius_m == 0.0 or not occupied.any():
        return result
    distance_cells = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance_m = distance_cells * np.float32(resolution_m)
    inflated = np.clip(1.0 - distance_m / np.float32(inflation_radius_m), 0.0, 1.0)
    finite = np.isfinite(result)
    result[finite] = np.maximum(result[finite], inflated[finite]).astype(np.float32)
    result[occupied] = 1.0
    return result
