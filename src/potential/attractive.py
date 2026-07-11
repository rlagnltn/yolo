from __future__ import annotations

from typing import Any
import numpy as np

from .goal import validate_goal_cell


def create_attractive_potential(
    grid_shape: tuple[int, int], goal_cell: tuple[int, int], resolution_m: float, gain: float,
    mode: str = "quadratic", saturation_distance_m: float | None = None,
) -> np.ndarray:
    if len(grid_shape) != 2 or min(grid_shape) <= 0 or not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("Grid shape and resolution must be valid.")
    if not np.isfinite(gain) or gain < 0 or mode not in {"quadratic", "conic"}:
        raise ValueError("Attractive potential parameters are invalid.")
    if saturation_distance_m is not None and (not np.isfinite(saturation_distance_m) or saturation_distance_m <= 0):
        raise ValueError("Saturation distance must be positive and finite.")
    row, col = validate_goal_cell(*goal_cell, grid_shape)
    rows, cols = np.indices(grid_shape, dtype=np.float32)
    distance = np.hypot(rows - row, cols - col) * np.float32(resolution_m)
    if mode == "conic":
        return (np.float32(gain) * distance).astype(np.float32)
    potential = np.float32(0.5 * gain) * distance * distance
    if saturation_distance_m is not None:
        saturated = distance > saturation_distance_m
        potential[saturated] = np.float32(gain * saturation_distance_m) * (
            distance[saturated] - np.float32(0.5 * saturation_distance_m)
        )
    return potential.astype(np.float32)
