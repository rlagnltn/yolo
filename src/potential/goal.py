from __future__ import annotations

from typing import Any
import numpy as np


def metric_goal_to_grid(x_m: float, z_m: float, bev_config: Any) -> tuple[int, int]:
    if not np.isfinite(x_m) or not np.isfinite(z_m):
        raise ValueError("Metric goal coordinates must be finite.")
    if not (bev_config.x_min_m <= x_m < bev_config.x_max_m and bev_config.z_min_m <= z_m < bev_config.z_max_m):
        raise ValueError("Metric goal is outside the BEV bounds.")
    col = int(np.floor((x_m - bev_config.x_min_m) / bev_config.resolution_m))
    row_from_near = int(np.floor((z_m - bev_config.z_min_m) / bev_config.resolution_m))
    return bev_config.height_cells - 1 - row_from_near, col


def validate_goal_cell(goal_row: int, goal_col: int, grid_shape: tuple[int, int]) -> tuple[int, int]:
    if isinstance(goal_row, bool) or isinstance(goal_col, bool):
        raise ValueError("Goal row and column must be integer indices.")
    if int(goal_row) != goal_row or int(goal_col) != goal_col:
        raise ValueError("Goal row and column must be integer indices.")
    row, col = int(goal_row), int(goal_col)
    if not (0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]):
        raise ValueError("Goal cell is outside the grid bounds.")
    return row, col
