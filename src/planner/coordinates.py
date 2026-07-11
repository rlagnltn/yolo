from __future__ import annotations
import numpy as np

def grid_path_to_metric(path_rc, bev_config):
    path = np.asarray(path_rc, dtype=np.int32)
    if path.ndim != 2 or path.shape[1] != 2: raise ValueError("Grid path must have shape (N, 2).")
    rows, cols = path[:, 0], path[:, 1]
    x = bev_config.x_min_m + (cols + .5) * bev_config.resolution_m
    z = bev_config.z_min_m + (bev_config.height_cells - rows - .5) * bev_config.resolution_m
    return np.column_stack((x, z)).astype(np.float32)

def metric_path_to_grid(path_xz, bev_config):
    path = np.asarray(path_xz, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2: raise ValueError("Metric path must have shape (N, 2).")
    x, z = path[:, 0], path[:, 1]
    if np.any((x < bev_config.x_min_m) | (x >= bev_config.x_max_m) | (z < bev_config.z_min_m) | (z >= bev_config.z_max_m)): raise ValueError("Metric path is outside BEV bounds.")
    cols = np.floor((x - bev_config.x_min_m) / bev_config.resolution_m).astype(np.int32)
    near_rows = np.floor((z - bev_config.z_min_m) / bev_config.resolution_m).astype(np.int32)
    return np.column_stack((bev_config.height_cells - 1 - near_rows, cols)).astype(np.int32)

def resolve_start_cell(grid_start=None, metric_start=None, bev_config=None):
    if (grid_start is None) == (metric_start is None): raise ValueError("Provide exactly one grid or metric start.")
    if grid_start is not None:
        if len(grid_start) != 2: raise ValueError("Grid start requires row and col.")
        row, col = grid_start
        if int(row) != row or int(col) != col: raise ValueError("Grid start indices must be integers.")
        return int(row), int(col)
    return tuple(metric_path_to_grid(np.asarray([metric_start]), bev_config)[0])

def calculate_path_length_m(path_xz):
    path = np.asarray(path_xz, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2: raise ValueError("Metric path must have shape (N, 2).")
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0
