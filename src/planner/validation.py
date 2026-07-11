from __future__ import annotations
import numpy as np

def validate_start_cell(start_cell, occupancy_grid):
    grid = np.asarray(occupancy_grid)
    if grid.ndim != 2: raise ValueError("Occupancy grid must be 2D.")
    row, col = (int(start_cell[0]), int(start_cell[1]))
    if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]): raise ValueError("Start cell is outside the grid bounds.")
    if grid[row, col] != 0: raise ValueError("Start cell must be an observed FREE cell.")
    return row, col

def validate_grid_path(path_rc, occupancy_grid, start_cell, goal_cell, connectivity, goal_tolerance_cells, prevent_corner_cutting=True):
    path = np.asarray(path_rc, dtype=np.int32); grid = np.asarray(occupancy_grid)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) == 0: raise ValueError("Grid path must have non-empty shape (N, 2).")
    if tuple(path[0]) != tuple(start_cell): raise ValueError("Path must start at start_cell.")
    for index, (row, col) in enumerate(path):
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]) or grid[row, col] != 0: raise ValueError("Path contains a non-FREE or out-of-bounds cell.")
        if index:
            dr, dc = int(row-path[index-1,0]), int(col-path[index-1,1])
            if max(abs(dr),abs(dc)) != 1 or (connectivity == 4 and dr and dc): raise ValueError("Path has an invalid move.")
            if prevent_corner_cutting and dr and dc and (grid[path[index-1,0], col] != 0 or grid[row, path[index-1,1]] != 0): raise ValueError("Path cuts an occupied or unknown corner.")
    if len({tuple(cell) for cell in path}) != len(path): raise ValueError("Path contains duplicate cells.")
    if np.linalg.norm(path[-1].astype(float)-np.asarray(goal_cell, float)) > goal_tolerance_cells: raise ValueError("Path does not end within goal tolerance.")
    return True
