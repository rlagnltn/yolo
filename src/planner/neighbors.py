"""Shared deterministic grid movement rules for selection and planning."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

FOUR_OFFSETS = ((-1, 0), (0, 1), (1, 0), (0, -1))
EIGHT_OFFSETS = FOUR_OFFSETS + ((-1, 1), (1, 1), (1, -1), (-1, -1))


def iter_free_neighbors(
    occupancy_grid: np.ndarray,
    cell: tuple[int, int],
    *,
    connectivity: int = 8,
    prevent_corner_cutting: bool = True,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield FREE neighbor cells and their offsets using planner-compatible rules."""

    grid = np.asarray(occupancy_grid)
    offsets = EIGHT_OFFSETS if connectivity == 8 else FOUR_OFFSETS
    row, col = cell
    for dr, dc in offsets:
        next_row, next_col = row + dr, col + dc
        if not (0 <= next_row < grid.shape[0] and 0 <= next_col < grid.shape[1]):
            continue
        if grid[next_row, next_col] != 0:
            continue
        if dr and dc and prevent_corner_cutting:
            if grid[row, next_col] != 0 or grid[next_row, col] != 0:
                continue
        yield next_row, next_col, dr, dc
