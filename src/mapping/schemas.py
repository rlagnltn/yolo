from __future__ import annotations

from typing import TypedDict


class MappingFrameRecord(TypedDict, total=False):
    coordinate_frame: str
    grid_type: str
    resolution_m: float
    shape: list[int]
    occupancy_grid_path: str
    cost_grid_path: str
    inflated_cost_grid_path: str
    free_cell_count: int
    occupied_cell_count: int
    unknown_cell_count: int
    inflated_cell_count: int
    inflation_radius_m: float
