"""Typed records for BEV outputs."""

from __future__ import annotations

from typing import TypedDict


class BEVFrameRecord(TypedDict, total=False):
    coordinate_frame: str
    projection_type: str
    unit: str
    resolution_m: float
    x_range_m: list[float]
    z_range_m: list[float]
    shape: list[int]
    class_grid_path: str
    class_grid_png_path: str
    drivable_mask_path: str
    non_drivable_mask_path: str
    visualization_path: str
    observed_cell_count: int
    drivable_cell_count: int
    non_drivable_cell_count: int
    unknown_cell_count: int
    has_semantic_labels: bool
    conflict_policy: str
