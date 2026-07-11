from __future__ import annotations

from typing import TypedDict


class PotentialFrameRecord(TypedDict, total=False):
    coordinate_frame: str
    grid_type: str
    resolution_m: float
    shape: list[int]
    goal: dict[str, float | int]
    attractive_path: str
    repulsive_path: str
    combined_path: str
    gradient_path: str
    unknown_policy: str
