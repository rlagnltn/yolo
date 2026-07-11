from __future__ import annotations
from typing import TypedDict
class PlannerFrameRecord(TypedDict, total=False):
    coordinate_frame: str
    algorithm: str
    status: str
    reached_goal: bool
    grid_path_path: str
    metric_path_path: str
    visualization_path: str
