"""JSON-friendly scene semantic-segmentation records."""

from __future__ import annotations

from typing import TypedDict


class ClassStatistic(TypedDict):
    class_id: int
    class_name: str
    pixel_count: int
    pixel_ratio: float


class RegionStatistics(TypedDict):
    drivable_pixel_count: int
    drivable_pixel_ratio: float
    non_drivable_pixel_count: int
    unknown_pixel_count: int


class SceneFrameRecord(TypedDict, total=False):
    frame_index: int
    timestamp_sec: float
    width: int
    height: int
    class_map_path: str
    color_map_path: str
    overlay_path: str
    drivable_mask_path: str
    non_drivable_mask_path: str
    class_statistics: list[ClassStatistic]
    regions: RegionStatistics
