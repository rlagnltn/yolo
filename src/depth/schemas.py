"""JSON-friendly depth output records."""

from __future__ import annotations

from typing import TypedDict


class DepthStatistics(TypedDict):
    min_depth: float | None
    max_depth: float | None
    mean_depth: float | None
    median_depth: float | None
    standard_deviation: float | None
    percentile_05: float | None
    percentile_25: float | None
    percentile_75: float | None
    percentile_95: float | None
    valid_pixel_count: int
    invalid_pixel_count: int
    valid_pixel_ratio: float


class DepthFrameRecord(TypedDict, total=False):
    frame_index: int
    timestamp_sec: float
    width: int
    height: int
    depth_type: str
    unit: str
    model_name: str
    requested_model_name: str
    raw_depth_path: str
    depth_png_path: str
    color_map_path: str
    overlay_path: str
    png_scale: float
    statistics: DepthStatistics
    depth_by_scene_class: dict[str, dict[str, float | int]]
