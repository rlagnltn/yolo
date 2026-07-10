"""Typed records for geometry outputs."""

from __future__ import annotations

from typing import TypedDict


class CameraIntrinsicsRecord(TypedDict):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class GeometryFrameRecord(TypedDict):
    point_cloud_path: str
    coordinate_frame: str
    unit: str
    point_count: int
    stride: int
    depth_range_m: list[float | None]
    intrinsics: CameraIntrinsicsRecord
    has_semantic_labels: bool
