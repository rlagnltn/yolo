"""JSON-friendly schemas for unified perception output."""

from __future__ import annotations

from typing import TypedDict

from src.scene_segmentation.schemas import SceneFrameRecord


class DetectionRecord(TypedDict):
    object_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


class SegmentRecord(TypedDict, total=False):
    segment_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    mask_area: int
    mask_path: str


class FusedObjectRecord(TypedDict, total=False):
    object_id: str
    class_id: int
    class_name: str
    detection_confidence: float | None
    segmentation_confidence: float | None
    bbox_xyxy: list[float]
    mask_area: int
    mask_path: str
    fusion_status: str


class FrameRecord(TypedDict):
    frame_index: int
    timestamp_sec: float
    width: int
    height: int
    detections: list[DetectionRecord]
    segments: list[SegmentRecord]
    fused_objects: list[FusedObjectRecord]
    scene_segmentation: SceneFrameRecord | None
    errors: list[str]
