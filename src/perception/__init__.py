"""Unified detection and instance-segmentation perception pipeline."""

from .fusion import calculate_iou, match_detections_and_segments
from .pipeline import PerceptionPipeline

__all__ = ["PerceptionPipeline", "calculate_iou", "match_detections_and_segments"]
