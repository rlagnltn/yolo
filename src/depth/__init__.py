"""Depth estimation extension point."""
"""Monocular metric depth-estimation utilities."""

from .estimator import DEFAULT_MODEL, DepthEstimator
from .postprocessing import (
    calculate_depth_by_class,
    calculate_depth_statistics,
    clip_depth_range,
    create_valid_depth_mask,
    depth_to_uint16,
    resize_depth_map,
    sample_depth_at_points,
    sanitize_depth_map,
    uint16_to_depth,
    validate_depth_map,
)

__all__ = [
    "DEFAULT_MODEL", "DepthEstimator", "calculate_depth_by_class",
    "calculate_depth_statistics", "clip_depth_range", "create_valid_depth_mask",
    "depth_to_uint16", "resize_depth_map", "sample_depth_at_points",
    "sanitize_depth_map", "uint16_to_depth", "validate_depth_map",
]
