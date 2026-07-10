"""Depth visualization that never alters metric source values."""

from __future__ import annotations

from typing import Any

import numpy as np

from .postprocessing import create_valid_depth_mask, validate_depth_map


def normalize_depth_for_visualization(
    depth_map: Any,
    percentile_min: float = 2.0,
    percentile_max: float = 98.0,
) -> np.ndarray:
    if not 0 <= percentile_min < percentile_max <= 100:
        raise ValueError("Visualization percentiles must satisfy 0 <= min < max <= 100.")
    depth = validate_depth_map(depth_map)
    valid = create_valid_depth_mask(depth)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    values = depth[valid]
    if not values.size:
        return normalized
    low, high = np.percentile(values, [percentile_min, percentile_max])
    if high <= low:
        normalized[valid] = 255
        return normalized
    scaled = (np.clip(depth[valid], low, high) - low) / (high - low)
    normalized[valid] = np.rint(scaled * 255).astype(np.uint8)
    return normalized


def colorize_depth_map(
    depth_map: Any,
    percentile_min: float = 2.0,
    percentile_max: float = 98.0,
) -> np.ndarray:
    import cv2

    depth = validate_depth_map(depth_map)
    valid = create_valid_depth_mask(depth)
    normalized = normalize_depth_for_visualization(depth, percentile_min, percentile_max)
    color = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def draw_depth_overlay(
    frame: Any,
    depth_map: Any,
    alpha: float = 0.45,
    percentile_min: float = 2.0,
    percentile_max: float = 98.0,
) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError("Depth overlay alpha must be between 0 and 1.")
    depth = validate_depth_map(depth_map)
    if tuple(frame.shape[:2]) != tuple(depth.shape):
        raise ValueError(f"Depth map shape {depth.shape} does not match frame shape {frame.shape[:2]}.")
    import cv2

    color = colorize_depth_map(depth, percentile_min, percentile_max)
    return cv2.addWeighted(color, alpha, frame.copy(), 1 - alpha, 0)
