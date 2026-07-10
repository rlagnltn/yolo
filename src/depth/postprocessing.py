"""Validation and vectorized post-processing for metric depth maps."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def validate_depth_map(depth_map: Any, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(depth_map)
    if array.ndim != 2:
        raise ValueError(f"Depth map must be 2-dimensional, got shape {array.shape}.")
    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"Depth map shape {array.shape} does not match expected shape {expected_shape}.")
    return array.astype(np.float32, copy=False)


def sanitize_depth_map(depth_map: Any, invalid_value: float = 0.0) -> np.ndarray:
    array = validate_depth_map(depth_map).copy()
    invalid = ~np.isfinite(array) | (array <= 0)
    array[invalid] = np.float32(invalid_value)
    return array


def resize_depth_map(depth_map: Any, height: int, width: int) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("Depth output height and width must be positive.")
    import cv2

    array = validate_depth_map(depth_map)
    resized = cv2.resize(array, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
    return sanitize_depth_map(resized)


def create_valid_depth_mask(depth_map: Any) -> np.ndarray:
    array = validate_depth_map(depth_map)
    return np.isfinite(array) & (array > 0)


def clip_depth_range(
    depth_map: Any,
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
) -> np.ndarray:
    if min_depth_m is not None and min_depth_m <= 0:
        raise ValueError("min_depth_m must be positive.")
    if max_depth_m is not None and max_depth_m <= 0:
        raise ValueError("max_depth_m must be positive.")
    if min_depth_m is not None and max_depth_m is not None and min_depth_m > max_depth_m:
        raise ValueError("min_depth_m cannot exceed max_depth_m.")
    array = sanitize_depth_map(depth_map)
    valid = array > 0
    if min_depth_m is not None:
        array[valid] = np.maximum(array[valid], np.float32(min_depth_m))
    if max_depth_m is not None:
        array[valid] = np.minimum(array[valid], np.float32(max_depth_m))
    return array


def calculate_depth_statistics(depth_map: Any) -> dict[str, float | int | None]:
    array = validate_depth_map(depth_map)
    valid_mask = create_valid_depth_mask(array)
    values = array[valid_mask].astype(np.float64)
    total = int(array.size)
    valid_count = int(values.size)
    result: dict[str, float | int | None] = {
        "min_depth": None, "max_depth": None, "mean_depth": None,
        "median_depth": None, "standard_deviation": None,
        "percentile_05": None, "percentile_25": None,
        "percentile_75": None, "percentile_95": None,
        "valid_pixel_count": valid_count,
        "invalid_pixel_count": total - valid_count,
        "valid_pixel_ratio": float(valid_count / total) if total else 0.0,
    }
    if valid_count:
        result.update({
            "min_depth": float(np.min(values)), "max_depth": float(np.max(values)),
            "mean_depth": float(np.mean(values)), "median_depth": float(np.median(values)),
            "standard_deviation": float(np.std(values)),
            "percentile_05": float(np.percentile(values, 5)),
            "percentile_25": float(np.percentile(values, 25)),
            "percentile_75": float(np.percentile(values, 75)),
            "percentile_95": float(np.percentile(values, 95)),
        })
    return result


def depth_to_uint16(depth_map: Any, scale: float = 1000.0) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Depth PNG scale must be a positive finite number.")
    array = sanitize_depth_map(depth_map)
    scaled = np.rint(array.astype(np.float64) * float(scale))
    return np.clip(scaled, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def uint16_to_depth(depth_png: Any, scale: float = 1000.0) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Depth PNG scale must be a positive finite number.")
    array = np.asarray(depth_png)
    if array.ndim != 2:
        raise ValueError(f"Depth PNG must be 2-dimensional, got shape {array.shape}.")
    if array.dtype != np.uint16:
        raise ValueError(f"Depth PNG must have uint16 dtype, got {array.dtype}.")
    return (array.astype(np.float32) / np.float32(scale)).astype(np.float32)


def sample_depth_at_points(depth_map: Any, points: Sequence[Sequence[int]]) -> list[float | None]:
    array = validate_depth_map(depth_map)
    height, width = array.shape
    samples: list[float | None] = []
    for point in points:
        if len(point) != 2:
            raise ValueError("Each sample point must contain x and y coordinates.")
        x, y = int(point[0]), int(point[1])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Depth sample point {(x, y)} is outside shape {array.shape}.")
        value = float(array[y, x])
        samples.append(value if np.isfinite(value) and value > 0 else None)
    return samples


def calculate_depth_by_class(
    depth_map: Any,
    class_map: Any,
    id2label: Mapping[int, str],
) -> dict[str, dict[str, float | int]]:
    depth = validate_depth_map(depth_map)
    classes = np.asarray(class_map)
    if classes.ndim != 2 or classes.shape != depth.shape:
        raise ValueError(f"Class map shape {classes.shape} does not match depth map shape {depth.shape}.")
    valid = create_valid_depth_mask(depth)
    result: dict[str, dict[str, float | int]] = {}
    for class_id in np.unique(classes[valid]):
        values = depth[valid & (classes == class_id)].astype(np.float64)
        if not values.size:
            continue
        label = str(id2label.get(int(class_id), f"class_{int(class_id)}"))
        result[label] = {
            "pixel_count": int(values.size),
            "min_depth": float(np.min(values)), "max_depth": float(np.max(values)),
            "mean_depth": float(np.mean(values)), "median_depth": float(np.median(values)),
            "percentile_05": float(np.percentile(values, 5)),
            "percentile_95": float(np.percentile(values, 95)),
        }
    return result
