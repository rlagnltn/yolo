"""Back-project metric depth maps to camera-coordinate point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .camera import CameraIntrinsics


def _validate_depth_limits(min_depth_m: float | None, max_depth_m: float | None) -> None:
    if min_depth_m is not None and (not np.isfinite(min_depth_m) or min_depth_m <= 0):
        raise ValueError("min_depth_m must be a positive finite value.")
    if max_depth_m is not None and (not np.isfinite(max_depth_m) or max_depth_m <= 0):
        raise ValueError("max_depth_m must be a positive finite value.")
    if min_depth_m is not None and max_depth_m is not None and min_depth_m > max_depth_m:
        raise ValueError("min_depth_m cannot exceed max_depth_m.")


def backproject_depth(
    depth_map: Any,
    intrinsics: CameraIntrinsics,
    valid_mask: Any | None = None,
    stride: int = 1,
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
) -> dict[str, np.ndarray]:
    """Convert valid depth pixels into XYZ camera-coordinate points."""

    if int(stride) < 1:
        raise ValueError("stride must be at least 1.")
    _validate_depth_limits(min_depth_m, max_depth_m)
    intrinsics.validate()
    depth = np.asarray(depth_map)
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2-dimensional, got shape {depth.shape}.")
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"Depth map shape {depth.shape} does not match camera size "
            f"({intrinsics.height}, {intrinsics.width})."
        )
    depth_float = depth.astype(np.float32, copy=False)
    mask = np.isfinite(depth_float) & (depth_float > 0)
    if valid_mask is not None:
        external = np.asarray(valid_mask)
        if external.shape != depth.shape:
            raise ValueError(f"valid_mask shape {external.shape} does not match depth map shape {depth.shape}.")
        mask &= external.astype(bool, copy=False)
    if min_depth_m is not None:
        mask &= depth_float >= np.float32(min_depth_m)
    if max_depth_m is not None:
        mask &= depth_float <= np.float32(max_depth_m)

    row_mask = np.zeros(depth.shape[0], dtype=bool)
    col_mask = np.zeros(depth.shape[1], dtype=bool)
    row_mask[:: int(stride)] = True
    col_mask[:: int(stride)] = True
    mask &= row_mask[:, None] & col_mask[None, :]

    v, u = np.nonzero(mask)
    if not u.size:
        return {
            "points_xyz": np.empty((0, 3), dtype=np.float32),
            "pixels_uv": np.empty((0, 2), dtype=np.int32),
            "depth_values": np.empty((0,), dtype=np.float32),
        }
    z = depth_float[v, u].astype(np.float32, copy=False)
    x = (u.astype(np.float32) - np.float32(intrinsics.cx)) * z / np.float32(intrinsics.fx)
    y = (v.astype(np.float32) - np.float32(intrinsics.cy)) * z / np.float32(intrinsics.fy)
    return {
        "points_xyz": np.column_stack((x, y, z)).astype(np.float32, copy=False),
        "pixels_uv": np.column_stack((u, v)).astype(np.int32, copy=False),
        "depth_values": z.astype(np.float32, copy=False),
    }


def attach_semantic_labels(pixels_uv: Any, class_map: Any) -> np.ndarray:
    pixels = np.asarray(pixels_uv)
    classes = np.asarray(class_map)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels_uv must have shape (N, 2), got {pixels.shape}.")
    if classes.ndim != 2:
        raise ValueError(f"class_map must be 2-dimensional, got shape {classes.shape}.")
    if not pixels.size:
        return np.empty((0,), dtype=np.int32)
    u = pixels[:, 0].astype(np.int64, copy=False)
    v = pixels[:, 1].astype(np.int64, copy=False)
    if np.any(u < 0) or np.any(v < 0) or np.any(u >= classes.shape[1]) or np.any(v >= classes.shape[0]):
        raise ValueError("pixels_uv contains coordinates outside class_map bounds.")
    return classes[v, u].astype(np.int32, copy=False)


def save_point_cloud_npz(
    output_path: str | Path,
    points_xyz: np.ndarray,
    pixels_uv: np.ndarray,
    depth_values: np.ndarray,
    semantic_labels: np.ndarray | None = None,
) -> Path:
    from src.utils.io_utils import ensure_dir

    path = Path(output_path)
    ensure_dir(path.parent)
    payload: dict[str, np.ndarray] = {
        "points_xyz": np.asarray(points_xyz, dtype=np.float32),
        "pixels_uv": np.asarray(pixels_uv, dtype=np.int32),
        "depth_values": np.asarray(depth_values, dtype=np.float32),
    }
    if semantic_labels is not None:
        payload["semantic_labels"] = np.asarray(semantic_labels, dtype=np.int32)
    np.savez_compressed(path, **payload)
    return path
