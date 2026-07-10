"""Persist per-frame metric depth artifacts and build JSON records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .postprocessing import calculate_depth_statistics, depth_to_uint16, validate_depth_map
from .visualization import colorize_depth_map, draw_depth_overlay


def build_depth_frame_result(
    frame: Any,
    prediction: dict[str, Any],
    frame_index: int,
    timestamp_sec: float,
    *,
    raw_depth_dir: str | Path,
    depth_png_dir: str | Path,
    color_map_dir: str | Path,
    visualization_dir: str | Path,
    save_raw_depth: bool = True,
    save_depth_png: bool = True,
    save_color_maps: bool = True,
    save_visualizations: bool = True,
    png_scale: float = 1000.0,
    alpha: float = 0.45,
    percentile_min: float = 2.0,
    percentile_max: float = 98.0,
) -> dict[str, Any]:
    from src.utils.io_utils import ensure_dir, save_image

    height, width = frame.shape[:2]
    depth = validate_depth_map(prediction["depth_map"], (height, width))
    stem = f"frame_{frame_index:06d}"
    result: dict[str, Any] = {
        "frame_index": int(frame_index), "timestamp_sec": float(timestamp_sec),
        "width": int(width), "height": int(height),
        "depth_type": str(prediction.get("depth_type", "metric")),
        "unit": str(prediction.get("unit", "meter")),
        "model_name": str(prediction.get("model_name", "")),
        "requested_model_name": str(prediction.get("requested_model_name", prediction.get("model_name", ""))),
        "png_scale": float(png_scale),
        "statistics": calculate_depth_statistics(depth),
    }
    if save_raw_depth:
        path = ensure_dir(raw_depth_dir) / f"{stem}.npy"
        np.save(path, depth.astype(np.float32, copy=False), allow_pickle=False)
        result["raw_depth_path"] = str(path)
    if save_depth_png:
        result["depth_png_path"] = str(save_image(
            depth_to_uint16(depth, png_scale), Path(depth_png_dir) / f"{stem}.png"
        ))
    if save_color_maps:
        result["color_map_path"] = str(save_image(
            colorize_depth_map(depth, percentile_min, percentile_max),
            Path(color_map_dir) / f"{stem}.png",
        ))
    if save_visualizations:
        result["overlay_path"] = str(save_image(
            draw_depth_overlay(frame, depth, alpha, percentile_min, percentile_max),
            Path(visualization_dir) / f"{stem}.png",
        ))
    return result
