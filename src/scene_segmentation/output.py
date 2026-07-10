"""Build and persist per-frame scene-segmentation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .postprocessing import (
    calculate_class_statistics, calculate_region_statistics,
    create_drivable_mask, create_non_drivable_mask, validate_class_map,
)


def build_scene_frame_result(
    frame: Any,
    class_map: Any,
    id2label: Mapping[int, str],
    frame_index: int,
    timestamp_sec: float,
    *,
    class_map_dir: str | Path,
    color_map_dir: str | Path,
    visualization_dir: str | Path,
    region_dir: str | Path,
    save_class_maps: bool = True,
    save_color_maps: bool = True,
    save_visualizations: bool = True,
    save_regions: bool = True,
    alpha: float = 0.45,
    drivable_labels: set[str] | None = None,
    non_drivable_labels: set[str] | None = None,
) -> dict[str, Any]:
    from src.utils.io_utils import save_image
    from src.utils.visualization import colorize_class_map, draw_scene_segmentation_overlay

    height, width = frame.shape[:2]
    class_map = validate_class_map(class_map, (height, width))
    drivable = create_drivable_mask(class_map, id2label, drivable_labels)
    non_drivable = create_non_drivable_mask(class_map, id2label, non_drivable_labels)
    stem = f"frame_{frame_index:06d}"
    result: dict[str, Any] = {
        "frame_index": frame_index,
        "timestamp_sec": float(timestamp_sec),
        "width": int(width),
        "height": int(height),
        "class_statistics": calculate_class_statistics(class_map, id2label),
        "regions": calculate_region_statistics(class_map, drivable, non_drivable),
    }
    if save_class_maps:
        result["class_map_path"] = str(save_image(class_map, Path(class_map_dir) / f"{stem}.png"))
    if save_color_maps:
        result["color_map_path"] = str(save_image(colorize_class_map(class_map), Path(color_map_dir) / f"{stem}.png"))
    if save_visualizations:
        result["overlay_path"] = str(save_image(
            draw_scene_segmentation_overlay(frame, class_map, alpha),
            Path(visualization_dir) / f"{stem}.png",
        ))
    if save_regions:
        result["drivable_mask_path"] = str(save_image(drivable, Path(region_dir) / f"{stem}_drivable.png"))
        result["non_drivable_mask_path"] = str(save_image(
            non_drivable, Path(region_dir) / f"{stem}_non_drivable.png"
        ))
    return result
