"""Visualization helpers for detection and segmentation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import ensure_dir


def draw_detections(frame: Any, detections: list[dict[str, Any]]) -> Any:
    annotated = frame.copy()
    if not detections:
        return annotated

    import cv2

    for detection in detections:
        bbox = detection.get("bbox_xyxy", detection.get("bbox"))
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def save_annotated_frame(
    frame: Any,
    detections: list[dict[str, Any]],
    output_dir: str | Path,
    frame_index: int,
) -> Path:
    import cv2

    output_dir = ensure_dir(output_dir)
    output_path = output_dir / f"frame_{frame_index:06d}.jpg"
    ok = cv2.imwrite(str(output_path), draw_detections(frame, detections))
    if not ok:
        raise RuntimeError(f"Failed to write visualization: {output_path}")
    return output_path


def draw_segmentation_overlay(frame: Any, segments: list[dict[str, Any]], alpha: float = 0.4) -> Any:
    """Draw segmentation masks and labels over a frame."""

    annotated = frame.copy()
    if not segments:
        return annotated

    import cv2
    import numpy as np

    for index, segment in enumerate(segments):
        mask = segment.get("mask")
        mask_path = segment.get("mask_path")
        if mask is None and mask_path:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        mask_array = np.squeeze(np.asarray(mask)).astype(bool)
        if mask_array.shape[:2] != annotated.shape[:2]:
            mask_array = cv2.resize(
                mask_array.astype("uint8"),
                (annotated.shape[1], annotated.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        color = _segment_color(index)
        overlay = annotated.copy()
        overlay[mask_array] = color
        annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)

        bbox = segment.get("bbox_xyxy", segment.get("bbox"))
        if bbox:
            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            label = f"{segment['class_name']} {segment['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    return annotated


def save_segmentation_overlay(
    frame: Any,
    segments: list[dict[str, Any]],
    output_dir: str | Path,
    frame_index: int,
    alpha: float = 0.4,
) -> Path:
    import cv2

    output_dir = ensure_dir(output_dir)
    output_path = output_dir / f"frame_{frame_index:06d}.png"
    ok = cv2.imwrite(str(output_path), draw_segmentation_overlay(frame, segments, alpha))
    if not ok:
        raise RuntimeError(f"Failed to write segmentation overlay: {output_path}")
    return output_path


def _segment_color(index: int) -> tuple[int, int, int]:
    palette = [
        (42, 157, 143),
        (231, 111, 81),
        (244, 162, 97),
        (38, 70, 83),
        (233, 196, 106),
        (67, 97, 238),
    ]
    return palette[index % len(palette)]


FUSION_COLORS = {
    "matched": (40, 200, 40),
    "detection_only": (0, 180, 255),
    "segmentation_only": (220, 120, 40),
}


def draw_perception_overlay(
    frame: Any,
    detections: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    fused_objects: list[dict[str, Any]],
    mask_alpha: float = 0.4,
    scene_class_map: Any | None = None,
    scene_alpha: float = 0.45,
) -> Any:
    """Draw masks first, then class-aware fused bounding boxes and labels."""

    annotated = (
        draw_scene_segmentation_overlay(frame, scene_class_map, scene_alpha)
        if scene_class_map is not None else frame.copy()
    )
    if not detections and not segments and not fused_objects:
        return annotated

    import cv2
    import numpy as np

    for index, segment in enumerate(segments):
        mask = segment.get("mask")
        if mask is None and segment.get("mask_path"):
            mask = cv2.imread(str(segment["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask_array = np.squeeze(np.asarray(mask)).astype(bool)
        if mask_array.shape[:2] != annotated.shape[:2]:
            mask_array = cv2.resize(
                mask_array.astype("uint8"),
                (annotated.shape[1], annotated.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        overlay = annotated.copy()
        overlay[mask_array] = _segment_color(index)
        annotated = cv2.addWeighted(overlay, mask_alpha, annotated, 1 - mask_alpha, 0)

    for fused_object in fused_objects:
        bbox = fused_object.get("bbox_xyxy")
        if not bbox:
            continue
        status = fused_object.get("fusion_status", "detection_only")
        color = FUSION_COLORS.get(status, (255, 255, 255))
        confidence = fused_object.get("detection_confidence")
        if confidence is None:
            confidence = fused_object.get("segmentation_confidence")
        label = f"{fused_object['class_name']} {float(confidence or 0):.2f} [{status}]"
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, label, (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
    return annotated


def save_perception_overlay(
    frame: Any,
    frame_result: dict[str, Any],
    output_dir: str | Path,
    frame_index: int,
    mask_alpha: float = 0.4,
) -> Path:
    import cv2

    output_dir = ensure_dir(output_dir)
    output_path = output_dir / f"frame_{frame_index:06d}.png"
    scene_result = frame_result.get("scene_segmentation") or {}
    scene_class_map = scene_result.get("_class_map")
    if scene_class_map is None and scene_result.get("class_map_path"):
        # Class-ID maps are single-channel artifacts.  Reading them as grayscale
        # also normalizes OpenCV builds that expose a PNG gray image as HxWx1.
        scene_class_map = cv2.imread(str(scene_result["class_map_path"]), cv2.IMREAD_GRAYSCALE)
        if scene_class_map is not None and scene_class_map.ndim == 3 and scene_class_map.shape[2] == 1:
            scene_class_map = scene_class_map[:, :, 0]
    annotated = draw_perception_overlay(
        frame,
        frame_result.get("detections", []),
        frame_result.get("segments", []),
        frame_result.get("fused_objects", []),
        mask_alpha,
        scene_class_map,
    )
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Failed to write perception overlay: {output_path}")
    return output_path


def colorize_class_map(
    class_map: Any,
    palette: dict[int, tuple[int, int, int]] | None = None,
    unknown_color: tuple[int, int, int] = (64, 64, 64),
) -> Any:
    """Convert a class-ID map to a stable BGR visualization."""

    import numpy as np

    from src.scene_segmentation.class_mapping import CITYSCAPES_BGR_PALETTE
    from src.scene_segmentation.postprocessing import validate_class_map

    array = validate_class_map(class_map)
    colors = palette or CITYSCAPES_BGR_PALETTE
    color_map = np.full((*array.shape, 3), unknown_color, dtype="uint8")
    for class_id, color in colors.items():
        color_map[array == class_id] = color
    return color_map


def draw_scene_segmentation_overlay(frame: Any, class_map: Any, alpha: float = 0.45) -> Any:
    """Blend a semantic color map over a copied BGR frame."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Overlay alpha must be between 0 and 1.")
    color_map = colorize_class_map(class_map)
    if tuple(frame.shape[:2]) != tuple(color_map.shape[:2]):
        raise ValueError(
            f"Class map shape {color_map.shape[:2]} does not match frame shape {frame.shape[:2]}."
        )
    import cv2

    return cv2.addWeighted(color_map, alpha, frame.copy(), 1 - alpha, 0)


def draw_drivable_region_overlay(frame: Any, drivable_mask: Any, alpha: float = 0.35) -> Any:
    """Highlight drivable pixels in green without modifying the input frame."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Overlay alpha must be between 0 and 1.")
    import cv2
    import numpy as np

    mask = np.asarray(drivable_mask) > 0
    if tuple(frame.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"Mask shape {mask.shape} does not match frame shape {frame.shape[:2]}.")
    overlay = frame.copy()
    overlay[mask] = (0, 220, 0)
    return cv2.addWeighted(overlay, alpha, frame.copy(), 1 - alpha, 0)
