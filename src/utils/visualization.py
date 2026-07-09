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
