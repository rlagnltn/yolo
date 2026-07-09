"""Visualization helpers for detection results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import ensure_dir


def draw_detections(frame: Any, detections: list[dict[str, Any]]) -> Any:
    import cv2

    annotated = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox"]]
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
    cv2.imwrite(str(output_path), draw_detections(frame, detections))
    return output_path
