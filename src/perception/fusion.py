"""Class-aware, one-to-one bbox fusion for detection and segmentation."""

from __future__ import annotations

from typing import Any, Sequence


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate intersection-over-union for two xyxy boxes."""

    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _same_class(detection: dict[str, Any], segment: dict[str, Any]) -> bool:
    if "class_id" in detection and "class_id" in segment:
        return detection["class_id"] == segment["class_id"]
    return detection.get("class_name") == segment.get("class_name")


def match_detections_and_segments(
    detections: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    require_same_class: bool = True,
) -> list[dict[str, Any]]:
    """Return matched and unmatched records without reusing a segment."""

    candidates: list[tuple[float, int, int]] = []
    for detection_index, detection in enumerate(detections):
        for segment_index, segment in enumerate(segments):
            if require_same_class and not _same_class(detection, segment):
                continue
            iou = calculate_iou(detection["bbox_xyxy"], segment["bbox_xyxy"])
            if iou >= iou_threshold:
                candidates.append((iou, detection_index, segment_index))

    matched_detections: set[int] = set()
    matched_segments: set[int] = set()
    matches: dict[int, int] = {}
    for _iou, detection_index, segment_index in sorted(candidates, reverse=True):
        if detection_index in matched_detections or segment_index in matched_segments:
            continue
        matches[detection_index] = segment_index
        matched_detections.add(detection_index)
        matched_segments.add(segment_index)

    fused: list[dict[str, Any]] = []
    for detection_index, detection in enumerate(detections):
        object_id = f"frame_{_frame_index(detection):06d}_obj_{len(fused):03d}"
        if detection_index in matches:
            segment = segments[matches[detection_index]]
            fused.append({
                "object_id": object_id,
                "class_id": detection["class_id"],
                "class_name": detection["class_name"],
                "detection_confidence": detection["confidence"],
                "segmentation_confidence": segment["confidence"],
                "bbox_xyxy": detection["bbox_xyxy"],
                "mask_area": segment.get("mask_area", 0),
                **({"mask_path": segment["mask_path"]} if segment.get("mask_path") else {}),
                "fusion_status": "matched",
            })
        else:
            fused.append({
                "object_id": object_id,
                "class_id": detection["class_id"],
                "class_name": detection["class_name"],
                "detection_confidence": detection["confidence"],
                "segmentation_confidence": None,
                "bbox_xyxy": detection["bbox_xyxy"],
                "mask_area": 0,
                "fusion_status": "detection_only",
            })

    for segment_index, segment in enumerate(segments):
        if segment_index in matched_segments:
            continue
        object_id = f"frame_{_frame_index(segment):06d}_obj_{len(fused):03d}"
        fused.append({
            "object_id": object_id,
            "class_id": segment["class_id"],
            "class_name": segment["class_name"],
            "detection_confidence": None,
            "segmentation_confidence": segment["confidence"],
            "bbox_xyxy": segment["bbox_xyxy"],
            "mask_area": segment.get("mask_area", 0),
            **({"mask_path": segment["mask_path"]} if segment.get("mask_path") else {}),
            "fusion_status": "segmentation_only",
        })
    return fused


def _frame_index(record: dict[str, Any]) -> int:
    record_id = str(record.get("object_id") or record.get("segment_id") or "")
    try:
        return int(record_id.split("_")[1])
    except (IndexError, ValueError):
        return int(record.get("frame_index", 0))
