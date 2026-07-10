"""Cityscapes label groups used by downstream driving-region logic."""

from __future__ import annotations

from typing import Mapping

DRIVABLE_CLASSES = {"road"}
CONDITIONALLY_DRIVABLE_CLASSES = {"parking"}
PEDESTRIAN_SURFACE_CLASSES = {"sidewalk"}
STATIC_OBSTACLE_CLASSES = {
    "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain",
}
DYNAMIC_OBJECT_CLASSES = {
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
}
BACKGROUND_CLASSES = {"sky"}

# Cityscapes train-ID colors represented in OpenCV BGR order.
CITYSCAPES_BGR_PALETTE = {
    0: (128, 64, 128), 1: (232, 35, 244), 2: (70, 70, 70),
    3: (156, 102, 102), 4: (153, 153, 190), 5: (153, 153, 153),
    6: (30, 170, 250), 7: (0, 220, 220), 8: (35, 142, 107),
    9: (152, 251, 152), 10: (180, 130, 70), 11: (60, 20, 220),
    12: (0, 0, 255), 13: (142, 0, 0), 14: (70, 0, 0),
    15: (100, 60, 0), 16: (100, 80, 0), 17: (230, 0, 0),
    18: (32, 11, 119),
}


def normalize_id2label(id2label: Mapping[int | str, str]) -> dict[int, str]:
    """Normalize model-config labels while keeping the model as source of truth."""

    return {int(class_id): str(label).strip().lower() for class_id, label in id2label.items()}


def classify_label(label: str) -> str:
    """Return the potential-field-oriented group for a semantic label."""

    normalized = label.strip().lower()
    groups = (
        (DRIVABLE_CLASSES, "drivable"),
        (CONDITIONALLY_DRIVABLE_CLASSES, "conditionally_drivable"),
        (PEDESTRIAN_SURFACE_CLASSES, "pedestrian_surface"),
        (STATIC_OBSTACLE_CLASSES, "static_obstacle"),
        (DYNAMIC_OBJECT_CLASSES, "dynamic_object"),
        (BACKGROUND_CLASSES, "background"),
    )
    for labels, group in groups:
        if normalized in labels:
            return group
    return "unknown"


def class_ids_for_labels(id2label: Mapping[int, str], labels: set[str]) -> set[int]:
    normalized_labels = {label.lower() for label in labels}
    return {class_id for class_id, label in id2label.items() if label.lower() in normalized_labels}
