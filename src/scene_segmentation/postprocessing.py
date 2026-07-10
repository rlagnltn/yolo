"""Tensor and NumPy postprocessing for scene segmentation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def resize_logits_to_frame(logits: Any, height: int, width: int) -> Any:
    if height <= 0 or width <= 0:
        raise ValueError("Frame dimensions must be positive.")
    import torch.nn.functional as functional

    return functional.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)


def logits_to_class_map(logits: Any, height: int, width: int) -> Any:
    resized = resize_logits_to_frame(logits, height, width)
    class_map = resized.argmax(dim=1)[0].detach().cpu().numpy()
    return _compact_integer_map(class_map)


def validate_class_map(class_map: Any, expected_shape: tuple[int, int] | None = None) -> Any:
    import numpy as np

    array = np.asarray(class_map)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Class map must be a non-empty 2D array.")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"Class map shape {array.shape} does not match frame shape {expected_shape}.")
    if not np.issubdtype(array.dtype, np.integer) or int(array.min()) < 0:
        raise ValueError("Class map must contain non-negative integer class IDs.")
    return array


def calculate_class_statistics(class_map: Any, id2label: Mapping[int, str]) -> list[dict[str, Any]]:
    import numpy as np

    array = validate_class_map(class_map)
    class_ids, counts = np.unique(array, return_counts=True)
    total = int(array.size)
    return [
        {
            "class_id": int(class_id),
            "class_name": id2label.get(int(class_id), f"unknown_{int(class_id)}"),
            "pixel_count": int(count),
            "pixel_ratio": float(count / total),
        }
        for class_id, count in zip(class_ids, counts)
    ]


def create_binary_region_mask(class_map: Any, class_ids: Iterable[int]) -> Any:
    import numpy as np

    array = validate_class_map(class_map)
    return (np.isin(array, list(class_ids)).astype("uint8") * 255)


def create_drivable_mask(
    class_map: Any,
    id2label: Mapping[int, str],
    drivable_labels: set[str] | None = None,
) -> Any:
    from .class_mapping import DRIVABLE_CLASSES, class_ids_for_labels

    labels = DRIVABLE_CLASSES if drivable_labels is None else drivable_labels
    return create_binary_region_mask(class_map, class_ids_for_labels(id2label, labels))


def create_non_drivable_mask(
    class_map: Any,
    id2label: Mapping[int, str],
    non_drivable_labels: set[str] | None = None,
) -> Any:
    from .class_mapping import (
        DYNAMIC_OBJECT_CLASSES, PEDESTRIAN_SURFACE_CLASSES, STATIC_OBSTACLE_CLASSES,
        class_ids_for_labels,
    )

    labels = non_drivable_labels or (
        PEDESTRIAN_SURFACE_CLASSES | STATIC_OBSTACLE_CLASSES | DYNAMIC_OBJECT_CLASSES
    )
    return create_binary_region_mask(class_map, class_ids_for_labels(id2label, labels))


def calculate_region_statistics(class_map: Any, drivable_mask: Any, non_drivable_mask: Any) -> dict[str, Any]:
    import numpy as np

    array = validate_class_map(class_map)
    drivable = np.asarray(drivable_mask) > 0
    non_drivable = np.asarray(non_drivable_mask) > 0
    if drivable.shape != array.shape or non_drivable.shape != array.shape:
        raise ValueError("Region masks must match the class-map shape.")
    total = int(array.size)
    drivable_count = int(drivable.sum())
    non_drivable_count = int(non_drivable.sum())
    return {
        "drivable_pixel_count": drivable_count,
        "drivable_pixel_ratio": float(drivable_count / total),
        "non_drivable_pixel_count": non_drivable_count,
        "unknown_pixel_count": int(total - (drivable | non_drivable).sum()),
    }


def _compact_integer_map(class_map: Any) -> Any:
    import numpy as np

    array = np.asarray(class_map)
    maximum = int(array.max()) if array.size else 0
    dtype = np.uint8 if maximum <= 255 else np.uint16
    return array.astype(dtype)
