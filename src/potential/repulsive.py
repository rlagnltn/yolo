from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def create_repulsive_potential(
    occupied_mask: Any, resolution_m: float, gain: float, influence_radius_m: float
) -> np.ndarray:
    occupied = np.asarray(occupied_mask, dtype=bool)
    if occupied.ndim != 2:
        raise ValueError("occupied_mask must be 2D.")
    if not all(np.isfinite(value) for value in (resolution_m, gain, influence_radius_m)):
        raise ValueError("Repulsive parameters must be finite.")
    if resolution_m <= 0 or gain < 0 or influence_radius_m <= 0:
        raise ValueError("Repulsive parameters are out of range.")
    if not occupied.any():
        return np.zeros(occupied.shape, dtype=np.float32)
    distances = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance_m = distances * np.float32(resolution_m)
    safe_distance = np.maximum(distance_m, np.finfo(np.float32).eps)
    active = distance_m <= influence_radius_m
    potential = np.zeros(occupied.shape, dtype=np.float32)
    potential[active] = np.float32(0.5 * gain) * (
        1.0 / safe_distance[active] - 1.0 / np.float32(influence_radius_m)
    ) ** 2
    potential[occupied] = np.float32(0.5 * gain) * (1.0 / np.finfo(np.float32).eps) ** 2
    return np.nan_to_num(potential, nan=0.0, posinf=np.finfo(np.float32).max).astype(np.float32)
