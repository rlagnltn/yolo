from __future__ import annotations

from typing import Any
import numpy as np


def calculate_potential_gradient(
    potential_grid: Any, resolution_m: float, blocked_mask: Any | None = None
) -> dict[str, np.ndarray]:
    potential = np.asarray(potential_grid, dtype=np.float32)
    if potential.ndim != 2 or not np.isfinite(potential).all():
        raise ValueError("potential_grid must be a finite 2D array.")
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution_m must be positive and finite.")
    blocked = np.zeros(potential.shape, dtype=bool) if blocked_mask is None else np.asarray(blocked_mask, dtype=bool)
    if blocked.shape != potential.shape:
        raise ValueError("blocked_mask must match potential_grid shape.")
    row_gradient, gradient_x = np.gradient(potential, np.float32(resolution_m), np.float32(resolution_m))
    gradient_z = -row_gradient  # Array rows increase downward; BEV Z increases toward smaller rows.
    for array in (gradient_x, gradient_z):
        array[blocked] = 0.0
    magnitude = np.hypot(gradient_x, gradient_z).astype(np.float32)
    return {
        "gradient_x": gradient_x.astype(np.float32), "gradient_z": gradient_z.astype(np.float32),
        "descent_x": (-gradient_x).astype(np.float32), "descent_z": (-gradient_z).astype(np.float32),
        "magnitude": magnitude,
    }
