"""Short-lived temporal occupancy evidence for camera-centric BEV grids."""

from __future__ import annotations

import numpy as np


class TemporalOccupancyFusion:
    """Retain prior observed cells only while the current frame is UNKNOWN.

    Current FREE and OCCUPIED observations always override history. Retained evidence
    expires after a small frame TTL and is never created from an unobserved cell.
    """

    def __init__(self, ttl_frames: int = 8, unknown_value: int = -1) -> None:
        if ttl_frames < 0:
            raise ValueError("Temporal occupancy TTL must be non-negative.")
        self.ttl_frames = int(ttl_frames)
        self.unknown_value = int(unknown_value)
        self._state: np.ndarray | None = None
        self._age: np.ndarray | None = None

    def update(self, occupancy_grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        current = np.asarray(occupancy_grid)
        if current.ndim != 2:
            raise ValueError("Temporal occupancy input must be a 2D grid.")
        observed = current != self.unknown_value
        if self._state is None or self._state.shape != current.shape:
            self._state = current.copy()
            self._age = np.zeros(current.shape, dtype=np.uint16)
        else:
            self._age = np.where(observed, 0, np.minimum(self._age.astype(np.uint32) + 1, 65535)).astype(np.uint16)
            self._state = np.where(observed, current, self._state)
            self._state = np.where(self._age <= self.ttl_frames, self._state, self.unknown_value)
        retained = (~observed) & (self._state != self.unknown_value)
        fused = self._state.copy()
        return fused, {
            "ttl_frames": self.ttl_frames,
            "current_observed_cell_count": int(observed.sum()),
            "retained_observation_cell_count": int(retained.sum()),
            "fused_free_cell_count": int((fused == 0).sum()),
            "fused_occupied_cell_count": int((fused == 100).sum()),
            "fused_unknown_cell_count": int((fused == self.unknown_value).sum()),
        }

    def reset(self) -> None:
        self._state = None
        self._age = None
