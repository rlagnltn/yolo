"""Configuration for camera-centric X-Z BEV grids."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class BEVConfig:
    x_min_m: float
    x_max_m: float
    z_min_m: float
    z_max_m: float
    resolution_m: float
    min_y_m: float | None = None
    max_y_m: float | None = None
    unknown_class_id: int = 255
    max_cells: int = 4_000_000

    def validate(self) -> None:
        for name in ("x_min_m", "x_max_m", "z_min_m", "z_max_m", "resolution_m"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"BEV config {name} must be finite.")
        for name in ("min_y_m", "max_y_m"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"BEV config {name} must be finite.")
        if self.x_min_m >= self.x_max_m:
            raise ValueError("BEV x_min_m must be smaller than x_max_m.")
        if self.z_min_m >= self.z_max_m:
            raise ValueError("BEV z_min_m must be smaller than z_max_m.")
        if self.resolution_m <= 0:
            raise ValueError("BEV resolution_m must be positive.")
        if self.min_y_m is not None and self.max_y_m is not None and self.min_y_m >= self.max_y_m:
            raise ValueError("BEV min_y_m must be smaller than max_y_m.")
        if self.width_cells * self.height_cells > self.max_cells:
            raise ValueError("BEV grid is too large; reduce range or increase resolution.")
        if not (0 <= int(self.unknown_class_id) <= 255):
            raise ValueError("unknown_class_id must fit in uint8.")

    @property
    def width_cells(self) -> int:
        return int(ceil((self.x_max_m - self.x_min_m) / self.resolution_m))

    @property
    def height_cells(self) -> int:
        return int(ceil((self.z_max_m - self.z_min_m) / self.resolution_m))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height_cells, self.width_cells)

    def to_dict(self) -> dict[str, float | int | None]:
        self.validate()
        return {
            "x_min_m": float(self.x_min_m), "x_max_m": float(self.x_max_m),
            "z_min_m": float(self.z_min_m), "z_max_m": float(self.z_max_m),
            "resolution_m": float(self.resolution_m),
            "min_y_m": None if self.min_y_m is None else float(self.min_y_m),
            "max_y_m": None if self.max_y_m is None else float(self.max_y_m),
            "unknown_class_id": int(self.unknown_class_id),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BEVConfig":
        cfg = cls(
            x_min_m=float(data["x_min_m"]),
            x_max_m=float(data["x_max_m"]),
            z_min_m=float(data["z_min_m"]),
            z_max_m=float(data["z_max_m"]),
            resolution_m=float(data["resolution_m"]),
            min_y_m=None if data.get("min_y_m") is None else float(data["min_y_m"]),
            max_y_m=None if data.get("max_y_m") is None else float(data["max_y_m"]),
            unknown_class_id=int(data.get("unknown_class_id", 255)),
        )
        cfg.validate()
        return cfg
