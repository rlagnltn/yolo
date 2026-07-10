"""Camera intrinsic parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters for one image size."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def validate(self) -> None:
        values = {
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
        }
        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(f"Camera intrinsic {name} must be finite.")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera focal lengths fx and fy must be positive.")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("Camera width and height must be positive.")

    def to_dict(self) -> dict[str, float | int]:
        self.validate()
        return {
            "fx": float(self.fx), "fy": float(self.fy),
            "cx": float(self.cx), "cy": float(self.cy),
            "width": int(self.width), "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraIntrinsics":
        missing = [key for key in ("fx", "fy", "cx", "cy", "width", "height") if data.get(key) is None]
        if missing:
            raise ValueError(f"Camera intrinsics are incomplete; missing: {', '.join(missing)}.")
        intrinsics = cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )
        intrinsics.validate()
        return intrinsics

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        self.validate()
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError("Scaled camera width and height must be positive.")
        scale_x = float(width) / float(self.width)
        scale_y = float(height) / float(self.height)
        return CameraIntrinsics(
            fx=float(self.fx) * scale_x,
            fy=float(self.fy) * scale_y,
            cx=float(self.cx) * scale_x,
            cy=float(self.cy) * scale_y,
            width=int(width),
            height=int(height),
        )
