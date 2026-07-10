"""Mask helpers for segmentation output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import ensure_dir


def mask_area(mask: Any) -> int:
    """Return the number of positive pixels in a binary-like mask."""

    if mask is None:
        return 0

    try:
        import numpy as np

        return int(np.asarray(mask).astype(bool).sum())
    except Exception:
        return int(sum(1 for row in mask for value in row if bool(value)))


def build_mask_path(
    mask_dir: str | Path,
    frame_index: int,
    object_index: int,
    object_label: str = "obj",
) -> Path:
    """Create a stable mask filename for one segmented object."""

    return Path(mask_dir) / f"frame_{frame_index:06d}_{object_label}_{object_index:03d}.png"


def save_binary_mask(mask: Any, output_path: str | Path) -> Path:
    """Save a binary mask as an 8-bit PNG image."""

    import cv2
    import numpy as np

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    mask_image = (np.asarray(mask).astype(bool).astype("uint8")) * 255
    ok = cv2.imwrite(str(output_path), mask_image)
    if not ok:
        raise RuntimeError(f"Failed to write mask image: {output_path}")
    return output_path
