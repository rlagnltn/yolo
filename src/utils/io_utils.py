"""Input and output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read YAML config files. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def save_image(image: Any, output_path: str | Path) -> Path:
    """Save an OpenCV-compatible image, creating its parent directory."""

    import cv2

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    extension = output_path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {output_path}")
    try:
        output_path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise RuntimeError(f"Failed to write image: {output_path}") from exc
    return output_path
