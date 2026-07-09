"""Video and image source helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def get_video_info(video_path: str | Path) -> dict[str, float | int]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        return {
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        }
    finally:
        capture.release()


def iter_video_frames(video_path: str | Path) -> Iterator[tuple[int, float, object]]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_sec = frame_index / fps if fps > 0 else 0.0
            yield frame_index, timestamp_sec, frame
            frame_index += 1
    finally:
        capture.release()
