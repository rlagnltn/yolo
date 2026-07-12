"""Project camera-centric BEV planner paths back onto source video frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.bev import BEVConfig, points_to_bev_indices


def path_cells_to_pixels(
    path_rc: Any,
    points_xyz: Any,
    pixels_uv: Any,
    bev_config: BEVConfig,
) -> list[tuple[int, int] | None]:
    """Map each planner cell to the nearest observed source pixel in that cell."""

    path = np.asarray(path_rc, dtype=np.int32)
    points = np.asarray(points_xyz, dtype=np.float32)
    pixels = np.asarray(pixels_uv)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path_rc must have shape (N, 2).")
    if pixels.ndim != 2 or pixels.shape != (len(points), 2):
        raise ValueError("pixels_uv must have shape (N, 2) matching points_xyz.")
    indices = points_to_bev_indices(points, bev_config)
    rows, cols, valid = indices["row_indices"], indices["col_indices"], indices["valid_point_indices"]
    if not len(valid):
        return [None] * len(path)
    flat = rows.astype(np.int64) * bev_config.width_cells + cols
    distances = np.linalg.norm(points[valid].astype(np.float64), axis=1)
    order = np.lexsort((valid, distances, flat))
    ordered_flat = flat[order]
    winners = order[np.r_[True, ordered_flat[1:] != ordered_flat[:-1]]]
    nearest_pixels = {
        int(flat[index]): tuple(int(value) for value in pixels[valid[index]])
        for index in winners
    }
    result: list[tuple[int, int] | None] = []
    for row, col in path:
        if not (0 <= row < bev_config.height_cells and 0 <= col < bev_config.width_cells):
            result.append(None)
        else:
            result.append(nearest_pixels.get(int(row) * bev_config.width_cells + int(col)))
    return result


def draw_projected_path(
    frame: Any,
    projected_pixels: Iterable[tuple[int, int] | None],
    *,
    color: tuple[int, int, int] = (0, 220, 0),
    thickness: int = 4,
) -> np.ndarray:
    """Draw only contiguous projected segments; do not bridge missing cells."""

    import cv2

    image = np.asarray(frame).copy()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("frame must have shape (H, W, 3).")
    if thickness < 1:
        raise ValueError("thickness must be positive.")
    segments: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for pixel in projected_pixels:
        if pixel is None:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(pixel)
    if current:
        segments.append(current)
    for segment in segments:
        if len(segment) > 1:
            cv2.polylines(image, [np.asarray(segment, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
        else:
            cv2.circle(image, segment[0], max(1, thickness // 2), color, -1, cv2.LINE_AA)
    if segments and segments[0]:
        cv2.circle(image, segments[0][0], thickness + 1, (255, 180, 0), -1, cv2.LINE_AA)
    if segments and segments[-1]:
        cv2.circle(image, segments[-1][-1], thickness + 1, (0, 0, 255), -1, cv2.LINE_AA)
    return image


def render_path_overlay_video(
    input_path: str | Path,
    perception: dict[str, Any],
    output_path: str | Path,
    *,
    repository_root: str | Path,
    codec: str = "mp4v",
    speed: float = 1.0,
) -> dict[str, Any]:
    """Render successful new planner paths over the perception result's frame span."""

    import cv2

    if len(codec) != 4:
        raise ValueError("codec must contain exactly four characters.")
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive.")
    metadata = perception.get("metadata", {})
    start_frame = int(metadata.get("start_frame", 0))
    end_frame = int(metadata.get("end_frame_exclusive", start_frame + len(perception.get("frames", []))))
    if end_frame <= start_frame:
        raise ValueError("Perception result has an invalid frame range.")
    frames = {int(frame["frame_index"]): frame for frame in perception.get("frames", [])}
    root = Path(repository_root).resolve()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Input video must have a positive frame size and FPS.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_fps = fps * float(speed)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), output_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output}")
    successful_paths = missing_path = missing_point_cloud = rendered = 0
    experimental = bool(metadata.get("experimental_intrinsics", False))
    source_index = -1
    try:
        while source_index + 1 < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            source_index += 1
            if source_index < start_frame:
                continue
            record = frames.get(source_index)
            planner = (record or {}).get("planner") or {}
            if planner.get("path_source") == "new" and planner.get("reached_goal"):
                successful_paths += 1
                path_name = planner.get("grid_path_path")
                point_name = ((record or {}).get("geometry") or {}).get("point_cloud_path")
                bev_record = (record or {}).get("bev") or {}
                if not path_name or not point_name or not bev_record:
                    missing_path += 1
                else:
                    try:
                        path = np.load(_resolve_artifact(path_name, root), allow_pickle=False)
                        cloud = np.load(_resolve_artifact(point_name, root), allow_pickle=False)
                        config = BEVConfig(
                            float(bev_record["x_range_m"][0]), float(bev_record["x_range_m"][1]),
                            float(bev_record["z_range_m"][0]), float(bev_record["z_range_m"][1]),
                            float(bev_record["resolution_m"]),
                        )
                        pixels = path_cells_to_pixels(path, cloud["points_xyz"], cloud["pixels_uv"], config)
                        if all(pixel is None for pixel in pixels):
                            missing_point_cloud += 1
                        else:
                            frame = draw_projected_path(frame, pixels)
                            rendered += 1
                    except (KeyError, OSError, ValueError):
                        missing_point_cloud += 1
            if experimental:
                cv2.putText(frame, "EXPERIMENTAL GEOMETRY", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 165, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    return {
        "output_path": str(output), "start_frame": start_frame, "end_frame_exclusive": end_frame,
        "rendered_frame_count": source_index - start_frame + 1,
        "successful_new_path_frame_count": successful_paths,
        "path_drawn_frame_count": rendered,
        "missing_path_frame_count": missing_path,
        "missing_point_cloud_frame_count": missing_point_cloud,
        "source_fps": fps, "output_fps": output_fps, "speed": float(speed), "width": width, "height": height,
    }


def _resolve_artifact(value: str | Path, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def merge_perception_chunks(perceptions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge non-overlapping frame chunks into one continuous render result."""

    items = list(perceptions)
    if not items:
        raise ValueError("At least one perception result is required.")
    merged_frames: dict[int, dict[str, Any]] = {}
    for perception in items:
        for frame in perception.get("frames", []):
            index = int(frame["frame_index"])
            if index in merged_frames:
                raise ValueError(f"Perception chunks overlap at frame {index}.")
            merged_frames[index] = frame
    if not merged_frames:
        raise ValueError("Perception chunks contain no frames.")
    ordered = [merged_frames[index] for index in sorted(merged_frames)]
    start = int(ordered[0]["frame_index"])
    end = int(ordered[-1]["frame_index"]) + 1
    metadata = dict(items[0].get("metadata", {}))
    metadata.update({
        "start_frame": start,
        "end_frame_exclusive": end,
        "processed_frame_count": len(ordered),
        "experimental_intrinsics": any(bool(item.get("metadata", {}).get("experimental_intrinsics", False)) for item in items),
    })
    return {"metadata": metadata, "frames": ordered}
