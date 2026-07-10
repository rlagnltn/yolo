"""Single-pass unified perception pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .fusion import match_detections_and_segments


class PerceptionPipeline:
    """Coordinate one detector and one instance segmenter over shared frames."""

    def __init__(
        self,
        detector: Any,
        segmenter: Any,
        *,
        iou_threshold: float = 0.5,
        require_same_class: bool = True,
        continue_on_error: bool = True,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.iou_threshold = iou_threshold
        self.require_same_class = require_same_class
        self.continue_on_error = continue_on_error

    def process_frame(
        self,
        frame: Any,
        frame_index: int,
        timestamp_sec: float = 0.0,
        *,
        mask_dir: str | Path | None = None,
        save_masks: bool = True,
    ) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        errors: list[str] = []
        try:
            detections = self.detector.detect_frame(frame)
        except Exception as exc:
            if not self.continue_on_error:
                raise
            errors.append(f"detection: {exc}")
        try:
            try:
                segments = self.segmenter.segment_frame(
                    frame, frame_index, mask_dir, save_masks,
                    mask_object_label="seg",
                )
            except TypeError as exc:
                if "mask_object_label" not in str(exc):
                    raise
                segments = self.segmenter.segment_frame(frame, frame_index, mask_dir, save_masks)
        except Exception as exc:
            if not self.continue_on_error:
                raise
            errors.append(f"segmentation: {exc}")

        for index, detection in enumerate(detections):
            detection["object_id"] = f"frame_{frame_index:06d}_det_{index:03d}"
        for index, segment in enumerate(segments):
            segment["segment_id"] = f"frame_{frame_index:06d}_seg_{index:03d}"

        fused_objects = match_detections_and_segments(
            detections, segments, self.iou_threshold, self.require_same_class
        )
        return {
            "frame_index": frame_index,
            "timestamp_sec": float(timestamp_sec),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "detections": detections,
            "segments": segments,
            "fused_objects": fused_objects,
            "errors": errors,
        }

    def process_video(
        self,
        input_path: str | Path,
        *,
        mask_dir: str | Path | None = None,
        visualization_dir: str | Path | None = None,
        save_masks: bool = True,
        save_visualizations: bool = True,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        from src.utils.video_utils import get_video_info, iter_video_frames
        from src.utils.visualization import save_perception_overlay

        input_path = Path(input_path)
        video_info = get_video_info(input_path)
        frames = []
        for frame_index, timestamp_sec, frame in iter_video_frames(input_path):
            if max_frames is not None and frame_index >= max_frames:
                break
            frame_result = self.process_frame(
                frame, frame_index, timestamp_sec,
                mask_dir=mask_dir, save_masks=save_masks,
            )
            frames.append(frame_result)
            if save_visualizations and visualization_dir is not None:
                try:
                    save_perception_overlay(frame, frame_result, visualization_dir, frame_index)
                except Exception as exc:
                    if not self.continue_on_error:
                        raise
                    frame_result["errors"].append(f"visualization: {exc}")

        return {
            "metadata": {
                "input": str(input_path),
                "detection_model": self.detector.model_name,
                "segmentation_model": self.segmenter.model_name,
                "frame_count": int(video_info["frame_count"]),
                "processed_frame_count": len(frames),
                "fps": float(video_info["fps"]),
                "width": int(video_info["width"]),
                "height": int(video_info["height"]),
            },
            "frames": frames,
        }
