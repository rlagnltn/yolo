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
        scene_segmenter: Any | None = None,
        depth_estimator: Any | None = None,
        iou_threshold: float = 0.5,
        require_same_class: bool = True,
        continue_on_error: bool = True,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.scene_segmenter = scene_segmenter
        self.depth_estimator = depth_estimator
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
        scene_output: dict[str, Any] | None = None,
        depth_output: dict[str, Any] | None = None,
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
        scene_result = None
        scene_class_map = None
        if self.scene_segmenter is not None:
            try:
                from src.scene_segmentation.output import build_scene_frame_result

                class_map = self.scene_segmenter.predict(frame)
                scene_class_map = class_map
                options = scene_output or {}
                scene_result = build_scene_frame_result(
                    frame, class_map, self.scene_segmenter.id2label,
                    frame_index, timestamp_sec,
                    class_map_dir=options.get("class_map_dir", "outputs/perception/scene/class_maps"),
                    color_map_dir=options.get("color_map_dir", "outputs/perception/scene/color_maps"),
                    visualization_dir=options.get("visualization_dir", "outputs/perception/scene/visualizations"),
                    region_dir=options.get("region_dir", "outputs/perception/scene/regions"),
                    save_class_maps=options.get("save_class_maps", True),
                    save_color_maps=options.get("save_color_maps", True),
                    save_visualizations=False,
                    save_regions=options.get("save_regions", True),
                    alpha=options.get("alpha", 0.45),
                )
            except Exception as exc:
                if not self.continue_on_error:
                    raise
                errors.append(f"scene_segmentation: {exc}")
        depth_result = None
        if self.depth_estimator is not None:
            try:
                from src.depth.output import build_depth_frame_result
                from src.depth.postprocessing import calculate_depth_by_class

                prediction = self.depth_estimator.predict(frame)
                options = depth_output or {}
                depth_result = build_depth_frame_result(
                    frame, prediction, frame_index, timestamp_sec,
                    raw_depth_dir=options.get("raw_depth_dir", "outputs/perception/depth/raw"),
                    depth_png_dir=options.get("depth_png_dir", "outputs/perception/depth/depth_maps"),
                    color_map_dir=options.get("color_map_dir", "outputs/perception/depth/color_maps"),
                    visualization_dir=options.get("visualization_dir", "outputs/perception/depth/visualizations"),
                    save_raw_depth=options.get("save_raw_depth", True),
                    save_depth_png=options.get("save_depth_png", True),
                    save_color_maps=options.get("save_color_maps", True),
                    save_visualizations=options.get("save_visualizations", True),
                    png_scale=options.get("png_scale", 1000.0),
                    alpha=options.get("alpha", 0.45),
                    percentile_min=options.get("percentile_min", 2.0),
                    percentile_max=options.get("percentile_max", 98.0),
                )
                if scene_class_map is not None:
                    depth_result["depth_by_scene_class"] = calculate_depth_by_class(
                        prediction["depth_map"], scene_class_map, self.scene_segmenter.id2label
                    )
            except Exception as exc:
                if not self.continue_on_error:
                    raise
                errors.append(f"depth: {exc}")
        return {
            "frame_index": frame_index,
            "timestamp_sec": float(timestamp_sec),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "detections": detections,
            "segments": segments,
            "fused_objects": fused_objects,
            "scene_segmentation": scene_result,
            "depth": depth_result,
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
        scene_output: dict[str, Any] | None = None,
        depth_output: dict[str, Any] | None = None,
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
                scene_output=scene_output,
                depth_output=depth_output,
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
                "scene_segmentation_model": (
                    self.scene_segmenter.model_name if self.scene_segmenter is not None else None
                ),
                "depth_model": (
                    self.depth_estimator.model_name if self.depth_estimator is not None else None
                ),
                "frame_count": int(video_info["frame_count"]),
                "processed_frame_count": len(frames),
                "fps": float(video_info["fps"]),
                "width": int(video_info["width"]),
                "height": int(video_info["height"]),
            },
            "frames": frames,
        }
