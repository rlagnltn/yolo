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
        geometry_output: dict[str, Any] | None = None,
        bev_output: dict[str, Any] | None = None,
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
        depth_map = None
        if self.depth_estimator is not None:
            try:
                from src.depth.output import build_depth_frame_result
                from src.depth.postprocessing import calculate_depth_by_class

                prediction = self.depth_estimator.predict(frame)
                depth_map = prediction["depth_map"]
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
        geometry_result, geometry_cloud, semantic_labels = self._build_geometry_result(
            depth_map, scene_class_map, frame_index, geometry_output, errors
        )
        bev_result = self._build_bev_result(
            geometry_cloud, semantic_labels, frame_index, bev_output, errors
        )
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
            "geometry": geometry_result,
            "bev": bev_result,
            "errors": errors,
        }

    def _build_geometry_result(
        self,
        depth_map: Any | None,
        scene_class_map: Any | None,
        frame_index: int,
        geometry_output: dict[str, Any] | None,
        errors: list[str],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Any | None]:
        options = geometry_output or {}
        if not options.get("enabled", False):
            return None, None, None
        if depth_map is None:
            errors.append("geometry: depth must be enabled before back-projection.")
            return None, None, None
        try:
            from src.geometry import CameraIntrinsics, attach_semantic_labels, backproject_depth, save_point_cloud_npz

            intrinsics = CameraIntrinsics.from_dict(options.get("intrinsics", {}))
            if tuple(depth_map.shape) != (intrinsics.height, intrinsics.width):
                intrinsics = intrinsics.scaled_to(int(depth_map.shape[1]), int(depth_map.shape[0]))
            stride = int(options.get("stride", 1))
            min_depth_m = options.get("min_depth_m")
            max_depth_m = options.get("max_depth_m")
            cloud = backproject_depth(
                depth_map, intrinsics,
                stride=stride,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            )
            labels = None
            if scene_class_map is not None:
                labels = attach_semantic_labels(cloud["pixels_uv"], scene_class_map)
            point_cloud_path = save_point_cloud_npz(
                Path(options.get("point_cloud_dir", "outputs/perception/geometry/point_clouds"))
                / f"frame_{frame_index:06d}.npz",
                cloud["points_xyz"],
                cloud["pixels_uv"],
                cloud["depth_values"],
                labels,
            )
            return {
                "point_cloud_path": str(point_cloud_path),
                "coordinate_frame": "camera",
                "unit": "meter",
                "point_count": int(cloud["points_xyz"].shape[0]),
                "stride": stride,
                "depth_range_m": [min_depth_m, max_depth_m],
                "intrinsics": intrinsics.to_dict(),
                "has_semantic_labels": labels is not None,
            }, cloud, labels
        except Exception as exc:
            if not self.continue_on_error:
                raise
            errors.append(f"geometry: {exc}")
            return None, None, None

    def _build_bev_result(
        self,
        geometry_cloud: dict[str, Any] | None,
        semantic_labels: Any | None,
        frame_index: int,
        bev_output: dict[str, Any] | None,
        errors: list[str],
    ) -> dict[str, Any] | None:
        options = bev_output or {}
        if not options.get("enabled", False):
            return None
        if geometry_cloud is None:
            errors.append("bev: geometry must be enabled before BEV generation.")
            return None
        try:
            from src.bev import BEVConfig, rasterize_observation_bev, rasterize_semantic_bev, save_bev_frame_result

            config = BEVConfig.from_dict(options.get("config", {}))
            conflict_policy = options.get("conflict_policy", "nearest")
            has_labels = semantic_labels is not None
            if has_labels:
                bev = rasterize_semantic_bev(
                    geometry_cloud["points_xyz"], semantic_labels, config, conflict_policy
                )
            else:
                bev = rasterize_observation_bev(geometry_cloud["points_xyz"], config)
            return save_bev_frame_result(
                frame_index, bev, config,
                id2label=options.get("id2label"),
                class_grid_dir=options.get("class_grid_dir", "outputs/perception/bev/class_grids"),
                drivable_grid_dir=options.get("drivable_grid_dir", "outputs/perception/bev/drivable"),
                non_drivable_grid_dir=options.get("non_drivable_grid_dir", "outputs/perception/bev/non_drivable"),
                visualization_dir=options.get("visualization_dir", "outputs/perception/bev/visualizations"),
                save_class_grid_npy=options.get("save_class_grid_npy", True),
                save_class_grid_png=options.get("save_class_grid_png", True),
                save_region_masks=options.get("save_region_masks", True),
                save_visualizations=options.get("save_visualizations", True),
                conflict_policy=conflict_policy,
                has_semantic_labels=has_labels,
            )
        except Exception as exc:
            if not self.continue_on_error:
                raise
            errors.append(f"bev: {exc}")
            return None

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
        geometry_output: dict[str, Any] | None = None,
        bev_output: dict[str, Any] | None = None,
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
                geometry_output=geometry_output,
                bev_output=bev_output,
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
