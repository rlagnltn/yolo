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
        mapping_output: dict[str, Any] | None = None,
        potential_output: dict[str, Any] | None = None,
        planner_output: dict[str, Any] | None = None,
        trajectory_output: dict[str, Any] | None = None,
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
        bev_result, bev_grid = self._build_bev_result(
            geometry_cloud, semantic_labels, frame_index, bev_output, errors
        )
        mapping_result, mapping_grid = self._build_mapping_result(
            bev_grid, frame_index, mapping_output, errors
        )
        potential_result, planner_context = self._build_potential_result(mapping_grid, frame_index, potential_output, errors)
        planner_result = self._build_planner_result(planner_context, frame_index, planner_output, errors)
        trajectory_result = self._build_trajectory_result(trajectory_output, planner_result, errors)
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
            "mapping": mapping_result,
            "potential": potential_result,
            "planner": planner_result,
            "trajectory": trajectory_result,
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
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        options = bev_output or {}
        if not options.get("enabled", False):
            return None, None
        if geometry_cloud is None:
            errors.append("bev: geometry must be enabled before BEV generation.")
            return None, None
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
            metadata = save_bev_frame_result(
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
            memory_grid = {
                **bev,
                "config": config,
                "id2label": options.get("id2label"),
                "has_semantic_labels": has_labels,
            }
            return metadata, memory_grid
        except Exception as exc:
            if not self.continue_on_error:
                raise
            errors.append(f"bev: {exc}")
            return None, None

    def _build_mapping_result(
        self,
        bev_grid: dict[str, Any] | None,
        frame_index: int,
        mapping_output: dict[str, Any] | None,
        errors: list[str],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        options = mapping_output or {}
        if not options.get("enabled", False):
            return None, None
        if bev_grid is None:
            errors.append("mapping: semantic BEV must be enabled before mapping.")
            return None, None
        if not bev_grid.get("has_semantic_labels") or "class_grid" not in bev_grid:
            errors.append("mapping: semantic labels are required for semantic occupancy generation.")
            return None, None
        try:
            from src.mapping import (
                MappingConfig, create_semantic_cost_grid, create_semantic_occupancy_grid,
                inflate_cost_grid, save_mapping_frame_result,
            )

            config = MappingConfig.from_dict(options.get("config", {}))
            occupancy = create_semantic_occupancy_grid(
                bev_grid["class_grid"], bev_grid["observed_mask"], bev_grid.get("id2label") or {}, config
            )
            cost_grid = create_semantic_cost_grid(
                bev_grid["class_grid"], bev_grid["observed_mask"], bev_grid.get("id2label") or {}, config
            )
            inflated = (
                inflate_cost_grid(
                    cost_grid, occupancy["occupied_mask"], bev_grid["config"].resolution_m,
                    config.inflation_radius_m, config.inflation_decay,
                ) if config.inflation_enabled else cost_grid.copy()
            )
            metadata = save_mapping_frame_result(
                frame_index, occupancy, cost_grid, inflated,
                resolution_m=bev_grid["config"].resolution_m,
                config=config,
                occupancy_dir=options.get("occupancy_dir", "outputs/perception/mapping/occupancy"),
                cost_grid_dir=options.get("cost_grid_dir", "outputs/perception/mapping/cost_grids"),
                inflated_cost_dir=options.get("inflated_cost_dir", "outputs/perception/mapping/inflated_cost_grids"),
                visualization_dir=options.get("visualization_dir", "outputs/perception/mapping/visualizations"),
                save_occupancy_npy=options.get("save_occupancy_npy", True),
                save_occupancy_png=options.get("save_occupancy_png", True),
                save_cost_npy=options.get("save_cost_npy", True),
                save_cost_png=options.get("save_cost_png", True),
                save_inflated_cost=options.get("save_inflated_cost", True),
                save_visualizations=options.get("save_visualizations", True),
            )
            return metadata, {
                "occupancy_grid": occupancy["occupancy_grid"], "inflated_cost_grid": inflated,
                "resolution_m": bev_grid["config"].resolution_m, "bev_config": bev_grid["config"],
            }
        except Exception as exc:
            if not self.continue_on_error:
                raise
            errors.append(f"mapping: {exc}")
            return None, None

    def _build_potential_result(
        self, mapping_grid: dict[str, Any] | None, frame_index: int,
        potential_output: dict[str, Any] | None, errors: list[str],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        options = potential_output or {}
        if not options.get("enabled", False):
            return None, None
        if mapping_grid is None:
            errors.append("potential: mapping must be enabled before potential generation.")
            return None, None
        try:
            from src.potential import (
                PotentialConfig, calculate_potential_gradient, combine_potential_fields,
                create_attractive_potential, create_repulsive_potential, metric_goal_to_grid,
                save_potential_frame_result, validate_goal_cell,
            )

            config = PotentialConfig.from_dict(options.get("config", {}))
            goal = options.get("goal", {})
            row, col, x_m, z_m = self._resolve_potential_goal(goal, mapping_grid)
            occupancy = mapping_grid["occupancy_grid"]
            if occupancy[row, col] != 0:
                raise ValueError("Goal cell must be an observed FREE cell.")
            attractive = create_attractive_potential(
                occupancy.shape, (row, col), mapping_grid["resolution_m"], config.attractive_gain,
                config.attractive_mode, config.attractive_saturation_distance_m,
            )
            repulsive = create_repulsive_potential(
                occupancy == 100, mapping_grid["resolution_m"], config.repulsive_gain,
                config.repulsive_influence_radius_m,
            )
            combined_fields = combine_potential_fields(
                attractive, repulsive, mapping_grid["inflated_cost_grid"], occupancy, config
            )
            combined = combined_fields["normalized_potential"]
            if combined is None:
                combined = combined_fields["raw_potential"]
            fields = {"attractive": attractive, "repulsive": repulsive, "combined": combined,
                      "blocked_mask": combined_fields["blocked_mask"]}
            gradient = calculate_potential_gradient(combined, mapping_grid["resolution_m"], combined_fields["blocked_mask"])
            result = save_potential_frame_result(
                frame_index, fields, gradient, goal_cell=(row, col), goal_metric=(x_m, z_m),
                resolution_m=mapping_grid["resolution_m"], config=config,
                attractive_dir=options.get("attractive_dir", "outputs/perception/potential/attractive"),
                repulsive_dir=options.get("repulsive_dir", "outputs/perception/potential/repulsive"),
                combined_dir=options.get("combined_dir", "outputs/perception/potential/combined"),
                gradient_dir=options.get("gradient_dir", "outputs/perception/potential/gradients"),
                visualization_dir=options.get("visualization_dir", "outputs/perception/potential/visualizations"),
                save_npy=options.get("save_npy", True), save_png=options.get("save_png", True),
                save_gradient=options.get("save_gradient", True), save_visualizations=options.get("save_visualizations", True),
            )
            return result, {"potential_grid": combined, "occupancy_grid": occupancy, "cost_grid": mapping_grid["inflated_cost_grid"], "goal_cell": (row, col), "bev_config": mapping_grid["bev_config"]}
        except Exception as exc:
            if not self.continue_on_error:
                raise
            errors.append(f"potential: {exc}")
            return None, None

    def _build_planner_result(self, context: dict[str, Any] | None, frame_index: int, planner_output: dict[str, Any] | None, errors: list[str]) -> dict[str, Any] | None:
        options = planner_output or {}
        if not options.get("enabled", False):
            return None
        if context is None:
            errors.append("planner: potential must be enabled before planning.")
            return None
        try:
            import numpy as np
            from src.planner import PlannerConfig, calculate_path_length_m, draw_path_on_potential, grid_path_to_metric, plan_astar, plan_gradient_descent, plan_hybrid, resolve_start_cell, validate_grid_path, validate_start_cell
            from src.utils.io_utils import ensure_dir, save_image, save_json
            config = PlannerConfig.from_dict(options.get("config", {}))
            start = options.get("start", {})
            grid_start = (start["row"], start["col"]) if start.get("row") is not None or start.get("col") is not None else None
            metric_start = (start["x_m"], start["z_m"]) if start.get("x_m") is not None or start.get("z_m") is not None else None
            start_cell = resolve_start_cell(grid_start, metric_start, context["bev_config"])
            validate_start_cell(start_cell, context["occupancy_grid"])
            if config.algorithm == "gradient_descent":
                result = plan_gradient_descent(context["potential_grid"], context["occupancy_grid"], start_cell, context["goal_cell"], config)
                selected_algorithm, fallback_used, fallback_reason, astar_result = "gradient_descent", False, None, None
            elif config.algorithm == "astar":
                result = plan_astar(context["occupancy_grid"], context["cost_grid"], start_cell, context["goal_cell"], config)
                selected_algorithm, fallback_used, fallback_reason, astar_result = "astar", False, None, result
            else:
                hybrid = plan_hybrid(context["potential_grid"], context["occupancy_grid"], context["cost_grid"], start_cell, context["goal_cell"], config)
                result, selected_algorithm, fallback_used, fallback_reason, astar_result = hybrid, hybrid["selected_algorithm"], hybrid["fallback_used"], hybrid["fallback_reason"], hybrid["astar_result"]
            path = result["path_rc"]; metric = grid_path_to_metric(path, context["bev_config"])
            if result["reached_goal"]: validate_grid_path(path, context["occupancy_grid"], start_cell, context["goal_cell"], config.connectivity, config.goal_tolerance_cells, config.prevent_corner_cutting)
            stem = f"frame_{frame_index:06d}"; path_dir = ensure_dir(options.get("path_dir", "outputs/perception/planner/paths")); vis_dir = options.get("visualization_dir", "outputs/perception/planner/visualizations")
            gradient_result = result.get("gradient_result")
            metadata = {"coordinate_frame":"camera_xz", "algorithm":config.algorithm, "selected_algorithm":selected_algorithm, "status":result["status"], "reached_goal":result["reached_goal"], "fallback_used":fallback_used, "fallback_reason":fallback_reason, "gradient_status":None if gradient_result is None else gradient_result["status"], "astar_status":None if astar_result is None else astar_result["status"], "expanded_node_count":None if astar_result is None else astar_result["expanded_node_count"], "path_cost":None if astar_result is None else astar_result["path_cost"], "start":{"row":start_cell[0],"col":start_cell[1]}, "goal":{"row":context["goal_cell"][0],"col":context["goal_cell"][1]}, "step_count":len(path) - 1, "path_point_count":len(path), "path_length_m":calculate_path_length_m(metric), "termination_reason":result["termination_reason"], "local_minimum_detected":False if gradient_result is None else gradient_result["diagnostics"]["local_minimum_detected"]}
            if options.get("save_path_npy", True):
                grid_file=path_dir/f"{stem}_grid.npy"; metric_file=path_dir/f"{stem}_metric.npy"; np.save(grid_file,path,allow_pickle=False); np.save(metric_file,metric,allow_pickle=False); metadata.update(grid_path_path=str(grid_file),metric_path_path=str(metric_file))
            if options.get("save_visualization", True): metadata["visualization_path"] = str(save_image(draw_path_on_potential(context["potential_grid"],path,start_cell,context["goal_cell"]), Path(vis_dir)/f"{stem}.png"))
            if options.get("save_path_json", True): metadata["metadata_path"] = str(save_json(metadata,path_dir/f"{stem}.json"))
            self._last_planner_memory = {"path_rc": path, "occupancy_grid": context["occupancy_grid"], "cost_grid": context["cost_grid"], "bev_config": context["bev_config"], "source_algorithm": selected_algorithm, "frame_index": frame_index}
            return metadata
        except Exception as exc:
            if not self.continue_on_error: raise
            errors.append(f"planner: {exc}")
            return None

    def _build_trajectory_result(self, trajectory_output: dict[str, Any] | None, planner_result: dict[str, Any] | None, errors: list[str]) -> dict[str, Any] | None:
        options = trajectory_output or {}
        if not options.get("enabled", False): return None
        if planner_result is None or not planner_result.get("reached_goal"):
            errors.append("trajectory: planner must reach the goal before trajectory generation."); return None
        try:
            import numpy as np
            from pathlib import Path
            from src.trajectory import TrajectoryConfig, generate_trajectory
            from src.utils.io_utils import ensure_dir, save_json
            memory = self._last_planner_memory
            result = generate_trajectory(memory["path_rc"], memory["occupancy_grid"], memory["bev_config"], TrajectoryConfig.from_dict(options.get("config", {})), memory["cost_grid"])
            if not result.get("collision_free"): return {"status": result["status"], "trajectory_type": "geometric_reference", "collision_free": False}
            data = ensure_dir(options.get("trajectory_dir", "outputs/perception/trajectory/data")); stem=f"frame_{memory['frame_index']:06d}"; traj=result["trajectory"]
            positions=data/f"{stem}_positions.npy"; bundle=data/f"{stem}_trajectory.npz"; np.save(positions,traj["positions_xz"],allow_pickle=False); np.savez_compressed(bundle,**traj,source_grid_path=memory["path_rc"],shortcut_grid_path=result["shortcut_path_rc"])
            meta={"coordinate_frame":"camera_xz","trajectory_type":"geometric_reference","status":result["status"],"source_algorithm":memory["source_algorithm"],"trajectory_point_count":len(traj["positions_xz"]),"path_length_m":float(traj["arc_length_m"][-1]),"collision_free":True,"smoothing_fallback_used":result["smoothing_fallback_used"],"maximum_curvature_1pm":result["diagnostics"]["maximum_observed_curvature"],"positions_path":str(positions),"trajectory_path":str(bundle)}
            if options.get("save_json", True): meta["metadata_path"]=str(save_json(meta,data/f"{stem}.json"))
            return meta
        except Exception as exc:
            if not self.continue_on_error: raise
            errors.append(f"trajectory: {exc}"); return None

    @staticmethod
    def _resolve_potential_goal(goal: dict[str, Any], mapping_grid: dict[str, Any]) -> tuple[int, int, float, float]:
        from src.potential import metric_goal_to_grid, validate_goal_cell

        row, col, x_m, z_m = goal.get("row"), goal.get("col"), goal.get("x_m"), goal.get("z_m")
        grid_given, metric_given = row is not None or col is not None, x_m is not None or z_m is not None
        if grid_given and metric_given:
            raise ValueError("Provide either a grid goal or a metric goal, not both.")
        if not grid_given and not metric_given:
            raise ValueError("Potential generation requires a goal.")
        config = mapping_grid["bev_config"]
        if metric_given:
            if x_m is None or z_m is None:
                raise ValueError("Metric goal requires both x_m and z_m.")
            x_m, z_m = float(x_m), float(z_m)
            row, col = metric_goal_to_grid(x_m, z_m, config)
        else:
            if row is None or col is None:
                raise ValueError("Grid goal requires both row and col.")
            row, col = validate_goal_cell(row, col, config.shape)
            x_m = config.x_min_m + (col + 0.5) * config.resolution_m
            z_m = config.z_min_m + (config.height_cells - row - 0.5) * config.resolution_m
        return int(row), int(col), float(x_m), float(z_m)

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
        mapping_output: dict[str, Any] | None = None,
        potential_output: dict[str, Any] | None = None,
        planner_output: dict[str, Any] | None = None,
        trajectory_output: dict[str, Any] | None = None,
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
                mapping_output=mapping_output,
                potential_output=potential_output,
                planner_output=planner_output,
                trajectory_output=trajectory_output,
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
