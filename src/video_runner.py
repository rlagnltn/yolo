"""Frame-by-frame MP4/JSONL runner for image-space planning visualization."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
import cv2
import numpy as np

from src.planner.coordinates import metric_path_to_grid
from src.trajectory import geometry, validate_metric_path_collision
from src.video_plan import TemporalPlanningState, grid_to_pixel, json_safe, render_overlay


@dataclass(frozen=True)
class VideoPlanOptions:
    codec: str = "mp4v"
    frame_stride: int = 1
    max_frames: int | None = None
    potential_smoothing: bool = True
    potential_alpha: float = .4
    trajectory_stabilization: bool = True
    trajectory_alpha: float = .5
    reuse_previous: bool = True
    max_reuse_frames: int = 3
    heatmap_alpha: float = .35
    show_detections: bool = True
    show_potential: bool = True
    show_occupancy: bool = False
    show_raw_path: bool = True
    show_trajectory: bool = True
    fail_fast: bool = False

    def validate(self) -> "VideoPlanOptions":
        if len(self.codec) != 4: raise ValueError("Codec must contain exactly four characters.")
        if self.frame_stride < 1: raise ValueError("Frame stride must be at least 1.")
        if self.max_frames is not None and self.max_frames <= 0: raise ValueError("max_frames must be positive.")
        if not (0 <= self.potential_alpha <= 1 and 0 <= self.trajectory_alpha <= 1 and 0 <= self.heatmap_alpha <= 1): raise ValueError("Alpha values must be within [0, 1].")
        if self.max_reuse_frames < 0: raise ValueError("max_reuse_frames must be non-negative.")
        return self


def run_video_plan(pipeline: Any, input_path: str | Path, output_path: str | Path, *,
                   frame_options: dict[str, Any] | None = None, metadata_path: str | Path | None = None,
                   options: VideoPlanOptions | None = None) -> dict[str, Any]:
    options = (options or VideoPlanOptions()).validate()
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path: raise ValueError("Input and output paths must differ.")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened(): raise FileNotFoundError(f"Could not open input video: {input_path}")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)); source_fps = source_fps if np.isfinite(source_fps) and source_fps > 0 else 30.0
    output_fps = source_fps / options.frame_stride
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*options.codec), output_fps, (width, height))
    if not writer.isOpened():
        capture.release(); raise RuntimeError(f"Could not create output video with codec {options.codec}: {output_path}")
    metadata_file = None
    if metadata_path is not None:
        metadata_path = Path(metadata_path); metadata_path.parent.mkdir(parents=True, exist_ok=True); metadata_file = metadata_path.open("w", encoding="utf-8")
    state, processed, failed, reused, source_index = TemporalPlanningState(), 0, 0, 0, -1
    started = time.perf_counter(); frame_options = frame_options or {}
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            source_index += 1
            if source_index % options.frame_stride: continue
            if options.max_frames is not None and processed >= options.max_frames: break
            tick = time.perf_counter(); error = None; result: dict[str, Any] = {}
            try:
                result = pipeline.process_frame(
                    frame, source_index, source_index / source_fps, **frame_options,
                    temporal_state=state, temporal_options={"potential_enabled": options.potential_smoothing, "potential_alpha": options.potential_alpha},
                )
            except Exception as exc:
                error = exc
                if options.fail_fast: raise
            planner_memory = getattr(pipeline, "_last_planner_memory", None) if result.get("planner") else None
            if planner_memory is None and error is None and result.get("potential"):
                context = getattr(pipeline, "_last_video_context", None)
                if context is not None:
                    planner_memory = {"path_rc": None, "potential_grid": context["potential_grid"], "occupancy_grid": context["occupancy_grid"], "cost_grid": context["cost_grid"], "bev_config": context["bev_config"], "goal_cell": context["goal_cell"], "start_cell": None, "source_algorithm": None}
            trajectory_memory = getattr(pipeline, "_last_trajectory_memory", None) if result.get("trajectory") else None
            current = None if trajectory_memory is None else trajectory_memory["positions_xz"]
            trajectory_source = "none"; final = None; final_geometry = None; trajectory_rc = None
            if planner_memory is not None:
                validator = lambda path: validate_metric_path_collision(path, planner_memory["occupancy_grid"], planner_memory["bev_config"], .05)
                if options.trajectory_stabilization:
                    final, trajectory_source = state.stabilize_trajectory(current, validator, options.trajectory_alpha, options.reuse_previous, options.max_reuse_frames)
                else:
                    final, trajectory_source = current, "current" if current is not None else "none"
                if trajectory_source == "reused_previous": reused += 1
                if final is not None:
                    final_geometry = geometry(final); trajectory_rc = metric_path_to_grid(final, planner_memory["bev_config"])
            elapsed = time.perf_counter() - tick
            status = "error" if error else (result.get("planner") or {}).get("status", "unavailable")
            if error is not None or status not in {"success", "goal_reached"}: failed += 1
            overlay = render_overlay(
                frame, detections=result.get("detections"), potential=None if planner_memory is None else planner_memory["potential_grid"],
                occupancy=None if planner_memory is None else planner_memory["occupancy_grid"],
                raw_path=None if planner_memory is None else planner_memory["path_rc"], trajectory_rc=trajectory_rc,
                start=None if planner_memory is None else planner_memory.get("start_cell"), goal=None if planner_memory is None else planner_memory["goal_cell"],
                grid_shape=None if planner_memory is None else planner_memory["occupancy_grid"].shape,
                heatmap_alpha=options.heatmap_alpha, status_text=f"frame={source_index} {status} {trajectory_source} {1/max(elapsed,1e-9):.1f} FPS",
                show_detections=options.show_detections, show_potential=options.show_potential, show_occupancy=options.show_occupancy,
                show_raw_path=options.show_raw_path, show_trajectory=options.show_trajectory,
            )
            if overlay.shape != (height, width, 3) or overlay.dtype != np.uint8: raise ValueError("Overlay frame must preserve BGR uint8 input shape.")
            writer.write(overlay)
            row = {
                "frame_index": processed, "source_frame_index": source_index, "timestamp_seconds": source_index/source_fps,
                "source_fps": source_fps, "output_fps": output_fps, "frame_width": width, "frame_height": height,
                "processing_time_ms": elapsed*1000, "instantaneous_fps": 1/max(elapsed,1e-9),
                "detection_count": len(result.get("detections", [])), "planner_name": (result.get("planner") or {}).get("selected_algorithm"),
                "start_image_xy": None if planner_memory is None or planner_memory.get("start_cell") is None else grid_to_pixel(*planner_memory["start_cell"], planner_memory["occupancy_grid"].shape, width, height),
                "goal_image_xy": None if planner_memory is None else grid_to_pixel(*planner_memory["goal_cell"], planner_memory["occupancy_grid"].shape, width, height),
                "start_grid_xy": None if planner_memory is None or planner_memory.get("start_cell") is None else [planner_memory["start_cell"][1], planner_memory["start_cell"][0]],
                "goal_grid_xy": None if planner_memory is None else [planner_memory["goal_cell"][1], planner_memory["goal_cell"][0]],
                "planning_status": status, "fallback_used": (result.get("planner") or {}).get("fallback_used", False),
                "potential_smoothing_enabled": options.potential_smoothing, "potential_alpha": options.potential_alpha,
                "trajectory_stabilization_enabled": options.trajectory_stabilization, "trajectory_alpha": options.trajectory_alpha,
                "trajectory_source": trajectory_source, "previous_trajectory_reused": trajectory_source == "reused_previous",
                "previous_trajectory_reuse_age": state.trajectory_reuse_age,
                "raw_path": None if planner_memory is None else planner_memory["path_rc"],
                "smoothed_path": None if trajectory_memory is None else trajectory_memory["shortcut_path_rc"],
                "final_trajectory_positions": None if final_geometry is None else final_geometry["positions_xz"],
                "final_trajectory_heading": None if final_geometry is None else final_geometry["heading_rad"],
                "final_trajectory_curvature": None if final_geometry is None else final_geometry["curvature_1pm"],
                "error_type": None if error is None else type(error).__name__, "error_message": None if error is None else str(error),
            }
            if metadata_file is not None:
                metadata_file.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n"); metadata_file.flush()
            processed += 1
    finally:
        capture.release(); writer.release()
        if metadata_file is not None: metadata_file.close()
    total = time.perf_counter() - started
    return {"input_path": str(input_path), "output_path": str(output_path), "metadata_path": None if metadata_path is None else str(metadata_path),
            "processed_frame_count": processed, "failed_frame_count": failed, "reused_trajectory_frame_count": reused,
            "source_fps": source_fps, "output_fps": output_fps, "width": width, "height": height,
            "average_processing_fps": processed/max(total,1e-9), "elapsed_seconds": total}
