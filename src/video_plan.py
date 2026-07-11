"""Temporal state, coordinate conversion, metadata, and overlays for video planning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math
import cv2
import numpy as np


def _validate_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")


def normalized_to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    _validate_size(width, height)
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("Normalized coordinates must be finite values within [0, 1].")
    return round(x * (width - 1)), round(y * (height - 1))


def pixel_to_normalized(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    _validate_size(width, height)
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
        raise ValueError("Pixel coordinates are outside the image.")
    return float(x / max(width - 1, 1)), float(y / max(height - 1, 1))


def pixel_to_grid(x: float, y: float, shape: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    pixel_to_normalized(x, y, width, height)
    return round(y * (shape[0] - 1) / max(height - 1, 1)), round(x * (shape[1] - 1) / max(width - 1, 1))


def grid_to_pixel(row: int, col: int, shape: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        raise ValueError("Grid coordinate is outside the grid.")
    return round(col * (width - 1) / max(shape[1] - 1, 1)), round(row * (height - 1) / max(shape[0] - 1, 1))


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class TemporalPlanningState:
    previous_smoothed_potential: np.ndarray | None = None
    previous_trajectory: np.ndarray | None = None
    previous_goal: tuple[int, int] | None = None
    previous_frame_shape: tuple[int, int] | None = None
    previous_grid_shape: tuple[int, int] | None = None
    trajectory_reuse_age: int = 0
    processed_frame_count: int = 0

    def reset(self) -> None:
        self.previous_smoothed_potential = None
        self.previous_trajectory = None
        self.previous_goal = None
        self.previous_frame_shape = None
        self.previous_grid_shape = None
        self.trajectory_reuse_age = 0

    def prepare(self, frame_shape: tuple[int, int], goal: tuple[int, int] | None, grid_shape: tuple[int, int] | None = None) -> None:
        if self.previous_frame_shape != frame_shape or (grid_shape is not None and self.previous_grid_shape != grid_shape) or (self.previous_goal is not None and self.previous_goal != goal):
            self.reset()
        self.previous_frame_shape, self.previous_grid_shape, self.previous_goal = frame_shape, grid_shape, goal

    def smooth_potential(self, current: Any, occupied: Any, alpha: float = .4) -> np.ndarray:
        if not 0 <= alpha <= 1:
            raise ValueError("Potential alpha must be within [0, 1].")
        field, hard = np.asarray(current, np.float32), np.asarray(occupied, bool)
        if field.ndim != 2 or hard.shape != field.shape or not np.isfinite(field).all():
            raise ValueError("Potential and occupied mask must be finite matching 2D arrays.")
        previous = self.previous_smoothed_potential
        result = field.copy() if previous is None or previous.shape != field.shape or not np.isfinite(previous).all() else alpha * field + (1 - alpha) * previous
        result[hard] = field[hard]
        self.previous_smoothed_potential = result.astype(np.float32)
        return self.previous_smoothed_potential

    def stabilize_trajectory(self, current: np.ndarray | None, validator: Callable[[np.ndarray], bool], alpha: float = .5, reuse: bool = True, max_reuse_frames: int = 3) -> tuple[np.ndarray | None, str]:
        if not 0 <= alpha <= 1 or max_reuse_frames < 0:
            raise ValueError("Trajectory alpha/reuse limit is invalid.")
        previous = self.previous_trajectory
        if current is None:
            if reuse and previous is not None and self.trajectory_reuse_age < max_reuse_frames and validator(previous):
                self.trajectory_reuse_age += 1
                return previous.copy(), "reused_previous"
            return None, "none"
        current = np.asarray(current, np.float32)
        source = "current"
        if previous is not None and len(current) and len(previous) and validator(previous):
            positions = np.linspace(0, len(previous) - 1, len(current))
            aligned = np.column_stack([np.interp(positions, np.arange(len(previous)), previous[:, axis]) for axis in range(2)])
            blended = alpha * current + (1 - alpha) * aligned
            if validator(blended):
                current, source = blended.astype(np.float32), "stabilized"
        self.previous_trajectory, self.trajectory_reuse_age = current.copy(), 0
        return current, source


def render_overlay(frame: Any, *, detections: list[dict[str, Any]] | None = None, potential: Any = None,
                   occupancy: Any = None, raw_path: Any = None, trajectory_rc: Any = None,
                   start: tuple[int, int] | None = None, goal: tuple[int, int] | None = None,
                   grid_shape: tuple[int, int] | None = None, heatmap_alpha: float = .35,
                   status_text: str = "", show_detections: bool = True, show_potential: bool = True,
                   show_occupancy: bool = False, show_raw_path: bool = True, show_trajectory: bool = True) -> np.ndarray:
    output = np.asarray(frame).copy()
    height, width = output.shape[:2]
    if show_potential and potential is not None:
        values = np.asarray(potential, np.float32); finite = np.isfinite(values)
        scaled = np.zeros(values.shape, np.uint8)
        if finite.any():
            low, high = np.percentile(values[finite], [2, 98])
            if high > low: scaled[finite] = np.clip((values[finite] - low) * 255 / (high - low), 0, 255)
        heat = cv2.resize(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), (width, height), interpolation=cv2.INTER_LINEAR)
        output = cv2.addWeighted(output, 1 - heatmap_alpha, heat, heatmap_alpha, 0)
    if show_occupancy and occupancy is not None:
        occupied = cv2.resize((np.asarray(occupancy) != 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        output[occupied] = (.5 * output[occupied] + .5 * np.array([0, 0, 255])).astype(np.uint8)
    if show_detections:
        for item in detections or []:
            x1, y1, x2, y2 = map(int, item.get("bbox_xyxy", (0, 0, 0, 0))); cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(output, f"{item.get('class_name', item.get('class_id', 'obj'))} {item.get('confidence', 0):.2f}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
    def draw(path: Any, color: tuple[int, int, int], thickness: int) -> None:
        if path is None or grid_shape is None: return
        cells = np.asarray(path)
        if cells.ndim == 2 and len(cells) > 1:
            points = np.asarray([grid_to_pixel(int(row), int(col), grid_shape, width, height) for row, col in cells], np.int32)
            cv2.polylines(output, [points], False, color, thickness)
    if show_raw_path: draw(raw_path, (255, 150, 0), 1)
    if show_trajectory: draw(trajectory_rc, (0, 255, 0), 2)
    if grid_shape:
        for cell, color in ((start, (255, 0, 255)), (goal, (0, 0, 255))):
            if cell is not None: cv2.circle(output, grid_to_pixel(*cell, grid_shape, width, height), 4, color, -1)
    if status_text: cv2.putText(output, status_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return output
