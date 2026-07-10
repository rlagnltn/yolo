"""Semantic occupancy classification and mapping output persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np

from .config import MappingConfig


def create_semantic_occupancy_grid(
    class_grid: Any,
    observed_mask: Any,
    id2label: Mapping[int, str],
    config: MappingConfig,
) -> dict[str, np.ndarray]:
    from src.scene_segmentation.class_mapping import classify_label, normalize_id2label

    grid = np.asarray(class_grid)
    observed = np.asarray(observed_mask, dtype=bool)
    if grid.ndim != 2 or observed.shape != grid.shape:
        raise ValueError("class_grid and observed_mask must be matching 2D arrays.")
    occupancy = np.full(grid.shape, config.unknown_value, dtype=np.int16)
    for class_id, label in normalize_id2label(id2label).items():
        mask = observed & (grid == class_id)
        group = classify_label(label)
        if group == "drivable":
            occupancy[mask] = config.free_value
        elif group in {"pedestrian_surface", "static_obstacle", "dynamic_object", "conditionally_drivable"}:
            occupancy[mask] = config.occupied_value
    free = occupancy == config.free_value
    occupied = occupancy == config.occupied_value
    unknown = occupancy == config.unknown_value
    return {"occupancy_grid": occupancy, "free_mask": free, "occupied_mask": occupied, "unknown_mask": unknown}


def save_mapping_frame_result(
    frame_index: int,
    occupancy: Mapping[str, np.ndarray],
    cost_grid: np.ndarray,
    inflated_cost_grid: np.ndarray,
    *,
    resolution_m: float,
    config: MappingConfig,
    occupancy_dir: str | Path,
    cost_grid_dir: str | Path,
    inflated_cost_dir: str | Path,
    visualization_dir: str | Path,
    save_occupancy_npy: bool = True,
    save_occupancy_png: bool = True,
    save_cost_npy: bool = True,
    save_cost_png: bool = True,
    save_inflated_cost: bool = True,
    save_visualizations: bool = True,
) -> dict[str, Any]:
    from src.utils.io_utils import ensure_dir, save_image
    from .visualization import colorize_cost_grid, colorize_occupancy_grid

    grid = np.asarray(occupancy["occupancy_grid"])
    costs = np.asarray(cost_grid, dtype=np.float32)
    inflated = np.asarray(inflated_cost_grid, dtype=np.float32)
    if grid.ndim != 2 or costs.shape != grid.shape or inflated.shape != grid.shape:
        raise ValueError("Mapping output grids must have matching 2D shapes.")
    stem = f"frame_{frame_index:06d}"
    result: dict[str, Any] = {
        "coordinate_frame": "camera_xz",
        "grid_type": "semantic_occupancy_cost",
        "resolution_m": float(resolution_m),
        "shape": [int(grid.shape[0]), int(grid.shape[1])],
        "observed_cell_count": int((~occupancy["unknown_mask"]).sum()),
        "free_cell_count": int(occupancy["free_mask"].sum()),
        "occupied_cell_count": int(occupancy["occupied_mask"].sum()),
        "unknown_cell_count": int(occupancy["unknown_mask"].sum()),
        "inflated_cell_count": int((np.isfinite(inflated) & (inflated > costs) & ~occupancy["occupied_mask"]).sum()),
        "inflation_radius_m": float(config.inflation_radius_m),
        "inflation_decay": config.inflation_decay,
        "unknown_policy": config.unknown_policy,
    }
    if save_occupancy_npy:
        path = ensure_dir(occupancy_dir) / f"{stem}.npy"
        np.save(path, grid, allow_pickle=False)
        result["occupancy_grid_path"] = str(path)
    if save_occupancy_png:
        result["occupancy_grid_png_path"] = str(save_image(
            colorize_occupancy_grid(grid, config, grayscale=True), Path(occupancy_dir) / f"{stem}.png"
        ))
    if save_cost_npy:
        path = ensure_dir(cost_grid_dir) / f"{stem}.npy"
        np.save(path, costs, allow_pickle=False)
        result["cost_grid_path"] = str(path)
    if save_cost_png:
        result["cost_grid_png_path"] = str(save_image(
            colorize_cost_grid(costs), Path(cost_grid_dir) / f"{stem}.png"
        ))
    if save_inflated_cost:
        path = ensure_dir(inflated_cost_dir) / f"{stem}.npy"
        np.save(path, inflated, allow_pickle=False)
        result["inflated_cost_grid_path"] = str(path)
        result["inflated_cost_grid_png_path"] = str(save_image(
            colorize_cost_grid(inflated), Path(inflated_cost_dir) / f"{stem}.png"
        ))
    if save_visualizations:
        result["visualization_path"] = str(save_image(
            colorize_cost_grid(inflated), Path(visualization_dir) / f"{stem}.png"
        ))
    return result
