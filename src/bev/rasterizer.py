"""Rasterize camera-coordinate points into camera-centric X-Z BEV grids."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import BEVConfig


def _points(points_xyz: Any) -> np.ndarray:
    points = np.asarray(points_xyz)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3), got {points.shape}.")
    return points.astype(np.float32, copy=False)


def points_to_bev_indices(points_xyz: Any, config: BEVConfig) -> dict[str, np.ndarray]:
    config.validate()
    points = _points(points_xyz)
    if points.shape[0] == 0:
        return {
            "row_indices": np.empty((0,), dtype=np.int32),
            "col_indices": np.empty((0,), dtype=np.int32),
            "valid_point_indices": np.empty((0,), dtype=np.int32),
        }
    finite = np.isfinite(points).all(axis=1)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    valid = finite & (x >= config.x_min_m) & (x < config.x_max_m) & (z >= config.z_min_m) & (z < config.z_max_m)
    if config.min_y_m is not None:
        valid &= y >= config.min_y_m
    if config.max_y_m is not None:
        valid &= y < config.max_y_m
    point_indices = np.nonzero(valid)[0].astype(np.int32, copy=False)
    if not point_indices.size:
        return {
            "row_indices": np.empty((0,), dtype=np.int32),
            "col_indices": np.empty((0,), dtype=np.int32),
            "valid_point_indices": point_indices,
        }
    col = np.floor((x[point_indices] - config.x_min_m) / config.resolution_m).astype(np.int32)
    row_near = np.floor((z[point_indices] - config.z_min_m) / config.resolution_m).astype(np.int32)
    row = (config.height_cells - 1 - row_near).astype(np.int32)
    return {"row_indices": row, "col_indices": col, "valid_point_indices": point_indices}


def rasterize_semantic_bev(
    points_xyz: Any,
    semantic_labels: Any,
    config: BEVConfig,
    conflict_policy: str = "nearest",
) -> dict[str, np.ndarray]:
    if conflict_policy != "nearest":
        raise ValueError(f"Unsupported BEV conflict_policy: {conflict_policy}.")
    points = _points(points_xyz)
    labels = np.asarray(semantic_labels)
    if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
        raise ValueError("semantic_labels must have shape (N,) matching points_xyz.")
    indices = points_to_bev_indices(points, config)
    rows, cols, valid_idx = indices["row_indices"], indices["col_indices"], indices["valid_point_indices"]
    class_grid = np.full(config.shape, int(config.unknown_class_id), dtype=np.uint8)
    observed_mask = np.zeros(config.shape, dtype=bool)
    point_count_grid = np.zeros(config.shape, dtype=np.uint32)
    if not valid_idx.size:
        return {"class_grid": class_grid, "observed_mask": observed_mask, "point_count_grid": point_count_grid}

    np.add.at(point_count_grid, (rows, cols), 1)
    observed_mask[rows, cols] = True
    distances = np.linalg.norm(points[valid_idx].astype(np.float64), axis=1)
    flat = rows.astype(np.int64) * config.width_cells + cols.astype(np.int64)
    order = np.lexsort((valid_idx, distances, flat))
    first = np.r_[True, flat[order][1:] != flat[order][:-1]]
    winners = order[first]
    class_grid[rows[winners], cols[winners]] = labels[valid_idx[winners]].astype(np.uint8, copy=False)
    return {"class_grid": class_grid, "observed_mask": observed_mask, "point_count_grid": point_count_grid}


def rasterize_observation_bev(points_xyz: Any, config: BEVConfig) -> dict[str, np.ndarray]:
    points = _points(points_xyz)
    indices = points_to_bev_indices(points, config)
    rows, cols, valid_idx = indices["row_indices"], indices["col_indices"], indices["valid_point_indices"]
    observed_mask = np.zeros(config.shape, dtype=bool)
    point_count_grid = np.zeros(config.shape, dtype=np.uint32)
    nearest_distance_grid = np.full(config.shape, np.inf, dtype=np.float32)
    if valid_idx.size:
        np.add.at(point_count_grid, (rows, cols), 1)
        observed_mask[rows, cols] = True
        distances = np.linalg.norm(points[valid_idx].astype(np.float64), axis=1).astype(np.float32)
        np.minimum.at(nearest_distance_grid, (rows, cols), distances)
    return {
        "observed_mask": observed_mask,
        "point_count_grid": point_count_grid,
        "nearest_distance_grid": nearest_distance_grid,
    }


def create_bev_region_masks(
    class_grid: Any,
    observed_mask: Any,
    id2label: Mapping[int, str],
) -> dict[str, np.ndarray]:
    from src.scene_segmentation.class_mapping import classify_label, normalize_id2label

    grid = np.asarray(class_grid)
    observed = np.asarray(observed_mask, dtype=bool)
    if grid.ndim != 2 or observed.shape != grid.shape:
        raise ValueError("class_grid and observed_mask must be 2D arrays with matching shapes.")
    labels = normalize_id2label(id2label)
    drivable = np.zeros(grid.shape, dtype=bool)
    non_drivable = np.zeros(grid.shape, dtype=bool)
    known = np.zeros(grid.shape, dtype=bool)
    for class_id, label in labels.items():
        mask = observed & (grid == int(class_id))
        group = classify_label(label)
        if group == "drivable":
            drivable |= mask
            known |= mask
        elif group in {"pedestrian_surface", "static_obstacle", "dynamic_object", "conditionally_drivable"}:
            non_drivable |= mask
            known |= mask
    unknown = ~known
    return {"drivable_mask": drivable, "non_drivable_mask": non_drivable, "unknown_mask": unknown}


def save_bev_frame_result(
    frame_index: int,
    bev: Mapping[str, np.ndarray],
    config: BEVConfig,
    *,
    id2label: Mapping[int, str] | None,
    class_grid_dir: str | Path,
    drivable_grid_dir: str | Path,
    non_drivable_grid_dir: str | Path,
    visualization_dir: str | Path,
    save_class_grid_npy: bool = True,
    save_class_grid_png: bool = True,
    save_region_masks: bool = True,
    save_visualizations: bool = True,
    conflict_policy: str = "nearest",
    has_semantic_labels: bool = True,
) -> dict[str, Any]:
    from src.utils.io_utils import ensure_dir, save_image
    from .visualization import colorize_bev_class_grid, mark_camera_origin

    stem = f"frame_{frame_index:06d}"
    class_grid = bev.get("class_grid")
    observed_mask = np.asarray(bev["observed_mask"], dtype=bool)
    if class_grid is None:
        class_grid = np.full(config.shape, config.unknown_class_id, dtype=np.uint8)
    class_grid = np.asarray(class_grid)
    if class_grid.shape != config.shape:
        raise ValueError("BEV class_grid shape does not match config.")
    masks = (
        create_bev_region_masks(class_grid, observed_mask, id2label or {})
        if has_semantic_labels else {
            "drivable_mask": np.zeros(config.shape, dtype=bool),
            "non_drivable_mask": np.zeros(config.shape, dtype=bool),
            "unknown_mask": ~observed_mask,
        }
    )
    result: dict[str, Any] = {
        "coordinate_frame": "camera_xz",
        "projection_type": "camera_centric_xz",
        "unit": "meter",
        "resolution_m": float(config.resolution_m),
        "x_range_m": [float(config.x_min_m), float(config.x_max_m)],
        "z_range_m": [float(config.z_min_m), float(config.z_max_m)],
        "shape": [int(config.height_cells), int(config.width_cells)],
        "observed_cell_count": int(observed_mask.sum()),
        "drivable_cell_count": int(masks["drivable_mask"].sum()),
        "non_drivable_cell_count": int(masks["non_drivable_mask"].sum()),
        "unknown_cell_count": int(masks["unknown_mask"].sum()),
        "has_semantic_labels": bool(has_semantic_labels),
        "conflict_policy": conflict_policy,
    }
    if save_class_grid_npy:
        path = ensure_dir(class_grid_dir) / f"{stem}.npy"
        np.save(path, class_grid.astype(np.uint8, copy=False), allow_pickle=False)
        result["class_grid_path"] = str(path)
    if save_class_grid_png:
        result["class_grid_png_path"] = str(save_image(
            class_grid.astype(np.uint8, copy=False), Path(class_grid_dir) / f"{stem}.png"
        ))
    if save_region_masks:
        result["drivable_mask_path"] = str(save_image(
            masks["drivable_mask"].astype(np.uint8) * 255, Path(drivable_grid_dir) / f"{stem}.png"
        ))
        result["non_drivable_mask_path"] = str(save_image(
            masks["non_drivable_mask"].astype(np.uint8) * 255, Path(non_drivable_grid_dir) / f"{stem}.png"
        ))
    if save_visualizations:
        vis = mark_camera_origin(colorize_bev_class_grid(class_grid, observed_mask, config))
        result["visualization_path"] = str(save_image(vis, Path(visualization_dir) / f"{stem}.png"))
    return result
