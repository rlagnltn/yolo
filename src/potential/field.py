from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np

from .config import PotentialConfig


def combine_potential_fields(
    attractive: Any, repulsive: Any, traversability_cost: Any, occupancy_grid: Any, config: PotentialConfig
) -> dict[str, np.ndarray | None]:
    attractive, repulsive = np.asarray(attractive, np.float32), np.asarray(repulsive, np.float32)
    costs, occupancy = np.asarray(traversability_cost, np.float32), np.asarray(occupancy_grid)
    if attractive.ndim != 2 or any(array.shape != attractive.shape for array in (repulsive, costs, occupancy)):
        raise ValueError("Potential inputs must be matching 2D arrays.")
    if not np.isfinite(attractive).all() or not np.isfinite(repulsive).all():
        raise ValueError("Attractive and repulsive potentials must be finite.")
    occupied = occupancy == 100
    unknown = occupancy == -1
    blocked = occupied | (unknown if config.unknown_policy == "blocked" else np.zeros(occupancy.shape, bool))
    safe_costs = np.nan_to_num(costs, nan=config.occupied_potential if config.unknown_policy == "high_cost" else 0.0)
    raw = attractive + repulsive + np.float32(config.cost_weight) * safe_costs
    raw[occupied] = np.float32(config.occupied_potential)
    if config.unknown_policy == "blocked":
        raw[unknown] = np.float32(config.occupied_potential)
    elif config.unknown_policy == "high_cost":
        raw[unknown] = np.maximum(raw[unknown], np.float32(config.occupied_potential))
    normalized: np.ndarray | None = None
    if config.normalize_output:
        normalized = raw.copy()
        free = ~blocked
        if free.any():
            low, high = float(raw[free].min()), float(raw[free].max())
            normalized[free] = 0.0 if high == low else (raw[free] - low) / (high - low)
        normalized[blocked] = np.float32(config.occupied_potential)
    return {"raw_potential": raw.astype(np.float32), "normalized_potential": None if normalized is None else normalized.astype(np.float32), "blocked_mask": blocked}


def save_potential_frame_result(
    frame_index: int, fields: Mapping[str, np.ndarray | None], gradient: Mapping[str, np.ndarray], *,
    goal_cell: tuple[int, int], goal_metric: tuple[float, float], resolution_m: float, config: PotentialConfig,
    attractive_dir: str | Path, repulsive_dir: str | Path, combined_dir: str | Path, gradient_dir: str | Path,
    visualization_dir: str | Path, save_npy: bool = True, save_png: bool = True, save_gradient: bool = True,
    save_visualizations: bool = True,
) -> dict[str, Any]:
    from src.utils.io_utils import ensure_dir, save_image
    from .visualization import colorize_potential, draw_goal_marker, draw_gradient_vectors

    attractive, repulsive, combined = (np.asarray(fields[name], np.float32) for name in ("attractive", "repulsive", "combined"))
    blocked = np.asarray(fields["blocked_mask"], bool)
    stem = f"frame_{frame_index:06d}"
    result: dict[str, Any] = {
        "coordinate_frame": "camera_xz", "grid_type": "goal_conditioned_potential",
        "resolution_m": float(resolution_m), "shape": [int(combined.shape[0]), int(combined.shape[1])],
        "goal": {"row": int(goal_cell[0]), "col": int(goal_cell[1]), "x_m": float(goal_metric[0]), "z_m": float(goal_metric[1])},
        "attractive_mode": config.attractive_mode, "attractive_gain": float(config.attractive_gain),
        "repulsive_gain": float(config.repulsive_gain), "repulsive_influence_radius_m": float(config.repulsive_influence_radius_m),
        "cost_weight": float(config.cost_weight), "unknown_policy": config.unknown_policy,
        "minimum_potential": float(combined[~blocked].min()) if (~blocked).any() else float(config.occupied_potential),
        "maximum_free_potential": float(combined[~blocked].max()) if (~blocked).any() else float(config.occupied_potential),
        "blocked_cell_count": int(blocked.sum()),
    }
    if save_npy:
        for field, directory, key in ((attractive, attractive_dir, "attractive_path"), (repulsive, repulsive_dir, "repulsive_path"), (combined, combined_dir, "combined_path")):
            path = ensure_dir(directory) / f"{stem}.npy"; np.save(path, field, allow_pickle=False); result[key] = str(path)
    if save_gradient:
        path = ensure_dir(gradient_dir) / f"{stem}.npz"; np.savez_compressed(path, **gradient); result["gradient_path"] = str(path)
    if save_png:
        for field, directory in ((attractive, attractive_dir), (repulsive, repulsive_dir), (combined, combined_dir)):
            save_image(colorize_potential(field, blocked), Path(directory) / f"{stem}.png")
    if save_visualizations:
        image = draw_gradient_vectors(draw_goal_marker(colorize_potential(combined, blocked), goal_cell), gradient)
        result["visualization_path"] = str(save_image(image, Path(visualization_dir) / f"{stem}.png"))
    return result
