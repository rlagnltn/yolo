"""Select connected FREE start and goal cells from a current occupancy grid."""

from __future__ import annotations

from typing import Any

import numpy as np


def select_auto_free_cells(
    occupancy_grid: Any,
    bev_config: Any,
    *,
    centerline_half_width_m: float = 2.0,
    minimum_forward_distance_m: float = 5.0,
    fallback_forward_distances_m: tuple[float, ...] = (4.0, 3.0),
    previous_start_cell: tuple[int, int] | None = None,
    start_stability_radius_m: float = 0.5,
    alternative_start_search_radius_m: float = 0.5,
    connectivity: int = 8,
    prevent_corner_cutting: bool = True,
) -> dict[str, Any]:
    """Return deterministic start/goal cells from one connected observed-FREE region."""

    occupancy = np.asarray(occupancy_grid)
    if occupancy.ndim != 2:
        raise ValueError("Occupancy grid must be 2-dimensional.")
    if centerline_half_width_m <= 0 or minimum_forward_distance_m <= 0 or start_stability_radius_m < 0 or alternative_start_search_radius_m < 0:
        raise ValueError("Auto planner distances must be positive.")
    distance_tiers = tuple(dict.fromkeys((minimum_forward_distance_m, *fallback_forward_distances_m)))
    if any(distance <= 0 for distance in distance_tiers) or tuple(sorted(distance_tiers, reverse=True)) != distance_tiers:
        raise ValueError("Forward-distance tiers must be positive and descending.")
    rows, cols = np.indices(occupancy.shape)
    x_m = bev_config.x_min_m + (cols + .5) * bev_config.resolution_m
    z_m = bev_config.z_min_m + (occupancy.shape[0] - rows - .5) * bev_config.resolution_m
    free = occupancy == 0
    center_free = free & (np.abs(x_m) <= centerline_half_width_m)
    stats = {"free_cell_count": int(free.sum()), "centerline_free_cell_count": int(center_free.sum())}
    if not center_free.any():
        return {"status": "failed", "reason_code": "NO_FREE_START", "statistics": stats}
    component_labels = _label_free_components(occupancy, connectivity, prevent_corner_cutting)
    start, start_source = _select_start(
        center_free, z_m, previous_start_cell, start_stability_radius_m, bev_config.resolution_m
    )
    reachable = component_labels == component_labels[start]
    stats["start_selection_source"] = start_source
    stats["connected_free_cell_count"] = int(reachable.sum())
    stats["planner_reachable_cell_count"] = int(reachable.sum())
    selected_forward = None
    selected_distance = None
    for distance in distance_tiers:
        forward = reachable & (np.abs(x_m) <= centerline_half_width_m) & (z_m >= z_m[start] + distance)
        stats[f"forward_goal_candidate_count_{distance:g}m"] = int(forward.sum())
        if forward.any():
            selected_forward, selected_distance = forward, distance
            break
    if selected_forward is None and alternative_start_search_radius_m > 0:
        alternative = _find_alternative_start(
            component_labels, center_free, x_m, z_m, start,
            alternative_start_search_radius_m, distance_tiers[-1],
        )
        if alternative is not None:
            start, reachable = alternative
            stats["start_selection_source"] = "alternative_connected_component"
            stats["planner_reachable_cell_count"] = int(reachable.sum())
            stats["connected_free_cell_count"] = int(reachable.sum())
            for distance in distance_tiers:
                forward = reachable & (np.abs(x_m) <= centerline_half_width_m) & (z_m >= z_m[start] + distance)
                stats[f"forward_goal_candidate_count_{distance:g}m"] = int(forward.sum())
                if forward.any():
                    selected_forward, selected_distance = forward, distance
                    break
    stats["forward_goal_candidate_count"] = 0 if selected_forward is None else int(selected_forward.sum())
    if selected_forward is None:
        return {
            "status": "failed", "reason_code": "NO_FORWARD_GOAL", "start_cell": start,
            "requested_forward_distance_m": minimum_forward_distance_m, "statistics": stats,
        }
    best_z = z_m[selected_forward].max()
    candidates = np.argwhere(selected_forward & (z_m == best_z))
    goal = tuple(int(value) for value in min(candidates.tolist(), key=lambda cell: (abs(x_m[tuple(cell)]), cell[1])))
    horizon_status = "full_horizon" if selected_distance == minimum_forward_distance_m else "short_horizon"
    return {
        "status": "selected", "reason_code": "SELECTED", "horizon_status": horizon_status,
        "requested_forward_distance_m": minimum_forward_distance_m,
        "selected_forward_distance_m": selected_distance,
        "start_cell": start, "goal_cell": goal, "statistics": stats,
    }


def _find_alternative_start(component_labels, center_free, x_m, z_m, primary_start, radius_m, minimum_horizon_m):
    nearest_z = float(z_m[center_free].min())
    candidates = np.argwhere(center_free & (z_m <= nearest_z + radius_m + 1e-9))
    choices = []
    candidate_labels = component_labels[candidates[:, 0], candidates[:, 1]]
    for label_id in np.unique(candidate_labels):
        if label_id == 0:
            continue
        component = component_labels == label_id
        component_candidates = np.argwhere(component & center_free & (z_m <= nearest_z + radius_m + 1e-9))
        if not len(component_candidates):
            continue
        start = tuple(map(int, min(component_candidates.tolist(), key=lambda item: (z_m[tuple(item)], abs(x_m[tuple(item)]), item[0], item[1]))))
        center_component = component & center_free
        forward_extent = float(z_m[center_component].max() - z_m[start])
        if forward_extent + 1e-9 < minimum_horizon_m:
            continue
        choices.append((-forward_extent, np.linalg.norm(np.subtract(start, primary_start)), abs(x_m[start]), start, component))
    if not choices:
        return None
    *_, start, component = min(choices)
    if start == primary_start:
        return None
    return start, component


def _select_start(center_free, z_m, previous_start_cell, radius_m, resolution_m):
    if previous_start_cell is not None:
        previous = np.asarray(previous_start_cell, dtype=float)
        cells = np.argwhere(center_free)
        distances_m = np.linalg.norm(cells - previous, axis=1) * resolution_m
        stable = cells[distances_m <= radius_m]
        if len(stable):
            chosen = min(stable.tolist(), key=lambda cell: (np.linalg.norm(np.asarray(cell)-previous), z_m[tuple(cell)], cell[0], cell[1]))
            return tuple(map(int, chosen)), "previous_start_neighborhood"
    start_index = np.argmin(np.where(center_free, z_m, np.inf))
    return tuple(int(value) for value in np.unravel_index(start_index, center_free.shape)), "nearest_centerline"


def _connected_component(occupancy: np.ndarray, start: tuple[int, int], *, connectivity: int, prevent_corner_cutting: bool) -> np.ndarray:
    labels = _label_free_components(occupancy, connectivity, prevent_corner_cutting)
    label_id = int(labels[start])
    return labels == label_id if label_id else np.zeros_like(occupancy, dtype=bool)


def _label_free_components(occupancy: np.ndarray, connectivity: int, prevent_corner_cutting: bool) -> np.ndarray:
    """Label planner-compatible FREE components with OpenCV's C implementation.

    With corner cutting disabled, every legal diagonal has an orthogonal FREE
    two-step connection, so 8-neighbor planner reachability is exactly the same
    component partition as 4-connected FREE labeling.
    """
    import cv2

    if connectivity not in {4, 8}:
        raise ValueError("Connectivity must be 4 or 8.")
    effective_connectivity = 4 if connectivity == 4 or prevent_corner_cutting else 8
    free = (np.asarray(occupancy) == 0).astype(np.uint8, copy=False)
    _, labels = cv2.connectedComponents(free, connectivity=effective_connectivity)
    return labels
