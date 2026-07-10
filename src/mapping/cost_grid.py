"""Traversability cost-grid creation."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from .config import MappingConfig
from .occupancy import create_semantic_occupancy_grid


def occupancy_to_cost_grid(occupancy_grid: Any, config: MappingConfig) -> np.ndarray:
    grid = np.asarray(occupancy_grid)
    if grid.ndim != 2:
        raise ValueError("occupancy_grid must be a 2D array.")
    valid = {config.unknown_value, config.free_value, config.occupied_value}
    if not np.isin(grid, list(valid)).all():
        raise ValueError("occupancy_grid contains an unsupported state value.")
    result = np.full(grid.shape, np.nan, dtype=np.float32)
    result[grid == config.free_value] = np.float32(config.free_cost)
    result[grid == config.occupied_value] = np.float32(config.occupied_cost)
    return result


def create_semantic_cost_grid(
    class_grid: Any, observed_mask: Any, id2label: Mapping[int, str], config: MappingConfig
) -> np.ndarray:
    occupancy = create_semantic_occupancy_grid(class_grid, observed_mask, id2label, config)["occupancy_grid"]
    result = occupancy_to_cost_grid(occupancy, config)
    labels = {int(key): str(value).strip().lower() for key, value in id2label.items()}
    grid = np.asarray(class_grid)
    observed = np.asarray(observed_mask, dtype=bool)
    for class_id, label in labels.items():
        if label in config.semantic_costs:
            result[observed & (grid == class_id)] = np.float32(config.semantic_costs[label])
    return result
