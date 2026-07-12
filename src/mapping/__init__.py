"""Semantic mapping extension point."""
from .config import MappingConfig
from .cost_grid import create_semantic_cost_grid, occupancy_to_cost_grid
from .inflation import inflate_cost_grid
from .occupancy import create_semantic_occupancy_grid, save_mapping_frame_result
from .visualization import colorize_cost_grid, colorize_occupancy_grid
from .temporal import TemporalOccupancyFusion

__all__ = [
    "MappingConfig", "create_semantic_occupancy_grid", "occupancy_to_cost_grid",
    "create_semantic_cost_grid", "inflate_cost_grid", "save_mapping_frame_result",
    "colorize_occupancy_grid", "colorize_cost_grid",
    "TemporalOccupancyFusion",
]
