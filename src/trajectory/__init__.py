from .config import TrajectoryConfig
from .core import shortcut_grid_path,smooth_metric_path,validate_metric_path_collision,resample_path_by_distance,geometry,generate_trajectory
calculate_arc_length=lambda p: geometry(p)["arc_length_m"]
calculate_heading=lambda p: geometry(p)["heading_rad"]
calculate_curvature=lambda p: geometry(p)["curvature_1pm"]
