import numpy as np

from src.bev import BEVConfig
from src.perception.pipeline import PerceptionPipeline
from src.planner import PlannerConfig


def _pipeline_with_path():
    pipeline = PerceptionPipeline(None, None)
    occupancy = np.zeros((6, 5), np.int16)
    bev = BEVConfig(-2.5, 2.5, 0, 6, 1)
    path = np.asarray([[5, 2], [4, 2], [3, 2], [2, 2]], np.int32)
    pipeline._last_successful_planner_memory = {
        "path_rc":path, "potential_grid":np.zeros_like(occupancy, dtype=np.float32),
        "occupancy_grid":occupancy, "cost_grid":np.zeros_like(occupancy, dtype=np.float32),
        "bev_config":bev, "source_algorithm":"astar", "frame_index":0,
        "goal_cell":(2,2), "start_cell":(5,2),
    }
    return pipeline, occupancy, bev


def test_reuse_is_collision_checked_and_does_not_reset_age():
    pipeline, occupancy, bev = _pipeline_with_path()
    context={"occupancy_grid":occupancy,"cost_grid":np.zeros_like(occupancy,dtype=np.float32),"bev_config":bev}
    reused=pipeline._reuse_last_valid_path(context, 3, 15, PlannerConfig(), "PLANNING_INTERVAL_SKIP")
    assert reused["status"] == "reused"
    assert reused["collision_validated"] is True
    assert reused["source_frame_index"] == 0
    assert reused["path_age_frames"] == 3
    later=pipeline._reuse_last_valid_path(context, 16, 15, PlannerConfig(), "NO_FORWARD_GOAL")
    assert later is None


def test_reuse_is_rejected_when_current_occupancy_blocks_path():
    pipeline, occupancy, bev = _pipeline_with_path()
    occupancy[3,2] = 100
    context={"occupancy_grid":occupancy,"cost_grid":np.zeros_like(occupancy,dtype=np.float32),"bev_config":bev}
    assert pipeline._reuse_last_valid_path(context, 1, 15, PlannerConfig(), "NO_FORWARD_GOAL") is None
