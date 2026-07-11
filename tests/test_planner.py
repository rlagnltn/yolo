import numpy as np
import pytest
from src.bev import BEVConfig
from src.planner import PlannerConfig, calculate_path_length_m, grid_path_to_metric, metric_path_to_grid, plan_gradient_descent, resolve_start_cell, validate_grid_path, validate_start_cell

def test_config_and_coordinates():
    with pytest.raises(ValueError): PlannerConfig(connectivity=6).validate()
    cfg=BEVConfig(-1, 2, 0, 3, 1); path=np.array([[2,1],[1,1],[0,1]])
    assert np.array_equal(metric_path_to_grid(grid_path_to_metric(path,cfg),cfg),path)
    assert calculate_path_length_m(grid_path_to_metric(path,cfg)) == pytest.approx(2)

def test_gradient_planner_tie_break_corner_and_local_minimum():
    free=np.zeros((3,3),np.int16); potential=np.array([[2,1,0],[3,2,1],[4,3,2]],np.float32)
    result=plan_gradient_descent(potential,free,(2,0),(0,2),PlannerConfig())
    assert result['reached_goal']; validate_grid_path(result['path_rc'],free,(2,0),(0,2),8,1)
    blocked=free.copy(); blocked[1,0]=100; blocked[2,1]=100
    result=plan_gradient_descent(potential,blocked,(2,0),(0,2),PlannerConfig())
    assert result['status'] == 'no_valid_neighbor'
    assert plan_gradient_descent(np.ones((3,3),np.float32),free,(2,0),(0,2),PlannerConfig())['status'] == 'local_minimum'

def test_start_validation():
    grid=np.zeros((2,2),np.int16); grid[0,0]=100
    with pytest.raises(ValueError): validate_start_cell((0,0),grid)
    assert resolve_start_cell((1,1),None,BEVConfig(0,2,0,2,1)) == (1,1)
