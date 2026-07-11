import numpy as np
from src.planner import PlannerConfig, heuristic, plan_astar, plan_hybrid, validate_grid_path

def test_astar_path_rules_cost_and_limit():
    free=np.zeros((5,5),np.int16); costs=np.zeros((5,5),np.float32)
    direct=plan_astar(free,costs,(4,0),(0,4),PlannerConfig())
    assert direct['reached_goal']; validate_grid_path(direct['path_rc'],free,(4,0),(0,4),8,1)
    costs[2,:]=1; costs[2,0]=costs[2,4]=0
    avoided=plan_astar(free,costs,(4,2),(0,2),PlannerConfig(traversal_cost_weight=10))
    assert avoided['reached_goal'] and not any((cell == (2,2)).all() for cell in avoided['path_rc'])
    free[2,:]=100; free[2,4]=-1
    assert plan_astar(free,costs,(4,0),(0,0),PlannerConfig())['status'] == 'no_path'
    assert plan_astar(np.zeros((5,5),np.int16),costs,(4,0),(0,4),PlannerConfig(max_expansions=1))['status'] == 'max_expansions_exceeded'

def test_astar_determinism_heuristics_and_hybrid_fallback():
    free=np.zeros((4,4),np.int16); cost=np.zeros((4,4),np.float32); cfg=PlannerConfig()
    one=plan_astar(free,cost,(3,0),(0,3),cfg); two=plan_astar(free,cost,(3,0),(0,3),cfg)
    assert np.array_equal(one['path_rc'],two['path_rc'])
    assert heuristic((3,0),(0,3),4) == 6 and heuristic((3,0),(0,3),8) < 6
    potential=np.ones((4,4),np.float32)
    result=plan_hybrid(potential,free,cost,(3,0),(0,3),PlannerConfig(algorithm='hybrid'))
    assert result['reached_goal'] and result['fallback_used'] and result['fallback_reason'] == 'local_minimum'
    success=plan_hybrid(np.array([[3,2,1,0]]*4,np.float32),free,cost,(0,0),(0,3),PlannerConfig(algorithm='hybrid'))
    assert success['selected_algorithm'] == 'gradient_descent' and not success['fallback_used']

def test_hybrid_invalid_start_does_not_fallback():
    occupancy=np.zeros((3,3),np.int16); occupancy[2,0]=100
    result=plan_hybrid(np.ones((3,3)),occupancy,np.zeros((3,3)),(2,0),(0,2),PlannerConfig(algorithm='hybrid'))
    assert not result['fallback_used'] and result['astar_result'] is None
