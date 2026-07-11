from __future__ import annotations
from .astar import plan_astar
from .gradient_descent import plan_gradient_descent

def plan_hybrid(potential_grid, occupancy_grid, cost_grid, start_cell, goal_cell, config):
    gradient=plan_gradient_descent(potential_grid,occupancy_grid,start_cell,goal_cell,config)
    if gradient["reached_goal"]:
        return {"path_rc":gradient["path_rc"],"status":"success","reached_goal":True,"selected_algorithm":"gradient_descent","fallback_used":False,"fallback_reason":None,"gradient_result":gradient,"astar_result":None,"termination_reason":gradient["termination_reason"]}
    if not config.fallback_enabled or gradient["status"] not in config.fallback_on:
        return {"path_rc":gradient["path_rc"],"status":"failed","reached_goal":False,"selected_algorithm":"gradient_descent","fallback_used":False,"fallback_reason":None,"gradient_result":gradient,"astar_result":None,"termination_reason":gradient["termination_reason"]}
    astar=plan_astar(occupancy_grid,cost_grid,start_cell,goal_cell,config)
    return {"path_rc":astar["path_rc"],"status":"success" if astar["reached_goal"] else "failed","reached_goal":astar["reached_goal"],"selected_algorithm":"astar_fallback","fallback_used":True,"fallback_reason":gradient["status"],"gradient_result":gradient,"astar_result":astar,"termination_reason":astar["termination_reason"]}
