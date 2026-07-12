from __future__ import annotations
from collections import Counter
import numpy as np
from .neighbors import iter_free_neighbors
from .validation import validate_start_cell

def plan_gradient_descent(potential_grid, occupancy_grid, start_cell, goal_cell, config):
    config.validate(); potential=np.asarray(potential_grid,np.float64); occupancy=np.asarray(occupancy_grid)
    base = {"path_rc": np.empty((0,2),np.int32), "status":"invalid_start", "reached_goal":False, "step_count":0, "final_cell":tuple(start_cell), "final_distance_to_goal_cells":float("inf"), "termination_reason":"invalid_start", "diagnostics":{"local_minimum_detected":False,"revisit_detected":False,"blocked_neighbor_count":0}}
    if potential.ndim != 2 or occupancy.shape != potential.shape: raise ValueError("Potential and occupancy grids must be matching 2D arrays.")
    try: current=validate_start_cell(start_cell, occupancy)
    except ValueError: return base
    goal=tuple(map(int,goal_cell))
    if not (0<=goal[0]<occupancy.shape[0] and 0<=goal[1]<occupancy.shape[1]) or occupancy[goal] != 0:
        base.update(status="invalid_goal", termination_reason="invalid_goal"); return base
    path=[current]; visits=Counter(path); connectivity=8 if config.connectivity==8 and config.allow_diagonal else 4
    status="max_steps_exceeded"; diag=base["diagnostics"]
    for _ in range(config.max_steps):
        distance=float(np.linalg.norm(np.subtract(current,goal)))
        if distance <= config.goal_tolerance_cells: status="success"; break
        candidates=[]; blocked=0
        free_neighbors=list(iter_free_neighbors(occupancy,current,connectivity=connectivity,prevent_corner_cutting=config.prevent_corner_cutting))
        blocked=(8 if connectivity == 8 else 4)-len(free_neighbors)
        for order,(nr,nc,dr,dc) in enumerate(free_neighbors):
            candidates.append((potential[nr,nc], float(np.hypot(nr-goal[0],nc-goal[1])), order, (nr,nc)))
        diag["blocked_neighbor_count"] += blocked
        if not candidates: status="no_valid_neighbor"; break
        best=min(candidates)
        if best[0] > potential[current] - config.minimum_potential_drop:
            status="local_minimum"; diag["local_minimum_detected"]=True; break
        current=best[3]; path.append(current); visits[current]+=1
        if visits[current] > config.revisit_limit: status="cycle_detected"; diag["revisit_detected"]=True; break
    array=np.asarray(path,np.int32); final=tuple(array[-1]); distance=float(np.linalg.norm(np.subtract(final,goal)))
    return {**base,"path_rc":array,"status":status,"reached_goal":status=="success","step_count":len(path)-1,"final_cell":final,"final_distance_to_goal_cells":distance,"termination_reason":"goal_reached" if status=="success" else status,"diagnostics":diag}
