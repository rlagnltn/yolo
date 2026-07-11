from __future__ import annotations

import heapq
import itertools
import math
import numpy as np

from .validation import validate_start_cell

_FOUR = ((-1, 0), (0, 1), (1, 0), (0, -1))
_EIGHT = _FOUR + ((-1, 1), (1, 1), (1, -1), (-1, -1))

def heuristic(cell, goal, connectivity):
    dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
    return float(dr + dc) if connectivity == 4 else float(max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc))

def plan_astar(occupancy_grid, cost_grid, start_cell, goal_cell, config):
    config.validate(); occupancy=np.asarray(occupancy_grid); costs=np.asarray(cost_grid, np.float64)
    base={"path_rc":np.empty((0,2),np.int32),"status":"invalid_start","reached_goal":False,"step_count":0,"expanded_node_count":0,"path_cost":float("inf"),"final_cell":tuple(start_cell),"termination_reason":"invalid_start"}
    if occupancy.ndim != 2 or costs.shape != occupancy.shape: raise ValueError("Occupancy and cost grids must be matching 2D arrays.")
    try: start=validate_start_cell(start_cell,occupancy)
    except ValueError: return base
    goal=tuple(map(int,goal_cell))
    if not (0<=goal[0]<occupancy.shape[0] and 0<=goal[1]<occupancy.shape[1]) or occupancy[goal] != 0:
        return {**base,"status":"invalid_goal","termination_reason":"invalid_goal"}
    if start == goal: return {**base,"path_rc":np.asarray([start],np.int32),"status":"success","reached_goal":True,"path_cost":0.,"final_cell":start,"termination_reason":"goal_reached"}
    offsets=_EIGHT if config.connectivity == 8 and config.allow_diagonal else _FOUR; order=itertools.count(); queue=[]
    heapq.heappush(queue,(heuristic(start,goal,config.connectivity),heuristic(start,goal,config.connectivity),next(order),start)); g={start:0.}; parent={}; expanded=0
    while queue:
        _,_,_,current=heapq.heappop(queue)
        if current == goal:
            cells=[current]
            while current in parent: current=parent[current]; cells.append(current)
            cells.reverse(); path=np.asarray(cells,np.int32)
            return {**base,"path_rc":path,"status":"success","reached_goal":True,"step_count":len(path)-1,"expanded_node_count":expanded,"path_cost":g[goal],"final_cell":goal,"termination_reason":"goal_reached"}
        expanded += 1
        if expanded > config.max_expansions: return {**base,"status":"max_expansions_exceeded","expanded_node_count":expanded,"final_cell":current,"termination_reason":"max_expansions_exceeded"}
        for dr,dc in offsets:
            nxt=(current[0]+dr,current[1]+dc)
            if not (0<=nxt[0]<occupancy.shape[0] and 0<=nxt[1]<occupancy.shape[1]) or occupancy[nxt] != 0 or not np.isfinite(costs[nxt]): continue
            if dr and dc and config.prevent_corner_cutting and (occupancy[current[0],nxt[1]] != 0 or occupancy[nxt[0],current[1]] != 0): continue
            tentative=g[current] + math.hypot(dr,dc) + config.traversal_cost_weight * float(costs[nxt])
            if tentative < g.get(nxt,float("inf")):
                g[nxt]=tentative; parent[nxt]=current; h=heuristic(nxt,goal,config.connectivity)
                heapq.heappush(queue,(tentative+h,h,next(order),nxt))
    return {**base,"status":"no_path","expanded_node_count":expanded,"final_cell":start,"termination_reason":"no_path"}
