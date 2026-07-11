from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PlannerConfig:
    connectivity: int = 8
    max_steps: int = 2000
    goal_tolerance_cells: int = 1
    minimum_potential_drop: float = 1e-6
    allow_diagonal: bool = True
    prevent_corner_cutting: bool = True
    revisit_limit: int = 2
    local_minimum_window: int = 5
    algorithm: str = "gradient_descent"
    fallback_enabled: bool = True
    fallback_algorithm: str = "astar"
    fallback_on: tuple[str, ...] = ("local_minimum", "cycle_detected", "no_valid_neighbor", "max_steps_exceeded")
    max_expansions: int = 100_000
    traversal_cost_weight: float = 5.0

    def validate(self) -> "PlannerConfig":
        if self.connectivity not in {4, 8}: raise ValueError("Planner connectivity must be 4 or 8.")
        if self.max_steps <= 0: raise ValueError("Planner max_steps must be positive.")
        if self.goal_tolerance_cells < 0: raise ValueError("Planner goal_tolerance_cells must be non-negative.")
        if self.minimum_potential_drop < 0: raise ValueError("Planner minimum_potential_drop must be non-negative.")
        if self.revisit_limit < 1: raise ValueError("Planner revisit_limit must be at least 1.")
        if self.local_minimum_window < 1: raise ValueError("Planner local_minimum_window must be at least 1.")
        if self.algorithm not in {"gradient_descent", "astar", "hybrid"}: raise ValueError("Planner algorithm must be gradient_descent, astar, or hybrid.")
        if self.fallback_algorithm != "astar": raise ValueError("Planner fallback_algorithm must be astar.")
        if self.max_expansions <= 0: raise ValueError("A* max_expansions must be positive.")
        if self.traversal_cost_weight < 0: raise ValueError("A* traversal_cost_weight must be non-negative.")
        return self

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlannerConfig":
        values = data.get("planner", data)
        hybrid, astar = data.get("hybrid", {}), data.get("astar", {})
        supplied = {key: values[key] for key in cls.__dataclass_fields__ if key in values}
        supplied.update({key: hybrid[key] for key in ("fallback_enabled", "fallback_algorithm") if key in hybrid})
        if "fallback_on" in hybrid: supplied["fallback_on"] = tuple(hybrid["fallback_on"])
        supplied.update({key: astar[key] for key in ("max_expansions", "traversal_cost_weight") if key in astar})
        return cls(**supplied).validate()
