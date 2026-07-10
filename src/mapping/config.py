"""Configuration for semantic occupancy and traversability grids."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class MappingConfig:
    unknown_value: int = -1
    free_value: int = 0
    occupied_value: int = 100
    free_cost: float = 0.0
    occupied_cost: float = 1.0
    unknown_policy: str = "nan"
    semantic_costs: Mapping[str, float] = field(default_factory=dict)
    inflation_enabled: bool = True
    inflation_radius_m: float = 1.0
    inflation_decay: str = "linear"

    def validate(self) -> "MappingConfig":
        if len({self.unknown_value, self.free_value, self.occupied_value}) != 3:
            raise ValueError("Occupancy state values must be distinct.")
        values = [self.free_cost, self.occupied_cost, self.inflation_radius_m, *self.semantic_costs.values()]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Mapping numeric values must be finite.")
        if not 0.0 <= self.free_cost <= 1.0 or not 0.0 <= self.occupied_cost <= 1.0:
            raise ValueError("Cost values must be within [0, 1].")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.semantic_costs.values()):
            raise ValueError("Semantic costs must be within [0, 1].")
        if self.inflation_radius_m < 0.0:
            raise ValueError("Inflation radius must be non-negative.")
        if self.unknown_policy != "nan":
            raise ValueError("Only the 'nan' unknown policy is supported.")
        if self.inflation_decay != "linear":
            raise ValueError("Only linear inflation decay is supported.")
        return self

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MappingConfig":
        occupancy = data.get("occupancy", {})
        cost = data.get("cost", {})
        inflation = data.get("inflation", {})
        return cls(
            unknown_value=int(occupancy.get("unknown_value", -1)),
            free_value=int(occupancy.get("free_value", 0)),
            occupied_value=int(occupancy.get("occupied_value", 100)),
            free_cost=float(cost.get("free_cost", 0.0)),
            occupied_cost=float(cost.get("occupied_cost", 1.0)),
            unknown_policy=str(cost.get("unknown_policy", "nan")),
            semantic_costs={str(k).lower(): float(v) for k, v in data.get("semantic_costs", {}).items()},
            inflation_enabled=bool(inflation.get("enabled", True)),
            inflation_radius_m=float(inflation.get("radius_m", 1.0)),
            inflation_decay=str(inflation.get("decay", "linear")),
        ).validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "occupancy": {"unknown_value": self.unknown_value, "free_value": self.free_value, "occupied_value": self.occupied_value},
            "cost": {"free_cost": self.free_cost, "occupied_cost": self.occupied_cost, "unknown_policy": self.unknown_policy},
            "semantic_costs": dict(self.semantic_costs),
            "inflation": {"enabled": self.inflation_enabled, "radius_m": self.inflation_radius_m, "decay": self.inflation_decay},
        }
