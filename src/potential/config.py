from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class PotentialConfig:
    attractive_gain: float = 1.0
    attractive_mode: str = "quadratic"
    attractive_saturation_distance_m: float | None = 20.0
    repulsive_gain: float = 5.0
    repulsive_influence_radius_m: float = 3.0
    cost_weight: float = 1.0
    occupied_potential: float = 1_000_000.0
    unknown_policy: str = "blocked"
    normalize_output: bool = True

    def validate(self) -> "PotentialConfig":
        values = [self.attractive_gain, self.repulsive_gain, self.repulsive_influence_radius_m,
                  self.cost_weight, self.occupied_potential]
        if self.attractive_saturation_distance_m is not None:
            values.append(self.attractive_saturation_distance_m)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Potential configuration values must be finite.")
        if self.attractive_gain < 0 or self.repulsive_gain < 0 or self.cost_weight < 0:
            raise ValueError("Potential gains and cost weight must be non-negative.")
        if self.repulsive_influence_radius_m <= 0:
            raise ValueError("Repulsive influence radius must be positive.")
        if self.attractive_saturation_distance_m is not None and self.attractive_saturation_distance_m <= 0:
            raise ValueError("Attractive saturation distance must be positive.")
        if self.attractive_mode not in {"quadratic", "conic"}:
            raise ValueError("Attractive mode must be 'quadratic' or 'conic'.")
        if self.unknown_policy not in {"blocked", "high_cost"}:
            raise ValueError("Unknown policy must be 'blocked' or 'high_cost'.")
        if self.occupied_potential <= 0:
            raise ValueError("Occupied potential must be positive.")
        return self

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PotentialConfig":
        attractive, repulsive = data.get("attractive", {}), data.get("repulsive", {})
        combination = data.get("combination", {})
        return cls(
            attractive_gain=float(attractive.get("gain", 1.0)),
            attractive_mode=str(attractive.get("mode", "quadratic")),
            attractive_saturation_distance_m=(None if attractive.get("saturation_distance_m") is None
                                               else float(attractive.get("saturation_distance_m"))),
            repulsive_gain=float(repulsive.get("gain", 5.0)),
            repulsive_influence_radius_m=float(repulsive.get("influence_radius_m", 3.0)),
            cost_weight=float(combination.get("cost_weight", 1.0)),
            occupied_potential=float(combination.get("occupied_potential", 1_000_000.0)),
            unknown_policy=str(combination.get("unknown_policy", "blocked")),
            normalize_output=bool(combination.get("normalize_output", True)),
        ).validate()
