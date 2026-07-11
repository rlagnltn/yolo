from dataclasses import dataclass
import math
@dataclass(frozen=True)
class TrajectoryConfig:
 shortcut_enabled: bool=True; smoothing_enabled: bool=True; smoothing_method:str="chaikin"; smoothing_iterations:int=2; resample_spacing_m:float=.2; collision_check_step_m:float=.05; preserve_endpoints:bool=True; maximum_curvature:float|None=None
 def validate(self):
  if self.smoothing_method not in {"chaikin","none"}: raise ValueError("Unsupported smoothing method.")
  if self.smoothing_iterations<0 or self.resample_spacing_m<=0 or self.collision_check_step_m<=0: raise ValueError("Trajectory spacing and iterations are invalid.")
  if self.maximum_curvature is not None and (not math.isfinite(self.maximum_curvature) or self.maximum_curvature<=0): raise ValueError("Maximum curvature must be positive.")
  return self
 @classmethod
 def from_dict(cls,d): return cls(**{k:v for k,v in d.get("trajectory",d).items() if k in cls.__dataclass_fields__}).validate()
