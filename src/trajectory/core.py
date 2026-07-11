import numpy as np
from src.planner.coordinates import grid_path_to_metric, metric_path_to_grid
def _line(a,b):
 r0,c0=map(int,a); r1,c1=map(int,b); n=max(abs(r1-r0),abs(c1-c0)); return [(round(r0+(r1-r0)*i/n),round(c0+(c1-c0)*i/n)) for i in range(n+1)] if n else [tuple(a)]
def _safe(a,b,grid,corner=True):
 cells=_line(a,b)
 for i,(r,c) in enumerate(cells):
  if not(0<=r<grid.shape[0] and 0<=c<grid.shape[1]) or grid[r,c]!=0:return False
  if corner and i:
   pr,pc=cells[i-1]
   if r!=pr and c!=pc and (grid[pr,c]!=0 or grid[r,pc]!=0):return False
 return True
def shortcut_grid_path(path_rc,occupancy_grid,prevent_corner_cutting=True):
 p=np.asarray(path_rc,np.int32); grid=np.asarray(occupancy_grid)
 if len(p)<2:return p.copy()
 out=[p[0]]; i=0
 while i<len(p)-1:
  j=len(p)-1
  while j>i+1 and not _safe(p[i],p[j],grid,prevent_corner_cutting):j-=1
  out.append(p[j]); i=j
 return np.asarray(out,np.int32)
def smooth_metric_path(path_xz,method="chaikin",iterations=2,preserve_endpoints=True):
 p=np.asarray(path_xz,np.float32).copy()
 for _ in range(iterations if method=="chaikin" else 0):
  if len(p)<2:break
  q=np.empty((2*(len(p)-1),2),np.float32); q[0::2]=.75*p[:-1]+.25*p[1:];q[1::2]=.25*p[:-1]+.75*p[1:]
  p=np.vstack((p[0],q,p[-1])) if preserve_endpoints else q
 return p
def validate_metric_path_collision(path_xz,grid,bev,step):
 p=np.asarray(path_xz,float)
 try: cells=metric_path_to_grid(p,bev)
 except ValueError:return False
 return all(_safe(a,b,np.asarray(grid)) for a,b in zip(cells,cells[1:])) and all(np.asarray(grid)[tuple(c)]==0 for c in cells)
def resample_path_by_distance(path_xz,spacing_m):
 p=np.asarray(path_xz,np.float32); p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-7]] if len(p)>1 else p
 if len(p)<2:return p.copy()
 d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]; samples=np.r_[np.arange(0,d[-1],spacing_m),d[-1]]
 return np.column_stack([np.interp(samples,d,p[:,i]) for i in range(2)]).astype(np.float32)
def geometry(p):
 p=np.asarray(p,np.float32); arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))].astype(np.float32) if len(p) else np.empty(0,np.float32)
 if len(p)<2:return {"positions_xz":p,"arc_length_m":arc,"heading_rad":np.zeros(len(p),np.float32),"curvature_1pm":np.zeros(len(p),np.float32)}
 dx=np.gradient(p[:,0],arc,edge_order=1);dz=np.gradient(p[:,1],arc,edge_order=1);h=np.unwrap(np.arctan2(dx,dz)); cur=np.gradient(h,arc,edge_order=1)
 return {"positions_xz":p,"arc_length_m":arc,"heading_rad":h.astype(np.float32),"curvature_1pm":np.nan_to_num(cur).astype(np.float32)}
def generate_trajectory(path,grid,bev,config,inflated_cost_grid=None):
 p=np.asarray(path,np.int32)
 if len(p)==0:return {"status":"invalid_input_path","collision_free":False}
 s=shortcut_grid_path(p,grid) if config.shortcut_enabled else p.copy(); metric=grid_path_to_metric(s,bev); candidate=metric; used=0; fallback=False
 if config.smoothing_enabled:
  for n in range(config.smoothing_iterations,-1,-1):
   q=smooth_metric_path(metric,config.smoothing_method,n,config.preserve_endpoints)
   if validate_metric_path_collision(q,grid,bev,config.collision_check_step_m):candidate=q;used=n;break
  else:fallback=True
 final=resample_path_by_distance(candidate,config.resample_spacing_m)
 if not validate_metric_path_collision(final,grid,bev,config.collision_check_step_m):return {"status":"collision_after_smoothing","collision_free":False}
 t=geometry(final); return {"status":"success","source_path_type":"shortcut","shortcut_path_rc":s,"smoothed_path_xz":candidate,"trajectory":t,"collision_free":True,"smoothing_fallback_used":fallback,"smoothing_fallback_reason":"fallback_to_shortcut" if fallback else None,"diagnostics":{"smoothing_iterations_used":used,"maximum_observed_curvature":float(np.abs(t['curvature_1pm']).max()) if len(final) else 0.}}
