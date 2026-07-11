"""Streaming image-space planning helpers for video-plan clients."""
from dataclasses import dataclass
import numpy as np
import cv2

def normalized_to_pixel(x,y,width,height): return (int(np.clip(round(x*(width-1)),0,width-1)),int(np.clip(round(y*(height-1)),0,height-1)))
def pixel_to_normalized(x,y,width,height): return (float(x/max(width-1,1)),float(y/max(height-1,1)))
def pixel_to_grid(x,y,shape,width,height): return (int(np.clip(round(y*(shape[0]-1)/max(height-1,1)),0,shape[0]-1)),int(np.clip(round(x*(shape[1]-1)/max(width-1,1)),0,shape[1]-1)))
def grid_to_pixel(row,col,shape,width,height): return (int(np.clip(round(col*(width-1)/max(shape[1]-1,1)),0,width-1)),int(np.clip(round(row*(height-1)/max(shape[0]-1,1)),0,height-1)))
@dataclass
class TemporalPlanningState:
 previous_smoothed_potential: object=None; previous_trajectory: object=None; previous_goal: object=None; previous_shape: object=None; trajectory_reuse_age:int=0
 def reset(self): self.previous_smoothed_potential=self.previous_trajectory=self.previous_goal=self.previous_shape=None; self.trajectory_reuse_age=0
 def smooth_potential(self,current,occupied,alpha=.4,goal=None):
  current=np.asarray(current,np.float32); occupied=np.asarray(occupied,bool)
  if self.previous_smoothed_potential is None or self.previous_smoothed_potential.shape!=current.shape or self.previous_goal!=goal: out=current.copy()
  else: out=alpha*current+(1-alpha)*self.previous_smoothed_potential
  out[occupied]=current[occupied]; out=np.nan_to_num(out); self.previous_smoothed_potential=out;self.previous_goal=goal;return out
 def stabilize(self,current,validator,alpha=.5,reuse=True,max_reuse_frames=3):
  if current is None:
   if reuse and self.previous_trajectory is not None and self.trajectory_reuse_age<max_reuse_frames and validator(self.previous_trajectory): self.trajectory_reuse_age+=1;return self.previous_trajectory,True
   return None,False
  if self.previous_trajectory is not None and validator(self.previous_trajectory):
   n=len(current); old=np.linspace(0,len(self.previous_trajectory)-1,n); prev=np.column_stack([np.interp(old,np.arange(len(self.previous_trajectory)),self.previous_trajectory[:,i]) for i in range(2)]); mixed=alpha*current+(1-alpha)*prev
   if validator(mixed): current=mixed.astype(np.float32)
  self.previous_trajectory=current;self.trajectory_reuse_age=0;return current,False
def render_overlay(frame,potential=None,trajectory_xz=None,grid_shape=None):
 out=np.asarray(frame).copy()
 if potential is not None:
  p=np.nan_to_num(np.asarray(potential,np.float32)); lo,hi=np.percentile(p,[2,98]); heat=cv2.applyColorMap(np.uint8(np.clip((p-lo)*255/max(hi-lo,1e-6),0,255)),cv2.COLORMAP_TURBO);out=cv2.addWeighted(out,.65,cv2.resize(heat,(out.shape[1],out.shape[0])),.35,0)
 if trajectory_xz is not None and grid_shape:
  pts=np.asarray(trajectory_xz); pix=[grid_to_pixel(r,c,grid_shape,out.shape[1],out.shape[0]) for r,c in pts]
  if len(pix)>1: cv2.polylines(out,[np.asarray(pix,np.int32)],False,(0,255,0),2)
 return out
