from __future__ import annotations
import numpy as np
def draw_path_on_potential(potential, path_rc, start_cell, goal_cell):
    import cv2
    grid=np.asarray(potential,np.float32); low, high=float(grid.min()),float(grid.max()); image=cv2.applyColorMap(np.uint8(255*(grid-low)/(high-low) if high>low else grid*0),cv2.COLORMAP_VIRIDIS)
    return _draw(image,path_rc,start_cell,goal_cell)
def draw_path_on_occupancy(occupancy,path_rc,start_cell,goal_cell):
    grid=np.asarray(occupancy); image=np.full((*grid.shape,3),128,np.uint8); image[grid==0]=(255,255,255); image[grid==100]=(0,0,0)
    return _draw(image,path_rc,start_cell,goal_cell)
def _draw(image,path,start,goal):
    import cv2
    out=image.copy(); path=np.asarray(path)
    if len(path)>1: cv2.polylines(out,[path[:,::-1].reshape(-1,1,2)],False,(0,0,255),1)
    for cell,color in ((start,(255,0,0)),(goal,(0,255,0))): cv2.circle(out,(int(cell[1]),int(cell[0])),2,color,-1)
    return out
