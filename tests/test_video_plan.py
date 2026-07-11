import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.bev import BEVConfig
from src.video_plan import (TemporalPlanningState, grid_to_pixel, json_safe, normalized_to_pixel,
                            pixel_to_grid, pixel_to_normalized, render_overlay)
from src.video_runner import VideoPlanOptions, run_video_plan


def test_coordinates_temporal_and_json_safety():
    assert normalized_to_pixel(.5, 1, 11, 11) == (5, 10)
    assert pixel_to_normalized(5, 10, 11, 11) == (.5, 1.)
    assert pixel_to_grid(10, 0, (3, 3), 11, 11) == (0, 2)
    assert grid_to_pixel(2, 2, (3, 3), 11, 11) == (10, 10)
    with pytest.raises(ValueError): normalized_to_pixel(1.1, 0, 10, 10)
    state=TemporalPlanningState(); current=np.array([[0,2],[4,6]],np.float32); occupied=np.array([[0,1],[0,0]],bool)
    np.testing.assert_array_equal(state.smooth_potential(current,occupied,.5),current)
    smoothed=state.smooth_potential(np.zeros((2,2),np.float32),occupied,.5)
    assert smoothed[0,0]==0 and smoothed[0,1]==0
    assert json_safe({"a":np.array([1]),"bad":float("nan")}) == {"a":[1],"bad":None}


def test_trajectory_stabilization_reuse_and_overlay():
    state=TemporalPlanningState(); current=np.array([[0,0],[1,1]],np.float32)
    first,source=state.stabilize_trajectory(current,lambda p:True); assert source=="current"
    reused,source=state.stabilize_trajectory(None,lambda p:True,max_reuse_frames=1); assert source=="reused_previous"
    missing,source=state.stabilize_trajectory(None,lambda p:True,max_reuse_frames=1); assert missing is None
    blocked,source=TemporalPlanningState(previous_trajectory=current).stabilize_trajectory(None,lambda p:False); assert blocked is None
    frame=np.zeros((20,30,3),np.uint8); image=render_overlay(frame,potential=np.ones((2,2)),status_text="no path")
    assert image.shape==frame.shape and image.dtype==np.uint8


class FakePipeline:
    def __init__(self): self.calls=0
    def process_frame(self,frame,index,timestamp,*args,temporal_state=None,temporal_options=None,**kwargs):
        self.calls+=1; grid=np.zeros((4,4),np.int16); potential=np.full((4,4),float(self.calls),np.float32)
        potential=temporal_state.smooth_potential(potential,grid==100,temporal_options["potential_alpha"])
        bev=BEVConfig(0,4,0,4,1); path=np.array([[3,1],[2,1],[1,1],[0,1]],np.int32)
        self._last_planner_memory={"path_rc":path,"potential_grid":potential,"occupancy_grid":grid,"cost_grid":np.zeros((4,4),np.float32),"bev_config":bev,"source_algorithm":"astar","frame_index":index,"goal_cell":(0,1),"start_cell":(3,1)}
        positions=np.array([[1.5,.5],[1.5,1.5],[1.5,2.5],[1.5,3.5]],np.float32)
        self._last_trajectory_memory={"positions_xz":positions,"heading_rad":np.zeros(4,np.float32),"curvature_1pm":np.zeros(4,np.float32),"shortcut_path_rc":path}
        return {"detections":[],"planner":{"status":"success","reached_goal":True,"selected_algorithm":"astar","fallback_used":False},"trajectory":{"status":"success"},"errors":[]}


def _make_video(path: Path, codec="mp4v"):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*codec),10,(32,24))
    if not writer.isOpened(): pytest.skip(f"codec {codec} unavailable")
    for index in range(4):
        frame=np.zeros((24,32,3),np.uint8);cv2.rectangle(frame,(index,5),(index+4,10),(255,255,255),-1);writer.write(frame)
    writer.release()


def test_streaming_video_jsonl_stride_and_metadata(tmp_path):
    source=tmp_path/"source.mp4"; output=tmp_path/"planned.mp4"; metadata=tmp_path/"planned.jsonl";_make_video(source)
    summary=run_video_plan(FakePipeline(),source,output,metadata_path=metadata,options=VideoPlanOptions(frame_stride=2,max_frames=2,show_potential=False,show_detections=False))
    assert summary["processed_frame_count"]==2 and summary["output_fps"]==pytest.approx(5)
    capture=cv2.VideoCapture(str(output)); assert capture.isOpened(); assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==2; assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))==(32,24);capture.release()
    rows=[json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines()]
    assert len(rows)==2 and rows[0]["source_frame_index"]==0 and rows[1]["source_frame_index"]==2
    assert rows[0]["trajectory_source"] in {"current","stabilized"}
    assert {"start_image_xy","goal_grid_xy","processing_time_ms","final_trajectory_heading"} <= rows[0].keys()


def test_video_options_validation(tmp_path):
    with pytest.raises(ValueError): VideoPlanOptions(frame_stride=0).validate()
    with pytest.raises(ValueError): VideoPlanOptions(potential_alpha=2).validate()
    source=tmp_path/"same.mp4";_make_video(source)
    with pytest.raises(ValueError,match="differ"): run_video_plan(FakePipeline(),source,source)


def test_frame_exception_does_not_stop_stream(tmp_path):
    class Flaky(FakePipeline):
        def process_frame(self, frame, index, timestamp, *args, **kwargs):
            if index == 1: raise RuntimeError("synthetic failure")
            return super().process_frame(frame,index,timestamp,*args,**kwargs)
    source=tmp_path/"source.mp4";output=tmp_path/"out.mp4";metadata=tmp_path/"out.jsonl";_make_video(source)
    summary=run_video_plan(Flaky(),source,output,metadata_path=metadata,options=VideoPlanOptions(max_frames=3,show_potential=False))
    assert summary["processed_frame_count"]==3 and summary["failed_frame_count"]==1
    rows=[json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines()]
    assert rows[1]["error_type"]=="RuntimeError" and rows[2]["planning_status"]=="success"
