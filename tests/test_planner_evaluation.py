from src.planner import evaluate_planner_frames


def test_evaluation_distinguishes_raw_reused_and_short_horizon_paths():
    planners = [
        {"status":"success","reached_goal":True,"path_source":"new","raw_planner_success":True,"collision_validated":True,"horizon_status":"full_horizon"},
        {"status":"reused","reached_goal":True,"path_source":"reused","raw_planner_success":False,"collision_validated":True},
        {"status":"success","reached_goal":True,"path_source":"new","raw_planner_success":True,"collision_validated":True,"horizon_status":"short_horizon"},
        {"status":"not_run","reached_goal":False,"path_source":"unavailable","reason_code":"NO_FORWARD_GOAL"},
    ]
    result={"metadata":{"fps":10},"frames":[{"frame_index":i,"planner":p} for i,p in enumerate(planners)]}
    report=evaluate_planner_frames(result)
    assert report["raw_planner_success_count"] == 1
    assert report["planning_attempt_count"] == 3
    assert report["validated_path_available_count"] == 3
    assert report["reused_path_frame_count"] == 1
    assert report["short_horizon_frame_count"] == 1
    assert report["failure_reason_distribution"] == {"NO_FORWARD_GOAL":1}


def test_evaluation_detects_temporal_gap_and_unvalidated_reuse():
    planners=[
        {"status":"success","reached_goal":True,"path_source":"new","collision_validated":True},
        {"status":"reused","reached_goal":True,"path_source":"reused","collision_validated":False},
    ] + [{"status":"not_run","reason_code":"NO_FORWARD_GOAL"} for _ in range(5)]
    result={"metadata":{"fps":10},"frames":[{"frame_index":i,"planner":p} for i,p in enumerate(planners)]}
    report=evaluate_planner_frames(result)
    assert report["reported_safety_violation_count"] == 1
    assert report["maximum_path_gap_sec"] == .6
    assert report["outages_over_0_50_sec"] == 1
