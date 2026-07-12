"""Temporal planner evaluation for unified perception JSON results."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any


def evaluate_planner_frames(result: dict[str, Any]) -> dict[str, Any]:
    frames = result.get("frames", [])
    fps = float(result.get("metadata", {}).get("fps") or 0.0)
    if fps <= 0:
        raise ValueError("Planner evaluation requires positive metadata.fps.")
    available: list[bool] = []
    raw_success: list[bool] = []
    planning_attempts = 0
    reasons: Counter[str] = Counter()
    safety_violations = 0
    reused_count = short_horizon_count = 0
    last_valid_frame = None
    for frame in frames:
        planner = frame.get("planner") or {}
        source = planner.get("path_source")
        if source is None and planner.get("status") == "success":
            source = "new"
        is_available = bool(planner.get("reached_goal")) and source in {"new", "reused"}
        is_raw = bool(planner.get("raw_planner_success", planner.get("status") == "success"))
        attempted = bool(planner.get("planning_attempted", source != "reused" and planner.get("reason_code") != "PLANNING_INTERVAL_SKIP"))
        planning_attempts += int(attempted)
        if planner.get("horizon_status") == "short_horizon":
            short_horizon_count += 1
            is_raw = False
        if source == "reused":
            reused_count += 1
        if is_available and planner.get("collision_validated") is False:
            safety_violations += 1
            is_available = False
        if is_available:
            last_valid_frame = int(frame.get("frame_index", len(available)))
        else:
            reasons[str(planner.get("reason_code", "MISSING_PLANNER_RESULT"))] += 1
        available.append(is_available)
        raw_success.append(is_raw and attempted)
    runs = _false_runs(available)
    p95_frames = _nearest_rank(runs, .95)
    max_frames = max(runs, default=0)
    threshold_frames = max(1, int(math.floor(.5 * fps)))
    frame_count = len(frames)
    return {
        "frame_count": frame_count, "fps": fps,
        "planning_attempt_count": planning_attempts,
        "raw_planner_success_count": sum(raw_success),
        "raw_planner_success_rate": _rate(sum(raw_success), planning_attempts),
        "validated_path_available_count": sum(available),
        "validated_path_availability_rate": _rate(sum(available), frame_count),
        "reused_path_frame_count": reused_count, "short_horizon_frame_count": short_horizon_count,
        "failure_run_count": len(runs), "failure_run_p95_frames": p95_frames,
        "failure_run_p95_sec": p95_frames / fps, "maximum_path_gap_frames": max_frames,
        "maximum_path_gap_sec": max_frames / fps,
        "outages_over_0_50_sec": sum(run > threshold_frames for run in runs),
        "reported_safety_violation_count": safety_violations,
        "last_valid_path_frame": last_valid_frame, "failure_reason_distribution": dict(reasons),
        "criteria": {
            "raw_success_rate_at_least_0_50": _rate(sum(raw_success), planning_attempts) >= .5,
            "validated_availability_at_least_0_95": _rate(sum(available), frame_count) >= .95,
            "failure_run_p95_at_most_0_30_sec": p95_frames / fps <= .3,
            "maximum_gap_at_most_0_50_sec": max_frames / fps <= .5,
            "no_outage_over_0_50_sec": not any(run > threshold_frames for run in runs),
            "reported_safety_violations_zero": safety_violations == 0,
        },
    }


def _false_runs(values: list[bool]) -> list[int]:
    runs: list[int] = []
    current = 0
    for value in values:
        if value:
            if current:
                runs.append(current)
                current = 0
        else:
            current += 1
    if current:
        runs.append(current)
    return runs


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0
