# Active Context

## Current Focus

Potential Field Gradient Path Planner is implemented; the next focus is Local Minimum Handling.

## Current Scope Boundary

Object detection, YOLO instance segmentation, SegFormer scene segmentation, monocular metric depth, camera intrinsics, 3D back-projection, camera-centric semantic BEV, semantic occupancy/cost grids, goal-conditioned potential fields, and unified perception are implemented. Planning, temporal smoothing, and fine-tuning are not implemented.

## Next Work Order

1. Add Local Minimum Handling.
2. Add an A* hybrid planner or trajectory smoothing later.

## Implementation Notes

Detection results use `bbox_xyxy` in `[x1, y1, x2, y2]` format.
Instance-segmentation results use `segments` with `bbox_xyxy`, `mask_area`, and `mask_path`.
Fusion uses same-class bbox IoU with a default threshold of 0.5 and preserves unmatched results.
`datasets/raw/` is ignored by Git, so sample videos are kept local.
Depth source arrays are float32 meter-valued NPY files; PNG and color maps are derived artifacts.
Geometry source points are camera-coordinate XYZ in meters, generated from in-memory metric depth and explicit intrinsics.

## Execution Environment

- Repository: `C:\Users\김휘수\Documents\yolo`
- Branch: `main`
- Python command: `py -3.14`
- Git executable: `C:\Users\김휘수\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe`
- Full validation: `py -3.14 -m compileall scripts src tests -q`, CLI `--help` smoke test, then `py -3.14 -m pytest -q`
- Do not use the Windows `python` alias.
- Exclude real model inference and video execution from default validation.

## Current Baseline

- Baseline: current `main` after the Goal-conditioned Potential Field commit
- Tests: pending final validation
- Remote target: `origin/main`
- Completed: Potential Field Gradient Path Planner
- Next: Local Minimum Handling
- After: A* Hybrid Planner / Trajectory Smoothing
