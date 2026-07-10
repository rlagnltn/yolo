# Active Context

## Current Focus

Semantic Occupancy / Cost Grid is implemented; the next focus is Goal-conditioned Potential Field.

## Current Scope Boundary

Object detection, YOLO instance segmentation, SegFormer scene segmentation, monocular metric depth, camera intrinsics, 3D back-projection, camera-centric semantic BEV, semantic occupancy/cost grids, and unified perception are implemented. Potential fields, planning, temporal smoothing, and fine-tuning are not implemented.

## Next Work Order

1. Build Goal-conditioned Potential Field.
2. Add path planning stages later.

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

- Baseline: current `main` after the Semantic Occupancy / Cost Grid commit
- Tests: `87 passed`
- Remote target: `origin/main`
- Completed: Semantic Occupancy / Cost Grid
- Next: Goal-conditioned Potential Field
- After: Path Planning
