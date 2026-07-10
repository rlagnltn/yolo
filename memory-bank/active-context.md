# Active Context

## Current Focus

Monocular metric depth is implemented; the next focus is camera geometry and 3D projection.

## Current Scope Boundary

Object detection, YOLO instance segmentation, SegFormer scene segmentation, monocular metric depth, and unified perception are implemented. Camera intrinsics, 3D projection, BEV, occupancy, potential fields, planning, temporal smoothing, and fine-tuning are not implemented.

## Next Work Order

1. Validate scene segmentation, depth, and unified perception with a real driving video.
2. Add camera intrinsics and 3D back-projection.
3. Build a semantic BEV grid.

## Implementation Notes

Detection results use `bbox_xyxy` in `[x1, y1, x2, y2]` format.
Instance-segmentation results use `segments` with `bbox_xyxy`, `mask_area`, and `mask_path`.
Fusion uses same-class bbox IoU with a default threshold of 0.5 and preserves unmatched results.
`datasets/raw/` is ignored by Git, so sample videos are kept local.
Depth source arrays are float32 meter-valued NPY files; PNG and color maps are derived artifacts.
