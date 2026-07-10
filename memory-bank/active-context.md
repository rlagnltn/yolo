# Active Context

## Current Focus

Camera-centric Semantic BEV Grid is implemented; the next focus is Semantic Occupancy / Cost Grid.

## Current Scope Boundary

Object detection, YOLO instance segmentation, SegFormer scene segmentation, monocular metric depth, camera intrinsics, 3D back-projection, camera-centric semantic BEV, and unified perception are implemented. Occupancy, potential fields, planning, temporal smoothing, and fine-tuning are not implemented.

## Next Work Order

1. Build Semantic Occupancy / Cost Grid.
2. Add Potential Field.
3. Add planning stages later.

## Implementation Notes

Detection results use `bbox_xyxy` in `[x1, y1, x2, y2]` format.
Instance-segmentation results use `segments` with `bbox_xyxy`, `mask_area`, and `mask_path`.
Fusion uses same-class bbox IoU with a default threshold of 0.5 and preserves unmatched results.
`datasets/raw/` is ignored by Git, so sample videos are kept local.
Depth source arrays are float32 meter-valued NPY files; PNG and color maps are derived artifacts.
Geometry source points are camera-coordinate XYZ in meters, generated from in-memory metric depth and explicit intrinsics.
