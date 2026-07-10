# Active Context

## Current Focus

SegFormer-based scene semantic segmentation.

## Current Scope Boundary

Object detection, YOLO instance segmentation, unified perception, and SegFormer scene semantic segmentation are implemented. Depth estimation, BEV transformation, potential-field generation, and path planning are intentionally not implemented in this step.

## Next Work Order

1. Validate scene segmentation and unified perception with a real driving video.
2. Next: implement monocular depth estimation.
3. Then: combine semantics and depth for 3D projection and BEV generation.

## Implementation Notes

Detection results use `bbox_xyxy` in `[x1, y1, x2, y2]` format.
Instance-segmentation results use `segments` with `bbox_xyxy`, `mask_area`, and `mask_path`.
Fusion uses same-class bbox IoU with a default threshold of 0.5 and preserves unmatched results.
`datasets/raw/` is ignored by Git, so sample videos are kept local.
