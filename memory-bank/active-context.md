# Active Context

## Current Focus

Detection + YOLO instance-segmentation result fusion.

## Current Scope Boundary

Object detection, YOLO instance segmentation, and their unified perception pipeline are implemented. Scene semantic segmentation, depth estimation, BEV transformation, potential-field generation, and path planning are intentionally not implemented in this step.

## Next Work Order

1. Validate unified perception with a real driving video.
2. Next: add scene semantic segmentation for road, sidewalk, lanes, buildings, sky, and vegetation.
3. Then: implement depth estimation and BEV coordinate transformation.

## Implementation Notes

Detection results use `bbox_xyxy` in `[x1, y1, x2, y2]` format.
Instance-segmentation results use `segments` with `bbox_xyxy`, `mask_area`, and `mask_path`.
Fusion uses same-class bbox IoU with a default threshold of 0.5 and preserves unmatched results.
`datasets/raw/` is ignored by Git, so sample videos are kept local.
