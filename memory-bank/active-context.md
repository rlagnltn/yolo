# Active Context

## Current Focus

Sample driving-video based YOLO detection validation.

## Current Scope Boundary

Only object detection is implemented. Semantic Segmentation, Depth Estimation, BEV transformation, Potential Field generation, and Path Planning are intentionally not implemented in this step.

## Next Work Order

1. User places a driving video at `datasets/raw/sample.mp4` or passes another path with `--input`.
2. Run YOLO detection and inspect `outputs/detections/detections.json`.
3. Inspect annotated frames in `outputs/visualizations/` when `--save-vis` is used.
4. Add the Semantic Segmentation module in the next stage.

## Implementation Notes

Detection results use `bbox_xyxy` in `[x1, y1, x2, y2]` format.
`datasets/raw/` is ignored by Git, so sample videos are kept local.
