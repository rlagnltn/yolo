# Active Context

## Current Focus

Vehicle driving-video based semantic segmentation.

## Current Scope Boundary

Object detection and YOLO segmentation are implemented. Depth Estimation, BEV transformation, Potential Field generation, and Path Planning are intentionally not implemented in this step.

## Next Work Order

1. User places a driving video at `datasets/raw/sample.mp4` or passes another path with `--input`.
2. Run YOLO segmentation and inspect `outputs/segmentations/segmentations.json`.
3. Inspect binary masks in `outputs/segmentations/masks/`.
4. Inspect overlay frames in `outputs/segmentations/visualizations/` when visualization saving is enabled.
5. Next stage: integrate detection + segmentation results on a shared frame basis.
6. Following stage: prepare BEV transformation inputs.

## Implementation Notes

Detection results use `bbox_xyxy` in `[x1, y1, x2, y2]` format.
Segmentation results use `segments` with `bbox_xyxy`, `mask_area`, and `mask_path`.
`datasets/raw/` is ignored by Git, so sample videos are kept local.
