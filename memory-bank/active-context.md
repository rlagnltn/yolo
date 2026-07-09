# Active Context

## Current Focus

Build a clean research scaffold and implement the first runnable YOLO detection pipeline.

## Current Scope Boundary

Only object detection is implemented. Segmentation, depth estimation, BEV transformation, potential field generation, and path planning are intentionally left as extension points.

## Next Work Order

1. Add a sample driving video or dataset loader.
2. Run detection and inspect `outputs/detections/detections.json`.
3. Add semantic segmentation.
4. Add depth estimation.
5. Design BEV and potential field data formats.

## Implementation Notes

Detection results use bounding boxes in `[x1, y1, x2, y2]` format.
