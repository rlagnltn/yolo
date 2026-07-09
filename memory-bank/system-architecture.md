# System Architecture

## Full Research Pipeline

1. Video input.
2. Object detection.
3. Semantic segmentation.
4. Depth estimation.
5. BEV transformation.
6. Semantic map construction.
7. Potential field generation.
8. Path planning.

## Current Pipeline

```text
Driving Video
  -> Frame Extraction
  -> YOLO Object Detection
  -> YOLO Segmentation
  -> Detection/Segmentation JSON
  -> Visualization Output
```

## Module Responsibilities

- `src/detection`: YOLO model loading and detection result formatting.
- `src/segmentation`: YOLO segmentation model loading and segment result formatting.
- `src/depth`: future depth estimation module.
- `src/bev`: future bird's-eye-view transformation module.
- `src/mapping`: future semantic map construction module.
- `src/potential`: future potential field generation module.
- `src/planner`: future path planning module.
- `src/utils`: reusable IO, video, mask, and visualization helpers.
