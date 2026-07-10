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
  -> Single Frame Iterator
       -> YOLO Object Detection
       -> YOLO Instance Segmentation
       -> SegFormer Scene Semantic Segmentation
  -> IoU/Class-based Fusion
  -> Scene Class Map
  -> Drivable / Non-drivable Masks
  -> Unified Perception JSON
  -> Unified Visualization
```

## Module Responsibilities

- `src/detection`: YOLO model loading and detection result formatting.
- `src/segmentation`: YOLO segmentation model loading and segment result formatting.
- `src/perception`: shared-frame orchestration, schemas, and class/IoU fusion.
- `src/scene_segmentation`: SegFormer inference, Cityscapes mapping, postprocessing, and scene artifacts.
- `src/depth`: future depth estimation module.
- `src/bev`: future bird's-eye-view transformation module.
- `src/mapping`: future semantic map construction module.
- `src/potential`: future potential field generation module.
- `src/planner`: future path planning module.
- `src/utils`: reusable IO, video, mask, and visualization helpers.
