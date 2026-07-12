# Technical Decisions

## Language

Python is used for the research pipeline because the computer-vision ecosystem is strongest there.

## Detection Model

The first detector uses the `ultralytics` YOLO package with `yolov8n.pt` as the default lightweight model. The code accepts a different model name or weight path from the CLI.

## Segmentation Model

The first segmentation implementation uses the `ultralytics` YOLO instance-segmentation model with `yolov8n-seg.pt` as the default lightweight model. It produces masks for supported object instances and does not replace full-scene semantic segmentation.

Future replacements may include SegFormer, Mask2Former, DeepLabV3+, or BEV-aware models when segmentation quality and model comparison become the active task.

## Scene Semantic Segmentation

The scene model is `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`. SegFormer-B0 was selected for a lightweight first validation. Its model-config `id2label` is the source of truth. The scene branch is disabled by default in unified perception for backward compatibility and to avoid unrequested model downloads.

Class maps are saved as single-channel PNGs, while JSON stores only paths and statistics. Only `road` is vehicle-drivable by default; `sidewalk` is not. Cityscapes domain gap must be considered for Korean roads and adverse conditions. Scene labels remain in the 2D image plane; metric depth is available separately, while camera geometry and 3D projection are the next stage.

## Video Processing

For the current 29.97 FPS research sample, the planner target cadence is 10 Hz (one attempt every three video frames). A last valid path may bridge at most five failed planning attempts, or approximately 0.50 seconds, only after collision validation against the current occupancy grid. Longer gaps are reported as `path_unavailable`. The full evaluation gates and measured baseline are recorded in `memory.md`.

Raw planner success is measured per actual planning attempt, not per video frame. Per-frame continuity is measured separately as validated path availability. A 3 m or 4 m `short_horizon` path can provide temporary validated availability but is excluded from the full 5 m raw-success numerator. Auto-cell connectivity and both planners use one shared FREE-neighbor policy.

FREE components are labeled with OpenCV instead of repeated Python BFS. Four-connected labeling is used whenever diagonal corner cutting is prohibited; this is equivalent to legal 8-neighbor reachability because every permitted diagonal has both orthogonal FREE cells. Chunk runs preserve source frame indices but reset temporal occupancy and path memory at the chunk boundary.

Original-video path overlays use the source `pixels_uv` saved beside each camera-coordinate point cloud. For a planner grid cell, the renderer chooses the nearest camera point in that same BEV cell; no camera height or pitch is guessed. Only newly successful paths are rendered, and discontinuities in observed path cells deliberately break the displayed line.

## Monocular Metric Depth

The configured default `depth-anything/Depth-Anything-V2-Metric-VKITTI-Small` is the original checkpoint repository, so the loader resolves it to the official Transformers-compatible `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`. Its config explicitly declares metric depth and `max_depth: 80`; JSON records the requested and loaded names.

The authoritative output is float32 NPY in meters. uint16 PNG uses an explicit default scale of 1000, reserves zero for invalid pixels, and clips overflow. Percentile normalization is visualization-only. Raw arrays are not embedded in JSON. Depth is disabled by default in unified perception and shares its single-pass frame iterator when enabled.

OpenCV handles image and video IO.

The unified pipeline iterates over the video once and sends each frame to both models. Each model is initialized once before processing begins.

## Perception Fusion

Detection and instance-segmentation records are matched by compatible class and bbox IoU, using a default threshold of 0.5. Greedy highest-IoU matching is one-to-one. Unmatched records are retained as `detection_only` and `segmentation_only`.

## Output Format

Detections are stored as JSON for readability and easy downstream processing. The frame-level schema uses `objects` with `bbox_xyxy` coordinates.

Segmentation records are also stored as JSON, but raw mask arrays are not embedded because they can become very large. Per-object binary masks are saved as image files, and JSON stores `mask_path`, `mask_area`, class, confidence, and bounding-box summary fields.

## Data and Output Versioning

`datasets/raw/` and `outputs/` stay out of Git. Sample videos are provided manually by the user, usually as `datasets/raw/sample.mp4`, or supplied with `--input`. Generated segmentation masks and overlays are also output artifacts and are not committed.

Official dataset download automation is intentionally deferred. BDD100K, KITTI, and nuScenes may require account setup, license acceptance, or large downloads, so those workflows will be added later when dataset integration becomes the active task.

## Dependency Policy

Keep dependencies lightweight in the first phase:

- `ultralytics`
- `opencv-python`
- `numpy`
- `PyYAML`
- `tqdm`
- `pytest`

## Future Dataset Candidates

- BDD100K
- KITTI
- nuScenes
