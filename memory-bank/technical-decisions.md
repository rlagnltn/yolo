# Technical Decisions

## Language

Python is used for the research pipeline because the computer-vision ecosystem is strongest there.

## Detection Model

The first detector uses the `ultralytics` YOLO package with `yolov8n.pt` as the default lightweight model. The code accepts a different model name or weight path from the CLI.

## Segmentation Model

The first segmentation implementation uses the `ultralytics` YOLO segmentation model with `yolov8n-seg.pt` as the default lightweight model. This keeps the environment close to the detection pipeline and enables quick mask output validation before introducing heavier segmentation frameworks.

Future replacements may include SegFormer, Mask2Former, DeepLabV3+, or BEV-aware models when segmentation quality and model comparison become the active task.

## Video Processing

OpenCV handles image and video IO.

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
