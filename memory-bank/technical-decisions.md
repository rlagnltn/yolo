# Technical Decisions

## Language

Python is used for the research pipeline because the computer-vision ecosystem is strongest there.

## Detection Model

The first detector uses the `ultralytics` YOLO package with `yolov8n.pt` as the default lightweight model. The code accepts a different model name or weight path from the CLI.

## Video Processing

OpenCV handles image and video IO.

## Output Format

Detections are stored as JSON for readability and easy downstream processing. The frame-level schema uses `objects` with `bbox_xyxy` coordinates.

## Data and Output Versioning

`datasets/raw/` and `outputs/` stay out of Git. Sample videos are provided manually by the user, usually as `datasets/raw/sample.mp4`, or supplied with `--input`.

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
