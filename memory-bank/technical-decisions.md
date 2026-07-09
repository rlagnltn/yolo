# Technical Decisions

## Language

Python is used for the research pipeline because the computer-vision ecosystem is strongest there.

## Detection Model

The first detector uses the `ultralytics` YOLO package with `yolov8n.pt` as the default lightweight model. The code accepts a different model name or weight path from the CLI.

## Video Processing

OpenCV handles image and video IO.

## Output Format

Detections are stored as JSON for readability and easy downstream processing.

## Dependency Policy

Keep dependencies lightweight in the first phase:

- `ultralytics`
- `opencv-python`
- `numpy`
- `PyYAML`
- `tqdm`

## Future Dataset Candidates

- BDD100K
- KITTI
- nuScenes
