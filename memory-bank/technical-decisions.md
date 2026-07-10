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

Class maps are saved as single-channel PNGs, while JSON stores only paths and statistics. Only `road` is vehicle-drivable by default; `sidewalk` is not. Cityscapes domain gap must be considered for Korean roads and adverse conditions. Current outputs remain in the 2D image plane and contain no physical distance. The next stage will add monocular depth estimation.

## Video Processing

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
