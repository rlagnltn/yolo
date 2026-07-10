# Vision Potential Field

YOLO-based research scaffold for detecting objects in driving videos, then extending the pipeline toward semantic mapping, BEV transformation, potential field generation, and path planning.

## Research Goal

The final target is a vision pipeline that accepts vehicle driving video, runs object detection, semantic segmentation, depth estimation, BEV transformation, semantic potential field generation, and path planning toward a safe target.

The current implementation focuses on a working YOLO object-detection pipeline and a YOLO segmentation pipeline:

- Read an image or video input.
- Run YOLO object detection with `ultralytics`.
- Run YOLO segmentation with `ultralytics`.
- Save frame-level detections as JSON.
- Save frame-level segmentation records as JSON.
- Optionally save bounding-box visualizations, segmentation masks, and overlay visualizations.

Depth, BEV, mapping, potential field, and planner modules are included as extension points but are not implemented yet.

## Project Structure

```text
yolo/
  configs/
  datasets/
    annotations/
    processed/
    raw/
  memory-bank/
  models/
    depth/
    segmentation/
    yolo/
  notebooks/
  outputs/
    detections/
    frames/
    segmentations/
      masks/
      visualizations/
    visualizations/
  scripts/
  src/
    bev/
    depth/
    detection/
    mapping/
    planner/
    potential/
    segmentation/
    utils/
  tests/
```

## Install

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

The detection config uses `yolov8n.pt`; the segmentation config uses `yolov8n-seg.pt`. Ultralytics will download model weights on first use if they are not already available.

## Sample Input Video

`datasets/raw/` is ignored by Git, so real driving videos are not committed to this repository.

Before running detection, place a driving video at:

```text
datasets/raw/sample.mp4
```

You can also pass any other image or video path with `--input`.

Dataset candidates for future experiments:

- BDD100K
- KITTI
- nuScenes

This project does not automate official dataset downloads yet. Download, license acceptance, and local dataset placement are handled manually at this stage.

## Run Detection

```bash
pip install -r requirements.txt
python scripts/run_detection.py --input datasets/raw/sample.mp4 --save-vis
python scripts/run_detection.py --input datasets/raw/sample.mp4 --save-vis --max-frames 100
```

Useful options:

```bash
python scripts/run_detection.py \
  --input datasets/raw/sample.mp4 \
  --output outputs/detections/detections.json \
  --visualization-dir outputs/visualizations \
  --model yolov8n.pt \
  --confidence 0.25 \
  --max-frames 100 \
  --save-vis
```

For images, pass an image path as `--input`; for videos, pass a video path.

## Outputs

- `outputs/detections/detections.json`: frame-level detection records.
- `outputs/visualizations/`: annotated images or video frames when `--save-vis` is enabled.

Detection JSON follows this structure:

```json
{
  "input": "datasets/raw/sample.mp4",
  "model": "yolov8n.pt",
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "width": 1280,
      "height": 720,
      "objects": [
        {
          "class_id": 2,
          "class_name": "car",
          "confidence": 0.91,
          "bbox_xyxy": [100.0, 200.0, 300.0, 400.0]
        }
      ]
    }
  ]
}
```

If `datasets/raw/sample.mp4` does not exist, the CLI prints a friendly error and asks you to either place the file there or pass another path with `--input`.

## Run Instance Segmentation

Detection returns object bounding boxes. The current YOLO instance-segmentation model returns a separate mask, bounding box, and class label for each supported object instance.

```bash
pip install -r requirements.txt
python scripts/run_segmentation.py --input datasets/raw/sample.mp4 --save-vis
python scripts/run_segmentation.py --input datasets/raw/sample.mp4 --save-vis --max-frames 100
```

Useful options:

```bash
python scripts/run_segmentation.py \
  --input datasets/raw/sample.mp4 \
  --output outputs/segmentations/segmentations.json \
  --mask-dir outputs/segmentations/masks \
  --visualization-dir outputs/segmentations/visualizations \
  --model yolov8n-seg.pt \
  --confidence 0.25 \
  --device auto \
  --max-frames 100 \
  --save-vis
```

Segmentation outputs:

- `outputs/segmentations/segmentations.json`: frame-level segmentation records.
- `outputs/segmentations/masks/`: per-object binary mask PNG files.
- `outputs/segmentations/visualizations/`: overlay frames when `--save-vis` or config visualization saving is enabled.

Segmentation JSON follows this structure:

```json
{
  "input": "datasets/raw/sample.mp4",
  "model": "yolov8n-seg.pt",
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "width": 1280,
      "height": 720,
      "segments": [
        {
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.91,
          "bbox_xyxy": [100.0, 200.0, 300.0, 500.0],
          "mask_area": 12450,
          "mask_path": "outputs/segmentations/masks/frame_000000_obj_000.png"
        }
      ]
    }
  ]
}
```

`yolov8n-seg.pt` is an instance-segmentation model, not a complete scene semantic-segmentation model. It can mask supported object classes such as cars, people, bicycles, motorcycles, buses, and trucks. It does not provide complete road, sidewalk, lane, building, sky, or vegetation regions. A separate scene semantic-segmentation backend is planned for the next stage. BEV transformation, potential field generation, and path planning remain out of scope here.

## Run Unified Perception

The unified pipeline reads each video frame once, sends that shared frame to the already-loaded detection and instance-segmentation models, and fuses their results using class compatibility and bounding-box IoU. Each segment can be matched at most once. Unmatched results are preserved as `detection_only` or `segmentation_only`.

PowerShell one-line command:

```powershell
python scripts/run_perception.py --input datasets/raw/sample.mp4 --save-vis --max-frames 100
```

Supported options include `--input`, `--config`, `--output`, `--save-vis`, `--save-masks`, `--max-frames`, `--device`, `--iou-threshold`, and `--continue-on-error`. Boolean options also accept the `--no-...` form, such as `--no-save-vis`.

The default output layout is:

```text
outputs/perception/
  perception.json
  masks/
    frame_000000_seg_000.png
  visualizations/
    frame_000000.png
```

The top-level JSON contains video metadata and frame records:

```json
{
  "metadata": {
    "input": "datasets/raw/sample.mp4",
    "detection_model": "yolov8n.pt",
    "segmentation_model": "yolov8n-seg.pt",
    "frame_count": 100,
    "processed_frame_count": 100,
    "fps": 30.0,
    "width": 1280,
    "height": 720
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "width": 1280,
      "height": 720,
      "detections": [],
      "segments": [],
      "fused_objects": [],
      "errors": []
    }
  ]
}
```

The default fusion threshold is `0.5`. A detection and segment must share a class and meet the IoU threshold to become `matched`; otherwise both records remain available. Raw mask arrays are never embedded in JSON.

## Roadmap

1. YOLO object detection for driving video.
2. YOLO instance segmentation module.
3. Unified detection + instance-segmentation perception pipeline.
4. Scene semantic segmentation for road, sidewalk, lanes, buildings, sky, and vegetation.
5. Depth estimation module.
6. BEV transformation and semantic map generation.
7. Semantic potential field generation.
8. Path planning and local-minima handling.
