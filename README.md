# Vision Potential Field

YOLO-based research scaffold for detecting objects in driving videos, then extending the pipeline toward semantic mapping, BEV transformation, potential field generation, and path planning.

## Research Goal

The final target is a vision pipeline that accepts vehicle driving video, runs object detection, semantic segmentation, depth estimation, BEV transformation, semantic potential field generation, and path planning toward a safe target.

This first implementation focuses on a working YOLO object-detection pipeline:

- Read an image or video input.
- Run YOLO object detection with `ultralytics`.
- Save frame-level detections as JSON.
- Optionally save bounding-box visualizations.

Segmentation, depth, BEV, mapping, potential field, and planner modules are included as extension points but are not implemented yet.

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

The default config uses `yolov8n.pt`. Ultralytics will download the model weights on first use if they are not already available.

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

## Roadmap

1. YOLO object detection for driving video.
2. Semantic segmentation module.
3. Depth estimation module.
4. BEV transformation and semantic map generation.
5. Semantic potential field generation.
6. Path planning and local-minima handling.
