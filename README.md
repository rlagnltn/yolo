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

## Run Segmentation

Detection returns object bounding boxes. Segmentation returns object/region masks, plus bounding boxes and class labels when the model provides them.

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

The current segmentation implementation uses YOLO segmentation for quick validation. The code is structured so the model wrapper can later be replaced with SegFormer, Mask2Former, DeepLabV3+, or another semantic segmentation backend. BEV transformation, potential field generation, and path planning are still intentionally out of scope at this stage.

## Roadmap

1. YOLO object detection for driving video.
2. YOLO semantic segmentation module.
3. Depth estimation module.
4. BEV transformation and semantic map generation.
5. Semantic potential field generation.
6. Path planning and local-minima handling.
