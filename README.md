# Vision Potential Field

YOLO-based research scaffold for detecting objects in driving videos, then extending the pipeline toward semantic mapping, BEV transformation, potential field generation, and path planning.

## Research Goal

The final target is a vision pipeline that accepts vehicle driving video, runs object detection, semantic segmentation, depth estimation, BEV transformation, semantic potential field generation, and path planning toward a safe target.

The current implementation includes object, scene, and metric-depth perception:

- Read an image or video input.
- Run YOLO object detection with `ultralytics`.
- Run YOLO segmentation with `ultralytics`.
- Save frame-level detections as JSON.
- Save frame-level segmentation records as JSON.
- Optionally save bounding-box visualizations, segmentation masks, and overlay visualizations.
- Estimate per-pixel outdoor metric depth and save lossless NPY, optional uint16 PNG, color maps, and overlays.
- Combine calibrated camera intrinsics and metric depth into camera-coordinate XYZ point clouds.
- Optionally attach same-frame scene class IDs to generated 3D points.

BEV, mapping, potential field, and deterministic gradient-descent planning are available as optional pipeline stages.

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

`yolov8n-seg.pt` is an instance-segmentation model, not a complete scene semantic-segmentation model. It can mask supported object classes such as cars and people. The separately implemented SegFormer branch supplies full-scene labels. BEV transformation, potential-field generation, and path planning remain out of scope here.

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

## Scene Semantic Segmentation

Instance segmentation separates supported object instances such as cars and people. Scene semantic segmentation instead assigns a class to every image pixel, including broad surfaces and background regions. This project uses `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`, a lightweight SegFormer-B0 model fine-tuned for 19 Cityscapes classes:

```text
road, sidewalk, building, wall, fence, pole, traffic light, traffic sign,
vegetation, terrain, sky, person, rider, car, truck, bus, train,
motorcycle, bicycle
```

Install dependencies and run the standalone pipeline in PowerShell:

```powershell
pip install -r requirements.txt
python scripts/run_scene_segmentation.py --input datasets/raw/sample.mp4 --save-vis --save-regions --max-frames 30
```

Enable it inside the unified perception pipeline with:

```powershell
python scripts/run_perception.py --input datasets/raw/sample.mp4 --enable-scene-segmentation --save-vis --max-frames 30
```

Scene segmentation is disabled by default in `configs/perception.yaml`, preserving the previous pipeline behavior and avoiding an automatic model download. Disabled unified-perception frames contain `"scene_segmentation": null`. Its outputs are distinct:

- Class map: single-channel PNG whose pixel values are model class IDs; this is the machine-readable result.
- Color map: BGR palette visualization of the class IDs.
- Overlay: color map blended over the original frame.
- Region masks: binary 0/255 PNGs for drivable and non-drivable pixels.
- JSON: paths and pixel statistics only; raw class-map arrays are not embedded.

The default driving-region policy treats only `road` as drivable. `sidewalk` is a pedestrian surface, not a vehicle-driving area. Buildings, walls, fences, poles, signs, vegetation, and terrain are non-drivable; dynamic object classes are also conservatively non-drivable. Sky and ungrouped pixels remain unknown/background rather than being silently treated as road.

Important limitations:

- Cityscapes differs from Korean roads, nighttime scenes, rain, and unusual road layouts, so domain gap can reduce accuracy.
- Results are 2D image-plane labels, not physical distances or BEV coordinates.
- A drivable mask alone is not a guaranteed safe path and does not estimate lane center or travel direction.
- No model training or fine-tuning is performed in this stage.
- These scene labels can now be paired with the implemented monocular metric-depth branch; camera geometry is still required before projection into BEV.

## Monocular Metric Depth Estimation

Relative depth preserves near/far ordering but has no guaranteed physical scale. Metric depth predicts physical distance; this stage uses meters. The configured name `depth-anything/Depth-Anything-V2-Metric-VKITTI-Small` resolves to the official Transformers-compatible `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`. Its config declares `depth_estimation_type: metric` and an outdoor maximum depth of 80 meters. Units are taken from that model contract, never guessed from tensor values.

Standalone and unified runs:

```powershell
python scripts/run_depth.py --input datasets/raw/sample.mp4 --save-raw --save-depth-png --save-color-maps --save-vis --max-frames 30
python scripts/run_perception.py --input datasets/raw/sample.mp4 --enable-scene-segmentation --enable-depth --save-vis --max-frames 10
```

Depth is disabled by default in unified perception, preserving prior behavior with `"depth": null`. When enabled, one estimator receives the same already-read frame as the other branches. With scene segmentation enabled, shapes are validated before per-class depth summaries are computed.

- `raw/frame_XXXXXX.npy`: authoritative lossless `float32` metric depth in meters.
- `depth_maps/frame_XXXXXX.png`: optional `uint16`, where `value = meters × png_scale`; default scale 1000, zero invalid, overflow clipped.
- `color_maps/frame_XXXXXX.png`: percentile-normalized visualization only.
- `visualizations/frame_XXXXXX.png`: color depth blended over the frame.
- JSON: paths, scale, units, validity statistics, percentiles, and optional scene-class summaries; never the full array.

Monocular metric depth still has scale error and a domain gap on Korean roads, night, rain, glare, and cameras unlike Virtual KITTI. It does not provide camera intrinsics, 3D coordinates, safe clearance, or BEV by itself. Those require the next Camera Geometry and 3D Projection stage.

## Camera Geometry and 3D Back-projection

Back-projection requires real camera calibration. The project does not guess focal length or principal point. Put calibrated values in `configs/camera.yaml`:

```yaml
camera:
  fx: 900.0
  fy: 900.0
  cx: 640.0
  cy: 360.0
  width: 1280
  height: 720
```

With `--enable-depth --enable-geometry`, each valid depth pixel is projected as `X = (u - cx) * Z / fx`, `Y = (v - cy) * Z / fy`, `Z = depth`. The coordinate frame is camera based: X right, Y down, Z forward, unit meter. Point clouds are saved as `outputs/perception/geometry/point_clouds/frame_XXXXXX.npz` with `points_xyz`, `pixels_uv`, `depth_values`, and optional `semantic_labels`. JSON stores only metadata and paths, not raw point arrays. If calibration values remain `null`, geometry cannot run. This is still camera-coordinate 3D output, not BEV; the next stage is Semantic BEV Grid.

## Camera-centric Semantic BEV Grid

`--enable-bev` projects current-frame camera-coordinate XYZ points onto the X-Z plane. X is left/right, Z is forward, and `configs/bev.yaml` controls grid range and `resolution_m`; the default is X `[-20, 20]`, Z `[0, 80]`, at `0.2 m/cell`. Semantic labels produce a class-ID grid, while region masks derive drivable/non-drivable/unknown cells from the existing scene class mapping. Conflicts use the nearest point by Euclidean distance with point-index tie break. This is camera-centric X-Z projection, not world-coordinate or calibrated ground-plane BEV: pitch, roll, camera height, and extrinsics are not applied. Drivable masks are semantic projections only and do not guarantee a safe path. The next stage is Occupancy/Cost Grid.

## Semantic Occupancy and Cost Grid

`--enable-mapping` converts the in-memory Semantic BEV into `UNKNOWN=-1`, `FREE=0`, and `OCCUPIED=100`: road alone is free by default, while sidewalk and static/dynamic obstacles are occupied; unobserved, sky, and unmapped cells remain unknown. Traversability costs use float32 values from 0 to 1 with unknown stored as NaN. Linear obstacle inflation uses a meter-valued radius (`1.0 m` by default). NPY files are computation sources; PNG files are visual encodings only (occupancy: unknown 127, free 255, occupied 0). This output is not yet a Potential Field.

## Goal-conditioned Potential Field

`--enable-potential` uses the in-memory mapping grid and either a grid goal (`--goal-row`, `--goal-col`) or camera-centric metric goal (`--goal-x`, `--goal-z`). Goals must be observed FREE cells, so unknown and occupied cells are rejected. Attractive, repulsive, and combined potentials include traversability cost; unknown is blocked by default.

## Potential Gradient Path Planner

`--enable-planner` consumes the current frame's in-memory potential and occupancy grids; it never reloads saved map artifacts. Supply exactly one start form: grid (`--start-row`, `--start-col`) or camera-centric metric (`--start-x`, `--start-z`). The planner traverses only FREE cells, supports deterministic 4/8 connectivity, and blocks diagonal corner cutting by default. It reports `success`, `local_minimum`, `cycle_detected`, `no_valid_neighbor`, or `max_steps_exceeded`; local minima are detected and reported, not recovered in this stage. Grid paths are `[row, col]`, metric paths are camera-frame `[x_m, z_m]` cell centers, and neither represents a vehicle trajectory or control command. The stage saves grid/metric NPY paths, JSON metadata, and a BGR PNG overlay under `outputs/perception/planner/`.

The default `hybrid` mode runs gradient descent first, then deterministically falls back to A* only after a supported gradient failure. A* excludes UNKNOWN/OCCUPIED cells and adds inflated traversability cost to its movement cost. Its fallback replans from the original start rather than joining a partial gradient path. This bypasses a local minimum through global search; it does not add path smoothing or vehicle kinematics.

## Streaming video planning

Enable scene segmentation, depth, geometry, BEV, mapping, potential, planner, and trajectory in `configs/perception.yaml`, provide calibrated camera intrinsics, then run:

```bash
python -m src.cli video-plan --input input.mp4 --output outputs/planned.mp4 --goal-x 0.5 --goal-y 0.1 --normalized-goal --metadata outputs/planned.jsonl --show-potential --show-trajectory
```

Normalized coordinates use `x=0..1` left-to-right and `y=0..1` top-to-bottom; the default start is the bottom-center pixel. Omitting the normalized flags selects pixel coordinates. Potential EMA defaults to `alpha=0.4` and restores current hard-obstacle values. Trajectory blending defaults to `alpha=0.5`; previous trajectories are reused only while collision-free, for at most three frames. Goal, frame-shape, or grid-shape changes reset temporal state.

MP4 and optional JSONL are written frame-by-frame. With `--frame-stride N`, timestamps retain the source timeline and output FPS is `source_fps / N`. The default `mp4v` codec depends on OpenCV build support. Overlay categories can be disabled individually.

This is camera-image visualization backed by the configured camera-centric grid. It does not correct perspective into world coordinates or produce steering, speed, vehicle dynamics, or hardware-control commands.

## Roadmap

1. YOLO object detection for driving video.
2. YOLO instance segmentation module.
3. Unified detection + instance-segmentation perception pipeline.
4. Scene semantic segmentation for road, sidewalk, lanes, buildings, sky, and vegetation.
5. Monocular metric depth estimation.
6. Camera intrinsics and 3D projection.
7. Semantic BEV transformation and map generation.
8. Semantic potential field generation.
9. Potential-gradient grid planning.
10. Local-minima handling, hybrid planning, and trajectory smoothing.
