# Progress

## Completed

- Created project folder structure.
- Added detection config.
- Added YOLO detector wrapper.
- Added image/video detection CLI.
- Added JSON and visualization utilities.
- Added memory-bank documentation.
- Initial research project structure is complete.
- Detection pipeline validation is ready.
- Added `--max-frames` for small sample runs.
- Documented manual sample-video placement and output locations.
- Detection pipeline implementation is complete.
- Added YOLO segmentation module scaffolding and CLI.
- Added mask output, segmentation JSON schema, and overlay visualization support.
- Added a unified detection + instance-segmentation perception pipeline.
- Added class-aware, one-to-one IoU fusion with matched and unmatched states.
- Added unified JSON, mask, and visualization outputs under `outputs/perception/`.
- Added SegFormer-B0 Cityscapes scene semantic segmentation.
- Added class-ID maps, color maps, semantic overlays, and drivable/non-drivable masks.
- Integrated optional scene segmentation into the single-pass perception pipeline.
- Added Depth Anything V2 outdoor monocular metric-depth estimation.
- Added float32 NPY, uint16 PNG, color-map, overlay, and depth-statistics outputs.
- Integrated optional depth and same-frame scene-class depth summaries into unified perception.
- Added depth tests and verified the real metric model on CPU.
- Added camera intrinsics handling and metric-depth 3D back-projection.
- Added camera-coordinate point cloud NPZ outputs with optional semantic class IDs.
- Added camera-centric Semantic BEV Grid generation.
- Added Semantic Occupancy / Traversability Cost Grid generation and metric obstacle inflation.
- Added Goal-conditioned Potential Field generation and gradients.
- Added deterministic potential-gradient grid planning with optional unified-pipeline execution.

## In Progress

- Final validation and commit of Potential Field Gradient Path Planner.

## Remaining

- Run on a real sample driving video after the user places a file at `datasets/raw/sample.mp4` or passes another path with `--input`.
- Implement potential field and planner modules.
- Path Planning is not implemented yet.

## Next Milestone

Run:

```bash
python scripts/run_perception.py --input datasets/raw/sample.mp4 --enable-scene-segmentation --enable-depth --enable-geometry --save-vis --max-frames 5
```

Then inspect segmentation JSON, mask images, and overlay frames.

For a shorter smoke test after adding a sample video:

```bash
python scripts/run_segmentation.py --input datasets/raw/sample.mp4 --save-vis --max-frames 100
```

Expected outputs:

- `outputs/perception/perception.json`
- `outputs/perception/masks/`
- `outputs/perception/visualizations/`
