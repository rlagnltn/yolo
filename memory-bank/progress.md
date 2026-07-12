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
- Added deterministic A* fallback from supported gradient-descent failures, using in-memory inflated cost grids.
- Added collision-checked geometric trajectories and streaming MP4/JSONL video planning with temporal stabilization.
- Defined the 10 Hz planning cadence, 0.50-second validated-path reuse limit, and temporal-continuity evaluation gates in `memory.md`.
- Implemented structured planner reason codes and `scripts/evaluate_planner.py` temporal evaluation.
- Implemented stable auto starts, 5/4/3 m horizon selection, and shared corner-safe neighbor rules.
- Implemented 10 Hz planning with current-occupancy-validated path reuse capped at 0.50 seconds.
- Re-ran all 299 frames: 39/100 full-horizon raw successes, 2 short horizons, zero safely reusable frames, and zero reported safety violations; temporal criteria remain unmet.
- Added stride-2 geometry, temporal occupancy, adaptive recovery planning, and nearby-component start fallback.
- Replaced repeated Python BFS with one OpenCV connected-component labeling pass and verified equivalence against planner movement rules.
- Added `--start-frame` with count-based `--max-frames` for independent 100-frame chunks.
- Holdout passed all gates at 96.66% availability; the hard 100-frame chunk completed in 5m53s with zero safety violations.
- Added original-video path overlay rendering from planner grid paths plus same-frame point-cloud pixel coordinates.
- Generated `outputs/perception/path_overlay_100_199.mp4`: 100 frames at 960x540 and 29.97 FPS, with 81 new successful paths drawn and no missing path/point-cloud artifacts.
- Generated `outputs/perception/path_overlay_full_0.3x.mp4` from three consistent source-video chunks: 299 frames at 960x540, 8.99 FPS (0.3x), 263 new paths drawn, and no missing path/point-cloud artifacts.

## In Progress

- Improve dynamic FREE-cell planning continuity against the baseline recorded in `memory.md`.

## Remaining

- Run on a real sample driving video after the user places a file at `datasets/raw/sample.mp4` or passes another path with `--input`.
- Vehicle kinematics, speed profiles, tracking, and dynamic replanning remain outside the current implementation.
- Reduce `no_connected_forward_goal` failures through temporal BEV stabilization, calibrated ground-plane geometry, and stable start/goal selection.

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
