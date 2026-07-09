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

## In Progress

- Sample driving-video based YOLO detection validation.

## Remaining

- Run on a real sample driving video after the user places a file at `datasets/raw/sample.mp4` or passes another path with `--input`.
- Implement semantic segmentation.
- Implement depth estimation.
- Implement BEV transformation.
- Implement potential field and planner modules.

## Next Milestone

Run:

```bash
python scripts/run_detection.py --input datasets/raw/sample.mp4 --save-vis
```

Then inspect detection JSON and annotated frames.

For a shorter smoke test after adding a sample video:

```bash
python scripts/run_detection.py --input datasets/raw/sample.mp4 --save-vis --max-frames 100
```

Expected outputs:

- `outputs/detections/detections.json`
- `outputs/visualizations/`
