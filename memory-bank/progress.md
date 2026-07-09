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

## In Progress

- Vehicle driving-video based semantic segmentation validation.

## Remaining

- Run on a real sample driving video after the user places a file at `datasets/raw/sample.mp4` or passes another path with `--input`.
- Verify detection and segmentation together on the same driving-video frames.
- Implement depth estimation.
- Implement BEV transformation.
- Implement potential field and planner modules.
- BEV, Potential Field, and Path Planning are not implemented yet.

## Next Milestone

Run:

```bash
python scripts/run_segmentation.py --input datasets/raw/sample.mp4 --save-vis
```

Then inspect segmentation JSON, mask images, and overlay frames.

For a shorter smoke test after adding a sample video:

```bash
python scripts/run_segmentation.py --input datasets/raw/sample.mp4 --save-vis --max-frames 100
```

Expected outputs:

- `outputs/segmentations/segmentations.json`
- `outputs/segmentations/masks/`
- `outputs/segmentations/visualizations/`
