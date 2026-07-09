# Progress

## Completed

- Created project folder structure.
- Added detection config.
- Added YOLO detector wrapper.
- Added image/video detection CLI.
- Added JSON and visualization utilities.
- Added memory-bank documentation.

## In Progress

- First detection pipeline validation.

## Remaining

- Run on a real sample driving video.
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
