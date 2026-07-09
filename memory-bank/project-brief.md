# Project Brief

## Project Name

Vision Potential Field

## Research Background

Autonomous driving and advanced driver-assistance systems need a compact understanding of the road scene before they can plan safe motion. This project starts from camera video and builds toward a semantic potential field that can guide path planning.

## Final Goal

Develop a pipeline that accepts vehicle driving video, detects objects, estimates scene semantics and depth, transforms the scene into BEV space, generates a semantic potential field, and plans a safe path toward a target.

## Current Implementation Scope

The current scope is limited to YOLO-based object detection:

- Image/video input.
- Frame-level YOLO inference.
- JSON detection output.
- Optional bounding-box visualization.

## Future Extensions

- Semantic segmentation.
- Depth estimation.
- BEV transformation.
- Semantic map generation.
- Potential field generation.
- Path planning and local-minima mitigation.
