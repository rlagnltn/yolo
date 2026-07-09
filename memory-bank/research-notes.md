# Research Notes

## Semantic Mapping From Driving Video

The project should eventually combine object instances, semantic regions, and depth cues into a spatial representation that can support planning.

## Object Detection vs Semantic Segmentation

Object detection identifies object instances with bounding boxes. Semantic segmentation assigns a class label to each pixel. Both are useful: detection gives compact object-level obstacles, while segmentation gives road, lane, sidewalk, and drivable-area context.

## Why BEV Matters

Path planning is easier in a top-down coordinate space than in perspective camera space. BEV transformation can align detections and semantic areas with motion-planning coordinates.

## Why Potential Fields

Potential fields can represent attractive forces toward a goal and repulsive forces away from obstacles. This makes them a useful bridge between perception outputs and local planning.

## Open Research Issue

Potential fields can suffer from local minima. Future work should evaluate escape strategies, hybrid planners, or graph-search fallbacks.
