"""SegFormer-based full-scene semantic segmentation."""

from .class_mapping import classify_label, normalize_id2label
from .segmenter import SceneSegmenter

__all__ = ["SceneSegmenter", "classify_label", "normalize_id2label"]
