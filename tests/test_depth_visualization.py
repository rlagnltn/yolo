import numpy as np
import pytest

from src.depth.visualization import (
    colorize_depth_map, draw_depth_overlay, normalize_depth_for_visualization,
)


def test_visualization_shapes_invalid_black_and_source_unchanged():
    frame = np.full((2, 3, 3), 20, dtype=np.uint8)
    frame_before = frame.copy()
    depth = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    color = colorize_depth_map(depth, 0, 100)
    overlay = draw_depth_overlay(frame, depth, 0.5, 0, 100)
    assert color.shape == frame.shape
    assert overlay.shape == frame.shape
    assert np.array_equal(color[0, 0], [0, 0, 0])
    assert np.array_equal(frame, frame_before)


def test_constant_depth_and_shape_errors_are_safe():
    normalized = normalize_depth_for_visualization(np.full((2, 2), 7.0, dtype=np.float32))
    assert np.all(normalized == 255)
    with pytest.raises(ValueError):
        draw_depth_overlay(np.zeros((2, 2, 3), dtype=np.uint8), np.ones((3, 2)))
