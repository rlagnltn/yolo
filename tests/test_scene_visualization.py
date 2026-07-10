import numpy as np
import pytest

from src.utils.visualization import colorize_class_map, draw_scene_segmentation_overlay


def test_color_map_shape_and_unknown_id_handling():
    class_map = np.array([[0, 250]], dtype=np.uint8)
    colored = colorize_class_map(class_map)
    assert colored.shape == (1, 2, 3)
    assert colored[0, 1].tolist() == [64, 64, 64]


def test_scene_overlay_does_not_mutate_input():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    original = frame.copy()
    result = draw_scene_segmentation_overlay(frame, np.zeros((2, 2), dtype=np.uint8))
    assert result.shape == frame.shape
    assert np.array_equal(frame, original)


def test_scene_overlay_validates_alpha_and_shape():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="alpha"):
        draw_scene_segmentation_overlay(frame, np.zeros((2, 2), dtype=np.uint8), 1.1)
    with pytest.raises(ValueError, match="does not match"):
        draw_scene_segmentation_overlay(frame, np.zeros((1, 2), dtype=np.uint8))
