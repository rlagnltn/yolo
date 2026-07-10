import numpy as np
import pytest

from src.depth.postprocessing import (
    calculate_depth_by_class, calculate_depth_statistics, create_valid_depth_mask,
    depth_to_uint16, resize_depth_map, sanitize_depth_map, uint16_to_depth,
)


def test_sanitize_mask_and_statistics_ignore_invalid_values():
    raw = np.array([[np.nan, np.inf, -1.0], [0.0, 2.0, 4.0]], dtype=np.float64)
    depth = sanitize_depth_map(raw)
    assert depth.dtype == np.float32
    assert np.array_equal(create_valid_depth_mask(depth), [[False, False, False], [False, True, True]])
    stats = calculate_depth_statistics(depth)
    assert stats["valid_pixel_count"] == 2
    assert stats["invalid_pixel_count"] == 4
    assert stats["mean_depth"] == 3.0
    assert stats["median_depth"] == 3.0
    assert stats["percentile_05"] == pytest.approx(2.1)


def test_statistics_with_no_valid_pixels_are_safe_and_json_friendly():
    stats = calculate_depth_statistics(np.zeros((2, 2), dtype=np.float32))
    assert stats["min_depth"] is None
    assert stats["valid_pixel_ratio"] == 0.0


def test_resize_preserves_requested_shape_and_float32():
    result = resize_depth_map(np.ones((2, 2), dtype=np.float64), 3, 5)
    assert result.shape == (3, 5)
    assert result.dtype == np.float32


def test_uint16_round_trip_overflow_and_scale_validation():
    depth = np.array([[0.0, 1.234, 100.0]], dtype=np.float32)
    encoded = depth_to_uint16(depth, 1000.0)
    assert encoded.dtype == np.uint16
    assert encoded.tolist() == [[0, 1234, 65535]]
    decoded = uint16_to_depth(encoded, 1000.0)
    assert decoded[0, 1] == pytest.approx(1.234)
    with pytest.raises(ValueError):
        depth_to_uint16(depth, 0)
    with pytest.raises(ValueError):
        uint16_to_depth(encoded, -1)


def test_depth_by_class_is_vectorized_summary_and_validates_shape():
    depth = np.array([[1.0, 0.0], [3.0, 5.0]], dtype=np.float32)
    classes = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    result = calculate_depth_by_class(depth, classes, {0: "road", 1: "car"})
    assert result["road"]["pixel_count"] == 1
    assert result["car"]["median_depth"] == 4.0
    with pytest.raises(ValueError):
        calculate_depth_by_class(depth, np.zeros((1, 1)), {})
