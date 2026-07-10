import numpy as np
import pytest

from src.scene_segmentation.postprocessing import (
    calculate_class_statistics,
    calculate_region_statistics,
    create_binary_region_mask,
    create_drivable_mask,
    create_non_drivable_mask,
    validate_class_map,
    logits_to_class_map,
)

LABELS = {0: "road", 1: "sidewalk", 2: "building", 3: "sky"}


def test_statistics_counts_all_pixels_and_ratios_sum_to_one():
    class_map = np.array([[0, 0], [1, 2]], dtype=np.uint8)
    statistics = calculate_class_statistics(class_map, LABELS)
    assert sum(item["pixel_count"] for item in statistics) == 4
    assert sum(item["pixel_ratio"] for item in statistics) == pytest.approx(1.0)


def test_region_masks_are_binary_uint8_and_road_only_is_drivable():
    class_map = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    drivable = create_drivable_mask(class_map, LABELS)
    non_drivable = create_non_drivable_mask(class_map, LABELS)
    assert drivable.dtype == np.uint8
    assert set(np.unique(drivable)) <= {0, 255}
    assert drivable.tolist() == [[255, 0], [0, 0]]
    assert non_drivable.tolist() == [[0, 255], [255, 0]]
    regions = calculate_region_statistics(class_map, drivable, non_drivable)
    assert regions == {
        "drivable_pixel_count": 1,
        "drivable_pixel_ratio": 0.25,
        "non_drivable_pixel_count": 2,
        "unknown_pixel_count": 1,
    }


def test_missing_class_ids_produce_empty_mask_safely():
    class_map = np.zeros((2, 2), dtype=np.uint8)
    assert not create_binary_region_mask(class_map, {99}).any()


@pytest.mark.parametrize("invalid", [np.array([]), np.zeros((1, 1, 1)), np.array([[0.5]])])
def test_invalid_class_map_is_rejected(invalid):
    with pytest.raises(ValueError):
        validate_class_map(invalid)


def test_wrong_expected_shape_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        validate_class_map(np.zeros((2, 3), dtype=np.uint8), (3, 2))


def test_logits_are_resized_to_original_frame_shape():
    import torch

    logits = torch.tensor([[[[0.0]], [[1.0]]]])
    class_map = logits_to_class_map(logits, 3, 4)
    assert class_map.shape == (3, 4)
    assert class_map.dtype == np.uint8
    assert np.all(class_map == 1)
