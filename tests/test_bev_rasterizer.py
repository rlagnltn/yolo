import numpy as np
import pytest

from src.bev import (
    BEVConfig,
    create_bev_region_masks,
    points_to_bev_indices,
    rasterize_observation_bev,
    rasterize_semantic_bev,
    save_bev_frame_result,
)


def test_points_to_bev_indices_known_xyz_and_filters():
    config = BEVConfig(-2.0, 2.0, 0.0, 4.0, 1.0, min_y_m=-1.0, max_y_m=1.0)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.9, 0.0, 3.9],
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [np.nan, 0.0, 1.0],
        [np.inf, 0.0, 1.0],
    ], dtype=np.float32)
    out = points_to_bev_indices(points, config)
    np.testing.assert_array_equal(out["row_indices"], [3, 0])
    np.testing.assert_array_equal(out["col_indices"], [2, 3])
    np.testing.assert_array_equal(out["valid_point_indices"], [0, 1])


def test_empty_point_cloud_and_bad_shape():
    config = BEVConfig(-1.0, 1.0, 0.0, 2.0, 1.0)
    out = points_to_bev_indices(np.empty((0, 3), dtype=np.float32), config)
    assert out["row_indices"].shape == (0,)
    with pytest.raises(ValueError, match="shape"):
        points_to_bev_indices(np.empty((3,), dtype=np.float32), config)


def test_float32_upper_boundary_never_produces_out_of_bounds_index():
    config = BEVConfig(-20.0, 20.0, 0.0, 80.0, 0.2)
    upper_x = np.nextafter(np.float32(20.0), np.float32(-np.inf))
    upper_z = np.nextafter(np.float32(80.0), np.float32(-np.inf))
    out = points_to_bev_indices(np.asarray([[upper_x, 0.0, upper_z]], np.float32), config)
    assert 0 <= out["col_indices"][0] < config.width_cells
    assert 0 <= out["row_indices"][0] < config.height_cells


def test_semantic_raster_nearest_tie_break_and_counts():
    config = BEVConfig(-1.0, 1.0, 0.0, 2.0, 1.0)
    points = np.array([
        [0.2, 0.0, 0.2],
        [0.3, 0.0, 0.3],
        [0.2, 0.0, 0.2],
    ], dtype=np.float32)
    labels = np.array([1, 2, 3], dtype=np.uint8)
    out = rasterize_semantic_bev(points, labels, config)
    assert out["class_grid"][1, 1] == 1
    assert out["observed_mask"][1, 1]
    assert out["point_count_grid"][1, 1] == 3


def test_semantic_raster_rejects_label_mismatch_and_policy():
    config = BEVConfig(-1.0, 1.0, 0.0, 2.0, 1.0)
    points = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="semantic_labels"):
        rasterize_semantic_bev(points, np.array([1]), config)
    with pytest.raises(ValueError, match="Unsupported"):
        rasterize_semantic_bev(points, np.array([1, 2]), config, "last")


def test_observation_bev_without_semantic_labels():
    config = BEVConfig(-1.0, 1.0, 0.0, 2.0, 1.0)
    out = rasterize_observation_bev(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), config)
    assert out["observed_mask"][0, 1]
    assert out["point_count_grid"][0, 1] == 1
    assert np.isfinite(out["nearest_distance_grid"][0, 1])


def test_region_masks_reuse_scene_mapping_policy():
    class_grid = np.array([[0, 1, 10], [11, 12, 255]], dtype=np.uint8)
    observed = np.array([[True, True, True], [True, True, False]])
    masks = create_bev_region_masks(
        class_grid, observed,
        {0: "road", 1: "sidewalk", 10: "sky", 11: "person", 12: "car"},
    )
    assert masks["drivable_mask"][0, 0]
    assert not masks["drivable_mask"][0, 1]
    assert masks["non_drivable_mask"][0, 1]
    assert masks["non_drivable_mask"][1, 0]
    assert masks["non_drivable_mask"][1, 1]
    assert masks["unknown_mask"][0, 2]
    assert masks["unknown_mask"][1, 2]


def test_bev_save_and_reload(tmp_path):
    config = BEVConfig(-1.0, 1.0, 0.0, 2.0, 1.0)
    bev = rasterize_semantic_bev(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([0], dtype=np.uint8),
        config,
    )
    meta = save_bev_frame_result(
        0, bev, config,
        id2label={0: "road"},
        class_grid_dir=tmp_path,
        drivable_grid_dir=tmp_path,
        non_drivable_grid_dir=tmp_path,
        visualization_dir=tmp_path,
    )
    loaded = np.load(meta["class_grid_path"])
    np.testing.assert_array_equal(loaded, bev["class_grid"])
    assert meta["projection_type"] == "camera_centric_xz"
