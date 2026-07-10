import numpy as np
import pytest

from src.mapping import (
    MappingConfig, colorize_cost_grid, colorize_occupancy_grid,
    create_semantic_cost_grid, create_semantic_occupancy_grid,
    inflate_cost_grid, occupancy_to_cost_grid, save_mapping_frame_result,
)


LABELS = {0: "road", 1: "sidewalk", 2: "car", 3: "person", 4: "sky", 5: "mystery"}


def test_mapping_config_validation():
    config = MappingConfig.from_dict({"inflation": {"radius_m": 1.5}})
    assert config.inflation_radius_m == 1.5
    with pytest.raises(ValueError, match="distinct"):
        MappingConfig(unknown_value=0).validate()
    with pytest.raises(ValueError, match="non-negative"):
        MappingConfig(inflation_radius_m=-1).validate()
    with pytest.raises(ValueError, match="linear"):
        MappingConfig(inflation_decay="none").validate()


def test_semantic_occupancy_states_and_partition():
    grid = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    observed = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)
    result = create_semantic_occupancy_grid(grid, observed, LABELS, MappingConfig())
    np.testing.assert_array_equal(result["occupancy_grid"], [[0, 100, 100], [100, -1, -1]])
    assert not (result["free_mask"] & result["occupied_mask"]).any()
    assert (result["free_mask"] | result["occupied_mask"] | result["unknown_mask"]).all()


def test_cost_grid_dtype_unknown_and_semantic_costs():
    config = MappingConfig(semantic_costs={"road": 0.2}).validate()
    occupancy = np.array([[0, 100, -1]], dtype=np.int16)
    costs = occupancy_to_cost_grid(occupancy, config)
    assert costs.dtype == np.float32
    np.testing.assert_allclose(costs[0, :2], [0.0, 1.0])
    assert np.isnan(costs[0, 2])
    semantic = create_semantic_cost_grid(np.array([[0, 4]]), np.ones((1, 2), bool), LABELS, config)
    assert semantic[0, 0] == pytest.approx(0.2)
    assert np.isnan(semantic[0, 1])


def test_inflation_radius_zero_linear_decay_and_unknown():
    costs = np.zeros((1, 5), dtype=np.float32)
    costs[0, 4] = np.nan
    occupied = np.zeros((1, 5), dtype=bool)
    occupied[0, 0] = True
    costs[0, 0] = 1.0
    original = inflate_cost_grid(costs, occupied, 0.5, 0.0)
    np.testing.assert_array_equal(original, costs)
    result = inflate_cost_grid(costs, occupied, 0.5, 1.0)
    np.testing.assert_allclose(result[0, :4], [1.0, 0.5, 0.0, 0.0], atol=1e-6)
    assert np.isnan(result[0, 4])
    assert np.isnan(costs[0, 4])


def test_mapping_save_reload_and_visualization(tmp_path):
    config = MappingConfig()
    occupancy = create_semantic_occupancy_grid(
        np.array([[0, 1, 4]], dtype=np.uint8), np.ones((1, 3), bool), LABELS, config
    )
    costs = occupancy_to_cost_grid(occupancy["occupancy_grid"], config)
    metadata = save_mapping_frame_result(
        0, occupancy, costs, costs, resolution_m=0.2, config=config,
        occupancy_dir=tmp_path / "occ", cost_grid_dir=tmp_path / "cost",
        inflated_cost_dir=tmp_path / "inflated", visualization_dir=tmp_path / "vis",
    )
    np.testing.assert_array_equal(np.load(metadata["occupancy_grid_path"]), occupancy["occupancy_grid"])
    assert np.isnan(np.load(metadata["cost_grid_path"])[0, 2])
    image = colorize_occupancy_grid(occupancy["occupancy_grid"], config)
    assert image.shape == (1, 3, 3)
    cost_image = colorize_cost_grid(costs)
    assert tuple(cost_image[0, 2]) == (127, 127, 127)
