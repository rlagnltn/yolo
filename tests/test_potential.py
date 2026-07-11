import numpy as np
import pytest

from src.bev import BEVConfig
from src.potential import (
    PotentialConfig, calculate_potential_gradient, combine_potential_fields,
    create_attractive_potential, create_repulsive_potential, metric_goal_to_grid,
    save_potential_frame_result, validate_goal_cell,
)


def test_potential_config_and_goal_validation():
    assert PotentialConfig.from_dict({"attractive": {"mode": "conic"}}).attractive_mode == "conic"
    with pytest.raises(ValueError, match="positive"):
        PotentialConfig(repulsive_influence_radius_m=0).validate()
    with pytest.raises(ValueError, match="bounds"):
        validate_goal_cell(3, 0, (3, 3))
    config = BEVConfig(-2, 2, 0, 4, 1)
    assert metric_goal_to_grid(0.1, 0.1, config) == (3, 2)


def test_attractive_modes_and_goal_zero():
    quadratic = create_attractive_potential((3, 3), (1, 1), 1.0, 2.0)
    conic = create_attractive_potential((3, 3), (1, 1), 1.0, 2.0, mode="conic")
    assert quadratic.dtype == np.float32
    assert quadratic[1, 1] == 0
    assert quadratic[1, 2] == pytest.approx(1.0)
    assert conic[1, 2] == pytest.approx(2.0)
    assert quadratic[0, 0] > quadratic[1, 2]


def test_repulsive_distance_and_no_nonfinite_values():
    occupied = np.zeros((1, 5), dtype=bool)
    occupied[0, 0] = True
    result = create_repulsive_potential(occupied, 1.0, 2.0, 2.0)
    assert result[0, 0] > result[0, 1] > result[0, 2]
    assert result[0, 3] == 0
    assert np.isfinite(result).all()


def test_combined_blocked_cost_and_normalization():
    config = PotentialConfig(occupied_potential=1000, normalize_output=True).validate()
    attractive = np.array([[0, 1, 2]], dtype=np.float32)
    repulsive = np.zeros_like(attractive)
    costs = np.array([[0, 0.5, np.nan]], dtype=np.float32)
    occupancy = np.array([[0, 0, -1]], dtype=np.int16)
    result = combine_potential_fields(attractive, repulsive, costs, occupancy, config)
    assert result["raw_potential"][0, 1] == pytest.approx(1.5)
    assert result["blocked_mask"][0, 2]
    assert result["normalized_potential"][0, 2] == 1000
    assert np.isfinite(result["raw_potential"]).all()


def test_gradient_axis_and_blocked_cells():
    potential = np.tile(np.arange(3, dtype=np.float32), (3, 1))
    gradient = calculate_potential_gradient(potential, 1.0, np.array([[False, True, False]] * 3))
    assert gradient["gradient_x"][0, 0] > 0
    assert gradient["descent_x"][0, 0] < 0
    assert gradient["magnitude"][0, 1] == 0


def test_potential_save_reload(tmp_path):
    fields = {"attractive": np.zeros((2, 2), np.float32), "repulsive": np.ones((2, 2), np.float32),
              "combined": np.ones((2, 2), np.float32), "blocked_mask": np.zeros((2, 2), bool)}
    gradient = calculate_potential_gradient(fields["combined"], 1.0)
    metadata = save_potential_frame_result(
        0, fields, gradient, goal_cell=(0, 0), goal_metric=(0.5, 1.5), resolution_m=1.0,
        config=PotentialConfig(), attractive_dir=tmp_path / "att", repulsive_dir=tmp_path / "rep",
        combined_dir=tmp_path / "combined", gradient_dir=tmp_path / "grad", visualization_dir=tmp_path / "vis",
    )
    assert np.load(metadata["combined_path"]).shape == (2, 2)
    assert "gradient_x" in np.load(metadata["gradient_path"])
