import pytest

from src.bev import BEVConfig


def test_bev_config_validates_and_calculates_shape():
    config = BEVConfig(-2.0, 2.0, 0.0, 4.0, 0.5)
    assert config.width_cells == 8
    assert config.height_cells == 8
    assert config.shape == (8, 8)
    assert config.to_dict()["resolution_m"] == 0.5


def test_bev_config_rejects_bad_ranges_resolution_and_y_order():
    with pytest.raises(ValueError, match="x_min"):
        BEVConfig(1.0, 1.0, 0.0, 4.0, 0.5).validate()
    with pytest.raises(ValueError, match="resolution"):
        BEVConfig(-1.0, 1.0, 0.0, 4.0, 0.0).validate()
    with pytest.raises(ValueError, match="min_y"):
        BEVConfig(-1.0, 1.0, 0.0, 4.0, 0.5, min_y_m=2.0, max_y_m=1.0).validate()


def test_bev_config_rejects_too_many_cells():
    with pytest.raises(ValueError, match="too large"):
        BEVConfig(-1000.0, 1000.0, 0.0, 1000.0, 0.01).validate()
