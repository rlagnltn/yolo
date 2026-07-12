import numpy as np
import pytest

from src.mapping import TemporalOccupancyFusion


def test_temporal_occupancy_retains_only_previous_observations_until_ttl():
    fusion = TemporalOccupancyFusion(ttl_frames=2)
    first = np.asarray([[0, -1], [100, -1]], np.int16)
    fused, stats = fusion.update(first)
    assert np.array_equal(fused, first)
    assert stats["retained_observation_cell_count"] == 0

    unknown = np.full((2, 2), -1, np.int16)
    fused, stats = fusion.update(unknown)
    assert fused[0, 0] == 0 and fused[1, 0] == 100
    assert stats["retained_observation_cell_count"] == 2
    fusion.update(unknown)
    expired, _ = fusion.update(unknown)
    assert np.all(expired == -1)


def test_current_observation_overrides_temporal_state():
    fusion = TemporalOccupancyFusion(ttl_frames=8)
    fusion.update(np.asarray([[0, 100]], np.int16))
    fused, _ = fusion.update(np.asarray([[100, 0]], np.int16))
    assert np.array_equal(fused, np.asarray([[100, 0]], np.int16))


def test_temporal_occupancy_rejects_invalid_configuration_and_shape():
    with pytest.raises(ValueError):
        TemporalOccupancyFusion(-1)
    with pytest.raises(ValueError):
        TemporalOccupancyFusion().update(np.zeros(3))
