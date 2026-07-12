import numpy as np
from collections import deque

from src.bev import BEVConfig
from src.planner import select_auto_free_cells
from src.planner.auto_free_cells import _connected_component
from src.planner.neighbors import iter_free_neighbors


def _bev():
    return BEVConfig(-5, 5, 0, 10, 1)


def test_select_auto_free_cells_uses_connected_forward_centerline_cells():
    occupancy = np.full((10, 10), -1, np.int16)
    occupancy[2:10, 4:7] = 0
    result = select_auto_free_cells(occupancy, _bev(), centerline_half_width_m=2, minimum_forward_distance_m=5)
    assert result["status"] == "selected"
    assert occupancy[result["start_cell"]] == 0
    assert occupancy[result["goal_cell"]] == 0
    assert result["goal_cell"][0] < result["start_cell"][0]


def test_select_auto_free_cells_reports_missing_connected_goal():
    occupancy = np.full((10, 10), -1, np.int16)
    occupancy[8:10, 4:7] = 0
    result = select_auto_free_cells(occupancy, _bev(), minimum_forward_distance_m=5)
    assert result["status"] == "failed"
    assert result["reason_code"] == "NO_FORWARD_GOAL"
    assert result["statistics"]["forward_goal_candidate_count"] == 0


def test_select_auto_free_cells_uses_short_horizon_and_stable_start():
    occupancy = np.full((10, 10), -1, np.int16)
    occupancy[4:10, 4:7] = 0
    result = select_auto_free_cells(
        occupancy, _bev(), minimum_forward_distance_m=7,
        fallback_forward_distances_m=(4, 3), previous_start_cell=(9, 5),
    )
    assert result["status"] == "selected"
    assert result["horizon_status"] == "short_horizon"
    assert result["selected_forward_distance_m"] == 4
    assert result["statistics"]["start_selection_source"] == "previous_start_neighborhood"


def test_auto_selection_uses_same_corner_rule_as_planner():
    occupancy = np.full((4, 4), -1, np.int16)
    occupancy[3, 1] = occupancy[2, 2] = occupancy[1, 2] = 0
    result = select_auto_free_cells(
        occupancy, BEVConfig(-2, 2, 0, 4, 1), minimum_forward_distance_m=1,
        fallback_forward_distances_m=(), centerline_half_width_m=2,
        connectivity=8, prevent_corner_cutting=True,
    )
    assert result["status"] == "failed"
    assert result["statistics"]["planner_reachable_cell_count"] == 1


def test_auto_selection_falls_back_to_nearby_connected_start_component():
    occupancy = np.full((10, 10), -1, np.int16)
    occupancy[9, 4] = 0
    occupancy[2:10, 6] = 0
    result = select_auto_free_cells(
        occupancy, _bev(), centerline_half_width_m=2,
        minimum_forward_distance_m=5, fallback_forward_distances_m=(4, 3),
        alternative_start_search_radius_m=.5,
    )
    assert result["status"] == "selected"
    assert result["start_cell"] == (9, 6)
    assert result["statistics"]["start_selection_source"] == "alternative_connected_component"


def test_fast_component_labeling_matches_planner_neighbor_rules():
    rng = np.random.default_rng(7)
    for connectivity, prevent_corner_cutting in ((4, True), (8, True), (8, False)):
        for _ in range(10):
            occupancy = np.where(rng.random((20, 25)) > .35, 0, -1).astype(np.int16)
            free_cells = np.argwhere(occupancy == 0)
            start = tuple(map(int, free_cells[rng.integers(len(free_cells))]))
            expected = np.zeros_like(occupancy, dtype=bool)
            expected[start] = True
            queue = deque([start])
            while queue:
                cell = queue.popleft()
                for row, col, _, _ in iter_free_neighbors(
                    occupancy, cell, connectivity=connectivity,
                    prevent_corner_cutting=prevent_corner_cutting,
                ):
                    if not expected[row, col]:
                        expected[row, col] = True
                        queue.append((row, col))
            actual = _connected_component(
                occupancy, start, connectivity=connectivity,
                prevent_corner_cutting=prevent_corner_cutting,
            )
            np.testing.assert_array_equal(actual, expected)
