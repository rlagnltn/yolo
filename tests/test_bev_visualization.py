import numpy as np

from src.bev import BEVConfig
from src.bev.visualization import colorize_bev_class_grid, draw_bev_grid_lines, mark_camera_origin


def test_bev_visualization_is_deterministic_and_marks_origin():
    config = BEVConfig(-1.0, 1.0, 0.0, 2.0, 1.0)
    grid = np.array([[255, 0], [1, 255]], dtype=np.uint8)
    observed = np.array([[False, True], [True, False]])
    image = colorize_bev_class_grid(grid, observed, config)
    image2 = colorize_bev_class_grid(grid, observed, config)
    np.testing.assert_array_equal(image, image2)
    lined = draw_bev_grid_lines(image, spacing_cells=1)
    marked = mark_camera_origin(lined)
    assert marked.shape == (2, 2, 3)
    assert marked[-1, 1].tolist() == [0, 255, 255]
