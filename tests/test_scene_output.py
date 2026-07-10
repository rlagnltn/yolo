import numpy as np
from pathlib import Path

from src.scene_segmentation.output import build_scene_frame_result


def test_scene_outputs_are_saved_as_separate_png_artifacts(tmp_path):
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    class_map = np.array([[0, 0, 1], [2, 3, 0]], dtype=np.uint8)
    result = build_scene_frame_result(
        frame, class_map, {0: "road", 1: "sidewalk", 2: "building", 3: "sky"}, 0, 0.0,
        class_map_dir=tmp_path / "class", color_map_dir=tmp_path / "color",
        visualization_dir=tmp_path / "vis", region_dir=tmp_path / "regions",
    )

    for key in (
        "class_map_path", "color_map_path", "overlay_path",
        "drivable_mask_path", "non_drivable_mask_path",
    ):
        assert key in result
        assert Path(result[key]).is_file()
    assert result["regions"]["drivable_pixel_count"] == 3
