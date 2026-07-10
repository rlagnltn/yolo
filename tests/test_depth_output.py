import json
from pathlib import Path

import cv2
import numpy as np

from src.depth.output import build_depth_frame_result


def test_depth_artifacts_and_schema_are_saved_without_embedding_array(tmp_path):
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    prediction = {
        "depth_map": np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32),
        "depth_type": "metric", "unit": "meter", "model_name": "mock",
    }
    result = build_depth_frame_result(
        frame, prediction, 0, 0.0,
        raw_depth_dir=tmp_path / "raw", depth_png_dir=tmp_path / "png",
        color_map_dir=tmp_path / "color", visualization_dir=tmp_path / "vis",
    )
    assert np.load(result["raw_depth_path"]).dtype == np.float32
    encoded = np.fromfile(result["depth_png_path"], dtype=np.uint8)
    assert cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED).dtype == np.uint16
    for key in ("raw_depth_path", "depth_png_path", "color_map_path", "overlay_path"):
        assert Path(result[key]).is_file()
    assert result["png_scale"] == 1000.0
    assert "depth_map" not in result
    json.dumps(result)
