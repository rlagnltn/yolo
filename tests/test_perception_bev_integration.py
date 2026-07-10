import numpy as np

from src.perception.pipeline import PerceptionPipeline


class EmptyDetector:
    model_name = "det.pt"
    def detect_frame(self, frame):
        return []


class EmptySegmenter:
    model_name = "seg.pt"
    def segment_frame(self, frame, *args, **kwargs):
        return []


class MockDepthEstimator:
    model_name = "depth"
    def predict(self, frame):
        return {
            "depth_map": np.ones(frame.shape[:2], dtype=np.float32),
            "depth_type": "metric",
            "unit": "meter",
            "model_name": self.model_name,
        }


class MockScene:
    model_name = "scene"
    id2label = {0: "road", 1: "car"}
    def predict(self, frame):
        return np.array([[0, 1], [0, 1]], dtype=np.uint8)


def _depth_options(tmp_path):
    return {"raw_depth_dir": tmp_path, "depth_png_dir": tmp_path,
            "color_map_dir": tmp_path, "visualization_dir": tmp_path,
            "save_raw_depth": False, "save_depth_png": False,
            "save_color_maps": False, "save_visualizations": False}


def _scene_options(tmp_path):
    return {"class_map_dir": tmp_path, "color_map_dir": tmp_path,
            "visualization_dir": tmp_path, "region_dir": tmp_path,
            "save_class_maps": False, "save_color_maps": False, "save_regions": False}


def _geometry_options(tmp_path):
    return {
        "enabled": True,
        "point_cloud_dir": tmp_path,
        "stride": 1,
        "min_depth_m": 0.1,
        "max_depth_m": 80.0,
        "intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "width": 2, "height": 2},
    }


def _bev_options(tmp_path):
    return {
        "enabled": True,
        "config": {"x_min_m": -1, "x_max_m": 2, "z_min_m": 0, "z_max_m": 2,
                   "resolution_m": 1, "unknown_class_id": 255},
        "id2label": {0: "road", 1: "car"},
        "class_grid_dir": tmp_path,
        "drivable_grid_dir": tmp_path,
        "non_drivable_grid_dir": tmp_path,
        "visualization_dir": tmp_path,
        "save_class_grid_npy": True,
        "save_class_grid_png": False,
        "save_region_masks": False,
        "save_visualizations": False,
    }


def test_bev_disabled_preserves_pipeline(tmp_path):
    result = PerceptionPipeline(
        EmptyDetector(), EmptySegmenter(), depth_estimator=MockDepthEstimator()
    ).process_frame(
        np.zeros((2, 2, 3), dtype=np.uint8), 0,
        depth_output=_depth_options(tmp_path),
        geometry_output=_geometry_options(tmp_path),
    )
    assert result["bev"] is None


def test_bev_requires_geometry_when_enabled(tmp_path):
    result = PerceptionPipeline(
        EmptyDetector(), EmptySegmenter(), depth_estimator=MockDepthEstimator()
    ).process_frame(
        np.zeros((2, 2, 3), dtype=np.uint8), 0,
        depth_output=_depth_options(tmp_path),
        bev_output=_bev_options(tmp_path),
    )
    assert result["bev"] is None
    assert result["errors"][-1] == "bev: geometry must be enabled before BEV generation."


def test_bev_metadata_from_mock_geometry_and_semantics(tmp_path):
    result = PerceptionPipeline(
        EmptyDetector(), EmptySegmenter(),
        scene_segmenter=MockScene(), depth_estimator=MockDepthEstimator(),
    ).process_frame(
        np.zeros((2, 2, 3), dtype=np.uint8), 0,
        scene_output=_scene_options(tmp_path),
        depth_output=_depth_options(tmp_path),
        geometry_output=_geometry_options(tmp_path),
        bev_output=_bev_options(tmp_path),
    )
    assert result["bev"]["coordinate_frame"] == "camera_xz"
    assert result["bev"]["has_semantic_labels"] is True
    assert result["bev"]["observed_cell_count"] > 0
    assert result["bev"]["class_grid_path"].endswith("frame_000000.npy")
