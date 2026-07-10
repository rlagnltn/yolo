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
    model_name = "depth-model"
    def __init__(self, fail=False):
        self.calls = 0
        self.frame = None
        self.fail = fail
    def predict(self, frame):
        self.calls += 1
        self.frame = frame
        if self.fail:
            raise RuntimeError("depth failed")
        return {"depth_map": np.ones(frame.shape[:2], dtype=np.float32),
                "depth_type": "metric", "unit": "meter", "model_name": self.model_name}


class MockScene:
    model_name = "scene"
    id2label = {0: "road", 1: "car"}
    def predict(self, frame):
        return np.array([[0, 0], [1, 1]], dtype=np.uint8)


def options(tmp_path):
    return {"raw_depth_dir": tmp_path, "depth_png_dir": tmp_path,
            "color_map_dir": tmp_path, "visualization_dir": tmp_path,
            "save_raw_depth": False, "save_depth_png": False,
            "save_color_maps": False, "save_visualizations": False}


def test_depth_disabled_preserves_pipeline_and_null_field():
    result = PerceptionPipeline(EmptyDetector(), EmptySegmenter()).process_frame(
        np.zeros((2, 2, 3), dtype=np.uint8), 0
    )
    assert result["depth"] is None


def test_depth_uses_same_frame_and_adds_scene_class_statistics(tmp_path):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = MockDepthEstimator()
    pipeline = PerceptionPipeline(
        EmptyDetector(), EmptySegmenter(), scene_segmenter=MockScene(), depth_estimator=depth
    )
    result = pipeline.process_frame(
        frame, 0,
        scene_output={"class_map_dir": tmp_path, "color_map_dir": tmp_path,
                      "visualization_dir": tmp_path, "region_dir": tmp_path,
                      "save_class_maps": False, "save_color_maps": False, "save_regions": False},
        depth_output=options(tmp_path),
    )
    assert depth.calls == 1
    assert depth.frame is frame
    assert result["depth"]["depth_by_scene_class"]["road"]["pixel_count"] == 2
    assert "depth_map" not in result["depth"]


def test_depth_error_uses_existing_continue_policy(tmp_path):
    pipeline = PerceptionPipeline(
        EmptyDetector(), EmptySegmenter(), depth_estimator=MockDepthEstimator(fail=True)
    )
    result = pipeline.process_frame(
        np.zeros((2, 2, 3), dtype=np.uint8), 0, depth_output=options(tmp_path)
    )
    assert result["depth"] is None
    assert result["errors"] == ["depth: depth failed"]


def test_geometry_disabled_preserves_pipeline(tmp_path):
    result = PerceptionPipeline(
        EmptyDetector(), EmptySegmenter(), depth_estimator=MockDepthEstimator()
    ).process_frame(
        np.zeros((2, 2, 3), dtype=np.uint8), 0, depth_output=options(tmp_path)
    )
    assert result["geometry"] is None


def test_geometry_uses_current_depth_and_semantic_class_map(tmp_path):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    pipeline = PerceptionPipeline(
        EmptyDetector(), EmptySegmenter(), scene_segmenter=MockScene(), depth_estimator=MockDepthEstimator()
    )
    result = pipeline.process_frame(
        frame, 0,
        scene_output={"class_map_dir": tmp_path, "color_map_dir": tmp_path,
                      "visualization_dir": tmp_path, "region_dir": tmp_path,
                      "save_class_maps": False, "save_color_maps": False, "save_regions": False},
        depth_output=options(tmp_path),
        geometry_output={
            "enabled": True,
            "point_cloud_dir": tmp_path,
            "stride": 1,
            "min_depth_m": 0.1,
            "max_depth_m": 80.0,
            "intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "width": 2, "height": 2},
        },
    )
    assert result["geometry"]["point_count"] == 4
    assert result["geometry"]["has_semantic_labels"] is True
    with np.load(result["geometry"]["point_cloud_path"]) as data:
        assert set(data.files) == {"points_xyz", "pixels_uv", "depth_values", "semantic_labels"}
        np.testing.assert_array_equal(data["semantic_labels"], [0, 0, 1, 1])
