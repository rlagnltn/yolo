import numpy as np

from src.perception.pipeline import PerceptionPipeline


class EmptyDetector:
    model_name = "det.pt"

    def detect_frame(self, frame):
        return []


class EmptyInstanceSegmenter:
    model_name = "inst.pt"

    def segment_frame(self, frame, *args, **kwargs):
        return []


class MockSceneSegmenter:
    model_name = "scene-model"
    id2label = {0: "road", 1: "sidewalk"}

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def predict(self, frame):
        self.calls += 1
        if self.fail:
            raise RuntimeError("scene failed")
        return np.zeros(frame.shape[:2], dtype=np.uint8)


def test_disabled_scene_keeps_scene_field_null():
    pipeline = PerceptionPipeline(EmptyDetector(), EmptyInstanceSegmenter())
    result = pipeline.process_frame(np.zeros((2, 3, 3), dtype=np.uint8), 0)
    assert result["scene_segmentation"] is None


def test_enabled_scene_uses_same_frame_once_and_adds_result(tmp_path):
    scene = MockSceneSegmenter()
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    pipeline = PerceptionPipeline(EmptyDetector(), EmptyInstanceSegmenter(), scene_segmenter=scene)
    result = pipeline.process_frame(
        frame, 0, scene_output={
            "class_map_dir": tmp_path, "color_map_dir": tmp_path,
            "visualization_dir": tmp_path, "region_dir": tmp_path,
            "save_class_maps": False, "save_color_maps": False, "save_regions": False,
        }
    )
    assert scene.calls == 1
    assert result["scene_segmentation"]["regions"]["drivable_pixel_ratio"] == 1.0


def test_scene_error_is_recorded_without_losing_object_results():
    pipeline = PerceptionPipeline(
        EmptyDetector(), EmptyInstanceSegmenter(), scene_segmenter=MockSceneSegmenter(fail=True)
    )
    result = pipeline.process_frame(np.zeros((2, 3, 3), dtype=np.uint8), 0)
    assert result["detections"] == []
    assert result["segments"] == []
    assert result["scene_segmentation"] is None
    assert result["errors"] == ["scene_segmentation: scene failed"]
