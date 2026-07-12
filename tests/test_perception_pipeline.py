from src.perception.pipeline import PerceptionPipeline
from src.utils.visualization import draw_perception_overlay


class DummyFrame:
    shape = (20, 30, 3)

    def __init__(self, copied=False):
        self.copied = copied

    def copy(self):
        return DummyFrame(copied=True)


class DummyDetector:
    model_name = "detector.pt"

    def detect_frame(self, frame):
        return [{
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.9,
            "bbox_xyxy": [0, 0, 10, 10],
        }]


class DummySegmenter:
    model_name = "segmenter.pt"

    def segment_frame(self, frame, frame_index, mask_dir, save_masks, **kwargs):
        return [{
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.8,
            "bbox_xyxy": [0, 0, 10, 10],
            "mask_area": 100,
        }]


def test_process_frame_assigns_ids_and_fuses_mock_results():
    pipeline = PerceptionPipeline(DummyDetector(), DummySegmenter())
    result = pipeline.process_frame(DummyFrame(), 7, 0.25, save_masks=False)

    assert result["detections"][0]["object_id"] == "frame_000007_det_000"
    assert result["segments"][0]["segment_id"] == "frame_000007_seg_000"
    assert result["fused_objects"][0]["object_id"] == "frame_000007_obj_000"
    assert result["fused_objects"][0]["fusion_status"] == "matched"
    assert result["errors"] == []


def test_process_frame_records_model_error_and_continues():
    class BrokenDetector(DummyDetector):
        def detect_frame(self, frame):
            raise RuntimeError("broken")

    pipeline = PerceptionPipeline(BrokenDetector(), DummySegmenter())
    result = pipeline.process_frame(DummyFrame(), 0, save_masks=False)

    assert result["detections"] == []
    assert result["fused_objects"][0]["fusion_status"] == "segmentation_only"
    assert result["errors"] == ["detection: broken"]


def test_empty_perception_visualization_returns_frame_copy():
    result = draw_perception_overlay(DummyFrame(), [], [], [])
    assert result.copied is True


def test_process_video_supports_independent_frame_chunks(monkeypatch, tmp_path):
    import src.utils.video_utils as video_utils

    monkeypatch.setattr(video_utils, "get_video_info", lambda _: {
        "frame_count": 6, "fps": 10.0, "width": 30, "height": 20,
    })
    monkeypatch.setattr(video_utils, "iter_video_frames", lambda _: iter(
        (index, index / 10.0, DummyFrame()) for index in range(6)
    ))
    pipeline = PerceptionPipeline(DummyDetector(), DummySegmenter())
    result = pipeline.process_video(
        tmp_path / "dummy.mp4", start_frame=2, max_frames=2,
        save_masks=False, save_visualizations=False,
    )
    assert [frame["frame_index"] for frame in result["frames"]] == [2, 3]
    assert result["metadata"]["start_frame"] == 2
    assert result["metadata"]["end_frame_exclusive"] == 4
    assert result["metadata"]["processed_frame_count"] == 2
