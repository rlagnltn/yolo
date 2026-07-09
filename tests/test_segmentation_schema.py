from src.segmentation.segmenter import Segment
from src.utils.mask_utils import build_mask_path, mask_area
from src.utils.visualization import draw_segmentation_overlay


class DummyFrame:
    def __init__(self):
        self.copied = False

    def copy(self):
        copied = DummyFrame()
        copied.copied = True
        return copied


def test_segment_dict_uses_expected_schema_without_raw_mask():
    segment = Segment(
        class_id=0,
        class_name="person",
        confidence=0.91,
        bbox_xyxy=[100.0, 200.0, 300.0, 500.0],
        mask_area=12450,
        mask_path="outputs/segmentations/masks/frame_000000_obj_000.png",
    )

    assert segment.to_dict() == {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.91,
        "bbox_xyxy": [100.0, 200.0, 300.0, 500.0],
        "mask_area": 12450,
        "mask_path": "outputs/segmentations/masks/frame_000000_obj_000.png",
    }


def test_mask_area_counts_positive_pixels():
    assert mask_area([[0, 1], [True, False]]) == 2


def test_build_mask_path_uses_stable_names(tmp_path):
    assert build_mask_path(tmp_path, 7, 3) == tmp_path / "frame_000007_obj_003.png"


def test_draw_segmentation_overlay_handles_empty_segments_without_cv2():
    frame = DummyFrame()

    annotated = draw_segmentation_overlay(frame, [])

    assert annotated.copied is True
