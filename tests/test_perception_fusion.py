import pytest

from src.perception.fusion import calculate_iou, match_detections_and_segments


def detection(class_id=2, bbox=None):
    return {
        "object_id": "frame_000000_det_000",
        "class_id": class_id,
        "class_name": "car" if class_id == 2 else "person",
        "confidence": 0.92,
        "bbox_xyxy": bbox or [0, 0, 10, 10],
    }


def segment(class_id=2, bbox=None, segment_id="frame_000000_seg_000"):
    return {
        "segment_id": segment_id,
        "class_id": class_id,
        "class_name": "car" if class_id == 2 else "person",
        "confidence": 0.89,
        "bbox_xyxy": bbox or [0, 0, 10, 10],
        "mask_area": 100,
        "mask_path": "mask.png",
    }


def test_iou_for_identical_boxes_is_one():
    assert calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_for_disjoint_boxes_is_zero():
    assert calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_for_partial_overlap():
    assert calculate_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_same_class_above_threshold_is_matched():
    result = match_detections_and_segments([detection()], [segment()])
    assert [item["fusion_status"] for item in result] == ["matched"]


def test_different_class_is_not_matched_by_default():
    result = match_detections_and_segments([detection()], [segment(class_id=0)])
    assert {item["fusion_status"] for item in result} == {"detection_only", "segmentation_only"}


def test_low_iou_preserves_both_unmatched_results():
    result = match_detections_and_segments(
        [detection()], [segment(bbox=[20, 20, 30, 30])]
    )
    assert [item["fusion_status"] for item in result] == ["detection_only", "segmentation_only"]


def test_segment_is_not_matched_twice():
    result = match_detections_and_segments(
        [detection(), detection(bbox=[1, 1, 9, 9])], [segment()]
    )
    assert sum(item["fusion_status"] == "matched" for item in result) == 1
    assert sum(item["fusion_status"] == "detection_only" for item in result) == 1


def test_empty_inputs_return_empty_result():
    assert match_detections_and_segments([], []) == []
