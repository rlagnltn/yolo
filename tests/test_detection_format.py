from src.detection.yolo_detector import Detection


def test_detection_dict_uses_expected_schema():
    detection = Detection(
        class_id=2,
        class_name="car",
        confidence=0.91,
        bbox_xyxy=[100.0, 200.0, 300.0, 400.0],
    )

    assert detection.to_dict() == {
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.91,
        "bbox_xyxy": [100.0, 200.0, 300.0, 400.0],
    }
