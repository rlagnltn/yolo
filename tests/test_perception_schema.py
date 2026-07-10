import json

from src.perception.schemas import FrameRecord


def test_perception_frame_schema_is_json_serializable():
    frame: FrameRecord = {
        "frame_index": 0,
        "timestamp_sec": 0.0,
        "width": 1280,
        "height": 720,
        "detections": [],
        "segments": [],
        "fused_objects": [],
        "errors": [],
    }

    assert json.loads(json.dumps(frame)) == frame
