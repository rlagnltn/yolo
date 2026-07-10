import json

from src.scene_segmentation.schemas import SceneFrameRecord


def test_scene_result_is_json_serializable_and_has_required_summary_fields():
    result: SceneFrameRecord = {
        "frame_index": 0,
        "timestamp_sec": 0.0,
        "width": 2,
        "height": 2,
        "class_map_path": "class.png",
        "class_statistics": [{
            "class_id": 0, "class_name": "road", "pixel_count": 4, "pixel_ratio": 1.0,
        }],
        "regions": {
            "drivable_pixel_count": 4, "drivable_pixel_ratio": 1.0,
            "non_drivable_pixel_count": 0, "unknown_pixel_count": 0,
        },
    }
    assert json.loads(json.dumps(result)) == result
    assert "class_map" not in result
