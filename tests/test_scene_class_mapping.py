from src.scene_segmentation.class_mapping import classify_label, normalize_id2label


def test_scene_class_groups_are_safe_for_driving():
    assert classify_label("road") == "drivable"
    assert classify_label("sidewalk") == "pedestrian_surface"
    assert classify_label("building") == "static_obstacle"
    assert classify_label("vegetation") == "static_obstacle"
    assert classify_label("terrain") == "static_obstacle"
    assert classify_label("car") == "dynamic_object"
    assert classify_label("person") == "dynamic_object"
    assert classify_label("not-a-real-label") == "unknown"


def test_id2label_is_normalized_from_model_config():
    assert normalize_id2label({"0": "Road", 1: "Sidewalk"}) == {0: "road", 1: "sidewalk"}
