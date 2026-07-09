import json

from src.utils.io_utils import ensure_dir, save_json


def test_ensure_dir_creates_nested_directory(tmp_path):
    target = tmp_path / "outputs" / "detections"

    created = ensure_dir(target)

    assert created == target
    assert target.is_dir()


def test_save_json_creates_parent_directory_and_writes_data(tmp_path):
    output_path = tmp_path / "nested" / "detections.json"
    data = {"frames": [{"frame_index": 0, "objects": []}]}

    saved_path = save_json(data, output_path)

    assert saved_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == data
