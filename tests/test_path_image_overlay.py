import cv2
import numpy as np

from src.bev import BEVConfig
from src.planner import draw_projected_path, merge_perception_chunks, path_cells_to_pixels, render_path_overlay_video


def _bev():
    return BEVConfig(0, 4, 0, 4, 1)


def test_path_cells_to_pixels_uses_nearest_point_per_bev_cell():
    path = np.asarray([[3, 1], [2, 1]], np.int32)
    points = np.asarray([
        [1.5, 0.0, .5], [1.5, 0.0, .5], [1.5, 0.0, 1.5], [3.5, 0.0, 3.5],
    ], np.float32)
    pixels = np.asarray([[11, 21], [12, 22], [13, 23], [30, 30]], np.int32)
    # The first point is nearest to the camera in the first cell.
    projected = path_cells_to_pixels(path, points, pixels, _bev())
    assert projected == [(11, 21), (13, 23)]


def test_draw_projected_path_does_not_bridge_missing_cells():
    frame = np.zeros((40, 40, 3), np.uint8)
    drawn = draw_projected_path(frame, [(5, 20), None, (35, 20)], thickness=2)
    assert tuple(drawn[20, 20]) == (0, 0, 0)
    assert drawn[20, 5].any() and drawn[20, 35].any()


def _write_video(path, count=3):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    if not writer.isOpened():
        raise RuntimeError("mp4v video writer unavailable")
    for _ in range(count):
        writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.release()


def test_video_renderer_draws_new_path_only_and_preserves_chunk_range(tmp_path):
    source = tmp_path / "input.mp4"
    output = tmp_path / "overlay.mp4"
    _write_video(source)
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    path_file = asset_dir / "path.npy"
    cloud_file = asset_dir / "cloud.npz"
    np.save(path_file, np.asarray([[3, 1], [2, 1]], np.int32), allow_pickle=False)
    np.savez_compressed(
        cloud_file,
        points_xyz=np.asarray([[1.5, 0.0, .5], [1.5, 0.0, 1.5]], np.float32),
        pixels_uv=np.asarray([[10, 20], [15, 15]], np.int32),
    )
    bev = {"x_range_m": [0, 4], "z_range_m": [0, 4], "resolution_m": 1.0}
    perception = {
        "metadata": {"start_frame": 1, "end_frame_exclusive": 3, "experimental_intrinsics": False},
        "frames": [
            {"frame_index": 1, "bev": bev, "geometry": {"point_cloud_path": "assets/cloud.npz"},
             "planner": {"path_source": "new", "reached_goal": True, "grid_path_path": "assets/path.npy"}},
            {"frame_index": 2, "bev": bev, "geometry": {"point_cloud_path": "assets/cloud.npz"},
             "planner": {"path_source": "reused", "reached_goal": True, "grid_path_path": "assets/path.npy"}},
        ],
    }
    summary = render_path_overlay_video(source, perception, output, repository_root=tmp_path, speed=.3)
    assert summary["rendered_frame_count"] == 2
    assert summary["successful_new_path_frame_count"] == 1
    assert summary["path_drawn_frame_count"] == 1
    assert summary["output_fps"] == 3.0
    capture = cv2.VideoCapture(str(output))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    ok, first = capture.read()
    assert ok and first.sum() > 0
    ok, second = capture.read()
    capture.release()
    assert ok and second.sum() < first.sum()


def test_merge_perception_chunks_preserves_range_and_rejects_overlap():
    first = {"metadata": {"experimental_intrinsics": False}, "frames": [{"frame_index": 0}, {"frame_index": 1}]}
    second = {"metadata": {"experimental_intrinsics": True}, "frames": [{"frame_index": 2}]}
    merged = merge_perception_chunks([first, second])
    assert [frame["frame_index"] for frame in merged["frames"]] == [0, 1, 2]
    assert merged["metadata"]["start_frame"] == 0
    assert merged["metadata"]["end_frame_exclusive"] == 3
    assert merged["metadata"]["experimental_intrinsics"] is True
    import pytest
    with pytest.raises(ValueError, match="overlap"):
        merge_perception_chunks([first, {"frames": [{"frame_index": 1}]}])
