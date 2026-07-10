import numpy as np
import pytest

from src.geometry import CameraIntrinsics, attach_semantic_labels, backproject_depth, save_point_cloud_npz


def test_center_pixel_projects_to_zero_xy():
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[1, 1] = 2.0
    intrinsics = CameraIntrinsics(fx=10, fy=10, cx=1, cy=1, width=3, height=3)
    cloud = backproject_depth(depth, intrinsics)
    np.testing.assert_allclose(cloud["points_xyz"], [[0.0, 0.0, 2.0]])
    np.testing.assert_array_equal(cloud["pixels_uv"], [[1, 1]])


def test_known_pixel_projects_with_pinhole_formula():
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[2, 0] = 4.0
    intrinsics = CameraIntrinsics(fx=2, fy=4, cx=1, cy=1, width=3, height=3)
    cloud = backproject_depth(depth, intrinsics)
    np.testing.assert_allclose(cloud["points_xyz"], [[-2.0, 1.0, 4.0]])


def test_invalid_depths_are_excluded():
    depth = np.array([[np.nan, np.inf], [0.0, -1.0]], dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=1, fy=1, cx=0, cy=0, width=2, height=2)
    cloud = backproject_depth(depth, intrinsics)
    assert cloud["points_xyz"].shape == (0, 3)


def test_min_max_and_stride_filtering():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    depth[0, 0] = 0.5
    depth[2, 2] = 9.0
    intrinsics = CameraIntrinsics(fx=1, fy=1, cx=0, cy=0, width=4, height=4)
    cloud = backproject_depth(depth, intrinsics, stride=2, min_depth_m=1.0, max_depth_m=5.0)
    np.testing.assert_array_equal(cloud["pixels_uv"], [[2, 0], [0, 2]])


def test_valid_mask_and_empty_cloud():
    depth = np.ones((2, 2), dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=1, fy=1, cx=0, cy=0, width=2, height=2)
    cloud = backproject_depth(depth, intrinsics, valid_mask=np.zeros((2, 2), dtype=bool))
    assert cloud["depth_values"].dtype == np.float32
    assert cloud["pixels_uv"].dtype == np.int32
    assert cloud["points_xyz"].shape == (0, 3)


def test_backprojection_rejects_bad_stride_and_size_mismatch():
    intrinsics = CameraIntrinsics(fx=1, fy=1, cx=0, cy=0, width=2, height=2)
    with pytest.raises(ValueError, match="stride"):
        backproject_depth(np.ones((2, 2), dtype=np.float32), intrinsics, stride=0)
    with pytest.raises(ValueError, match="does not match"):
        backproject_depth(np.ones((3, 2), dtype=np.float32), intrinsics)


def test_attach_semantic_labels_and_bounds_check():
    pixels = np.array([[0, 0], [1, 1]], dtype=np.int32)
    class_map = np.array([[3, 4], [5, 6]], dtype=np.uint8)
    np.testing.assert_array_equal(attach_semantic_labels(pixels, class_map), [3, 6])
    with pytest.raises(ValueError, match="outside"):
        attach_semantic_labels(np.array([[2, 0]], dtype=np.int32), class_map)


def test_save_point_cloud_npz_roundtrip(tmp_path):
    output = tmp_path / "frame_000000.npz"
    save_point_cloud_npz(
        output,
        np.array([[1, 2, 3]], dtype=np.float32),
        np.array([[4, 5]], dtype=np.int32),
        np.array([3], dtype=np.float32),
        np.array([7], dtype=np.int32),
    )
    with np.load(output) as data:
        assert set(data.files) == {"points_xyz", "pixels_uv", "depth_values", "semantic_labels"}
        np.testing.assert_array_equal(data["semantic_labels"], [7])
