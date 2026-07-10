import pytest

from src.geometry import CameraIntrinsics


def test_intrinsics_validate_and_to_dict():
    intrinsics = CameraIntrinsics(fx=900, fy=901, cx=640, cy=360, width=1280, height=720)
    assert intrinsics.to_dict() == {
        "fx": 900.0, "fy": 901.0, "cx": 640.0, "cy": 360.0,
        "width": 1280, "height": 720,
    }


def test_intrinsics_reject_invalid_focal_length():
    with pytest.raises(ValueError, match="focal"):
        CameraIntrinsics(fx=0, fy=900, cx=640, cy=360, width=1280, height=720).validate()


def test_intrinsics_reject_missing_config_values():
    with pytest.raises(ValueError, match="missing"):
        CameraIntrinsics.from_dict({"fx": None, "fy": 900, "cx": 1, "cy": 1, "width": 2, "height": 2})


def test_intrinsics_scaled_to_new_image_size():
    scaled = CameraIntrinsics(fx=100, fy=200, cx=50, cy=75, width=100, height=150).scaled_to(200, 300)
    assert scaled.fx == 200
    assert scaled.fy == 400
    assert scaled.cx == 100
    assert scaled.cy == 150
    assert scaled.width == 200
    assert scaled.height == 300
