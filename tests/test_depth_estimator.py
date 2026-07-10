from types import SimpleNamespace

import numpy as np
import torch

from src.depth.estimator import DepthEstimator


class MockProcessor:
    def __init__(self):
        self.image = None

    def __call__(self, *, images, return_tensors):
        self.image = images
        return {"pixel_values": torch.ones((1, 3, 2, 2))}


class MockModel:
    config = SimpleNamespace(depth_estimation_type="metric")

    def __init__(self):
        self.device = None
        self.calls = 0

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        self.calls += 1
        return SimpleNamespace(predicted_depth=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))


def test_mock_estimator_converts_bgr_to_rgb_selects_cpu_and_resizes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    processor, model = MockProcessor(), MockModel()
    estimator = DepthEstimator("mock-metric", "auto", processor=processor, model=model)
    frame = np.zeros((3, 5, 3), dtype=np.uint8)
    frame[0, 0] = [10, 20, 30]
    result = estimator.predict(frame)
    assert processor.image[0, 0].tolist() == [30, 20, 10]
    assert result["depth_map"].shape == (3, 5)
    assert result["depth_map"].dtype == np.float32
    assert result["depth_type"] == "metric"
    assert result["unit"] == "meter"
    assert estimator.device == "cpu"
    assert model.calls == 1
