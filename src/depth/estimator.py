"""Hugging Face monocular metric depth-estimation wrapper."""

from __future__ import annotations

from typing import Any

from .postprocessing import clip_depth_range, resize_depth_map, sanitize_depth_map

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-VKITTI-Small"
TRANSFORMERS_OUTDOOR_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
MODEL_ALIASES = {DEFAULT_MODEL: TRANSFORMERS_OUTDOOR_MODEL}


class DepthEstimator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        min_depth_m: float | None = None,
        max_depth_m: float | None = None,
        *,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for depth estimation. Install project dependencies.") from exc
        self.requested_model_name = model_name
        self.model_name = MODEL_ALIASES.get(model_name, model_name)
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device '{device}' was requested, but CUDA is not available.")
        else:
            self.device = str(device)
        if processor is None or model is None:
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                processor = AutoImageProcessor.from_pretrained(self.model_name)
                model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load depth model '{self.model_name}' (requested as '{model_name}'). "
                    "Check transformers compatibility, "
                    "network access, model name, and the local Hugging Face cache."
                ) from exc
        self.processor = processor
        self.model = model.to(self.device)
        self.model.eval()
        config = getattr(self.model, "config", None)
        self.depth_type = str(getattr(config, "depth_estimation_type", "metric")).lower()
        self.unit = "meter" if self.depth_type == "metric" else "relative"
        if self.depth_type != "metric":
            raise ValueError(
                f"Model '{self.model_name}' reports depth_estimation_type={self.depth_type!r}; metric depth is required."
            )

    def predict(self, frame: Any) -> dict[str, Any]:
        import cv2
        import numpy as np
        import torch

        if frame is None or getattr(frame, "ndim", 0) != 3:
            raise ValueError("Depth estimation requires a non-empty BGR image frame.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        predicted = getattr(outputs, "predicted_depth", None)
        if predicted is None:
            raise RuntimeError("Depth model output does not contain predicted_depth.")
        array = predicted.detach().float().cpu().numpy()
        array = np.squeeze(array)
        if array.ndim != 2:
            raise RuntimeError(f"Expected a 2D predicted depth map, got shape {array.shape}.")
        depth = resize_depth_map(array, frame.shape[0], frame.shape[1])
        depth = clip_depth_range(sanitize_depth_map(depth), self.min_depth_m, self.max_depth_m)
        return {
            "depth_map": depth.astype(np.float32, copy=False),
            "depth_type": self.depth_type,
            "unit": self.unit,
            "model_name": self.model_name,
            "requested_model_name": self.requested_model_name,
            "device": self.device,
        }
