"""Hugging Face SegFormer scene semantic-segmentation wrapper."""

from __future__ import annotations

from typing import Any

from .class_mapping import normalize_id2label
from .postprocessing import logits_to_class_map

DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


class SceneSegmenter:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        *,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for scene segmentation. Install project dependencies.") from exc

        self.model_name = model_name
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device '{device}' was requested, but CUDA is not available.")
        else:
            self.device = device

        if processor is None or model is None:
            try:
                from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

                processor = AutoImageProcessor.from_pretrained(model_name)
                model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load scene-segmentation model '{model_name}'. "
                    "Check network access, model name, and local Hugging Face cache."
                ) from exc
        self.processor = processor
        self.model = model.to(self.device)
        self.model.eval()
        self.id2label = normalize_id2label(self.model.config.id2label)

    def predict(self, frame: Any) -> Any:
        """Return a class-ID map with the same height and width as a BGR frame."""

        import cv2
        import torch

        if frame is None or getattr(frame, "ndim", 0) != 3:
            raise ValueError("Scene segmentation requires a non-empty BGR image frame.")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb_frame, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return logits_to_class_map(outputs.logits, frame.shape[0], frame.shape[1])
