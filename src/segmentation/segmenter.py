"""YOLO segmentation wrapper for images and video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.mask_utils import build_mask_path, mask_area, save_binary_mask


@dataclass(frozen=True)
class Segment:
    """Single segmentation record without embedding raw mask arrays in JSON."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    mask_area: int
    mask_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy": self.bbox_xyxy,
            "mask_area": self.mask_area,
        }
        if self.mask_path is not None:
            data["mask_path"] = self.mask_path
        return data


class YOLOSegmenter:
    """Thin wrapper around an Ultralytics YOLO segmentation model."""

    def __init__(
        self,
        model_name: str = "yolov8n-seg.pt",
        confidence_threshold: float = 0.25,
        device: str = "auto",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for YOLO segmentation. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = None if device == "auto" else device

        try:
            self.model = YOLO(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO segmentation model '{model_name}'. "
                "Check that the model name/path is valid and that weights can be downloaded "
                "or are available locally."
            ) from exc

    def segment_frame(
        self,
        frame: Any,
        frame_index: int = 0,
        mask_dir: str | Path | None = None,
        save_masks: bool = True,
    ) -> list[dict[str, Any]]:
        """Run segmentation on a BGR OpenCV frame."""

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or result.masks is None:
            return []

        names = result.names
        boxes = list(result.boxes)
        mask_tensors = result.masks.data
        segments: list[dict[str, Any]] = []

        for object_index, box in enumerate(boxes):
            if object_index >= len(mask_tensors):
                break

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].detach().cpu().tolist()
            binary_mask = self._to_frame_sized_mask(mask_tensors[object_index], frame)
            output_mask_path: Path | None = None

            if save_masks and mask_dir is not None:
                output_mask_path = build_mask_path(mask_dir, frame_index, object_index)
                save_binary_mask(binary_mask, output_mask_path)

            segment = Segment(
                class_id=class_id,
                class_name=str(names.get(class_id, class_id)),
                confidence=confidence,
                bbox_xyxy=[float(value) for value in xyxy],
                mask_area=mask_area(binary_mask),
                mask_path=str(output_mask_path) if output_mask_path is not None else None,
            ).to_dict()
            segments.append(segment)

        return segments

    def segment_image(
        self,
        image_path: str | Path,
        mask_dir: str | Path | None = None,
        save_masks: bool = True,
    ) -> dict[str, Any]:
        """Load an image and return segmentation results."""

        import cv2

        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        return {
            "input": str(image_path),
            "model": self.model_name,
            "type": "image",
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp_sec": 0.0,
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "segments": self.segment_frame(frame, 0, mask_dir, save_masks),
                }
            ],
        }

    @staticmethod
    def _to_frame_sized_mask(mask_tensor: Any, frame: Any) -> Any:
        import cv2
        import numpy as np

        mask = np.squeeze(mask_tensor.detach().cpu().numpy()) > 0.5
        frame_height, frame_width = frame.shape[:2]
        if mask.shape[:2] != (frame_height, frame_width):
            mask = cv2.resize(
                mask.astype("uint8"),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return np.asarray(mask).astype(bool)
