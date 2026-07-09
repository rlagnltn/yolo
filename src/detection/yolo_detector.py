"""YOLO object detector wrapper for images and video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Detection:
    """Single object detection record."""

    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


class YOLODetector:
    """Thin wrapper around an Ultralytics YOLO model."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: str = "auto",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for YOLO detection. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = None if device == "auto" else device
        self.model = YOLO(model_name)

    def detect_frame(self, frame: Any) -> list[dict[str, Any]]:
        """Run detection on a BGR OpenCV frame."""

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        detections: list[dict[str, Any]] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].detach().cpu().tolist()
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(names.get(class_id, class_id)),
                    confidence=confidence,
                    bbox=[float(value) for value in xyxy],
                ).to_dict()
            )

        return detections

    def detect_image(self, image_path: str | Path) -> dict[str, Any]:
        """Load an image and return detection results."""

        import cv2

        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        return {
            "source": str(image_path),
            "type": "image",
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp_sec": 0.0,
                    "detections": self.detect_frame(frame),
                }
            ],
        }
