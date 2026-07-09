"""Run YOLO detection for an image or video source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO object detection.")
    parser.add_argument("--config", default="configs/detection.yaml", help="Detection YAML config path.")
    parser.add_argument("--input", dest="input_path", help="Input image or video path.")
    parser.add_argument("--output", help="Output detection JSON path.")
    parser.add_argument("--visualization-dir", help="Directory for annotated frames/images.")
    parser.add_argument("--model", help="YOLO model name or weight path.")
    parser.add_argument("--confidence", type=float, help="Detection confidence threshold.")
    parser.add_argument("--device", help="Device passed to Ultralytics, such as cpu, 0, or auto.")
    parser.add_argument("--save-vis", action="store_true", help="Save annotated visualizations.")
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    from src.utils.io_utils import load_yaml

    config = load_yaml(args.config)
    model_config = config.get("model", {})
    input_config = config.get("input", {})
    output_config = config.get("output", {})

    return {
        "input_path": args.input_path or input_config.get("source"),
        "output_path": args.output or output_config.get("detection_json", "outputs/detections/detections.json"),
        "visualization_dir": args.visualization_dir
        or output_config.get("visualization_dir", "outputs/visualizations"),
        "model_name": args.model or model_config.get("name", "yolov8n.pt"),
        "confidence": args.confidence
        if args.confidence is not None
        else float(model_config.get("confidence_threshold", 0.25)),
        "device": args.device or model_config.get("device", "auto"),
    }


def run_image(
    detector: Any,
    input_path: Path,
    output_path: Path,
    visualization_dir: Path,
    save_vis: bool,
) -> Path:
    import cv2

    from src.utils.io_utils import save_json
    from src.utils.visualization import save_annotated_frame

    result = detector.detect_image(input_path)

    if save_vis:
        frame = cv2.imread(str(input_path))
        save_annotated_frame(frame, result["frames"][0]["detections"], visualization_dir, 0)

    return save_json(result, output_path)


def run_video(
    detector: Any,
    input_path: Path,
    output_path: Path,
    visualization_dir: Path,
    save_vis: bool,
) -> Path:
    from tqdm import tqdm

    from src.utils.io_utils import save_json
    from src.utils.video_utils import get_video_info, iter_video_frames
    from src.utils.visualization import save_annotated_frame

    video_info = get_video_info(input_path)
    frames = []
    total = int(video_info.get("frame_count", 0))

    for frame_index, timestamp_sec, frame in tqdm(
        iter_video_frames(input_path),
        total=total if total > 0 else None,
        desc="Detecting",
    ):
        detections = detector.detect_frame(frame)
        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "detections": detections,
            }
        )
        if save_vis:
            save_annotated_frame(frame, detections, visualization_dir, frame_index)

    result = {
        "source": str(input_path),
        "type": "video",
        "video_info": video_info,
        "frames": frames,
    }
    return save_json(result, output_path)


def main() -> int:
    args = parse_args()
    settings = resolve_settings(args)
    input_path = Path(settings["input_path"])

    if not input_path.exists():
        raise FileNotFoundError(f"Input source does not exist: {input_path}")

    from src.detection import YOLODetector
    from src.utils.video_utils import is_image_file, is_video_file

    detector = YOLODetector(
        model_name=settings["model_name"],
        confidence_threshold=settings["confidence"],
        device=settings["device"],
    )

    output_path = Path(settings["output_path"])
    visualization_dir = Path(settings["visualization_dir"])

    if is_image_file(input_path):
        saved_path = run_image(detector, input_path, output_path, visualization_dir, args.save_vis)
    elif is_video_file(input_path):
        saved_path = run_video(detector, input_path, output_path, visualization_dir, args.save_vis)
    else:
        raise ValueError(f"Unsupported input type: {input_path.suffix}")

    print(f"Saved detections: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
