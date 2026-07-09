"""Run YOLO segmentation for an image or video source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO semantic segmentation.")
    parser.add_argument("--config", default="configs/segmentation.yaml", help="Segmentation YAML config path.")
    parser.add_argument("--input", dest="input_path", help="Input image or video path.")
    parser.add_argument("--output", help="Output segmentation JSON path.")
    parser.add_argument("--mask-dir", help="Directory for binary mask images.")
    parser.add_argument("--visualization-dir", help="Directory for segmentation overlays.")
    parser.add_argument("--model", help="YOLO segmentation model name or weight path.")
    parser.add_argument("--confidence", type=float, help="Segmentation confidence threshold.")
    parser.add_argument("--device", help="Device passed to Ultralytics, such as cpu, 0, cuda, or auto.")
    parser.add_argument("--save-vis", action="store_true", help="Save segmentation overlay visualizations.")
    parser.add_argument("--max-frames", type=int, help="Process only the first N frames.")
    parser.add_argument("--no-save-masks", action="store_true", help="Do not save per-object mask PNG files.")
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": {
            "name": "yolov8n-seg.pt",
            "confidence_threshold": 0.25,
            "device": "auto",
        },
        "input": {
            "source": "datasets/raw/sample.mp4",
        },
        "output": {
            "segmentation_json": "outputs/segmentations/segmentations.json",
            "mask_dir": "outputs/segmentations/masks",
            "visualization_dir": "outputs/segmentations/visualizations",
        },
        "runtime": {
            "max_frames": None,
            "save_masks": True,
            "save_visualizations": True,
        },
    }
    config_path = Path(args.config)
    if config_path.exists():
        try:
            from src.utils.io_utils import load_yaml

            loaded_config = load_yaml(config_path)
            config = _merge_dicts(config, loaded_config)
        except ModuleNotFoundError as exc:
            is_missing_yaml = exc.name == "yaml" or "PyYAML is required" in str(exc)
            if not is_missing_yaml or args.config != "configs/segmentation.yaml":
                raise

    model_config = config.get("model", {})
    input_config = config.get("input", {})
    output_config = config.get("output", {})
    runtime_config = config.get("runtime", {})

    return {
        "input_path": args.input_path or input_config.get("source"),
        "output_path": args.output or output_config.get("segmentation_json"),
        "mask_dir": args.mask_dir or output_config.get("mask_dir"),
        "visualization_dir": args.visualization_dir or output_config.get("visualization_dir"),
        "model_name": args.model or model_config.get("name", "yolov8n-seg.pt"),
        "confidence": args.confidence
        if args.confidence is not None
        else float(model_config.get("confidence_threshold", 0.25)),
        "device": args.device or model_config.get("device", "auto"),
        "max_frames": args.max_frames if args.max_frames is not None else runtime_config.get("max_frames"),
        "save_masks": False if args.no_save_masks else bool(runtime_config.get("save_masks", True)),
        "save_visualizations": args.save_vis or bool(runtime_config.get("save_visualizations", False)),
    }


def run_image(
    segmenter: Any,
    input_path: Path,
    output_path: Path,
    mask_dir: Path,
    visualization_dir: Path,
    save_masks: bool,
    save_vis: bool,
) -> Path:
    import cv2

    from src.utils.io_utils import save_json
    from src.utils.visualization import save_segmentation_overlay

    result = segmenter.segment_image(input_path, mask_dir, save_masks)

    if save_vis:
        frame = cv2.imread(str(input_path))
        save_segmentation_overlay(frame, result["frames"][0]["segments"], visualization_dir, 0)

    return save_json(result, output_path)


def run_video(
    segmenter: Any,
    input_path: Path,
    output_path: Path,
    mask_dir: Path,
    visualization_dir: Path,
    save_masks: bool,
    save_vis: bool,
    max_frames: int | None,
) -> Path:
    from tqdm import tqdm

    from src.utils.io_utils import save_json
    from src.utils.video_utils import get_video_info, iter_video_frames
    from src.utils.visualization import save_segmentation_overlay

    video_info = get_video_info(input_path)
    total = int(video_info.get("frame_count", 0))
    progress_total = min(total, max_frames) if total > 0 and max_frames else total
    frames = []

    for frame_index, timestamp_sec, frame in tqdm(
        iter_video_frames(input_path),
        total=progress_total if progress_total > 0 else None,
        desc="Segmenting",
    ):
        if max_frames is not None and frame_index >= max_frames:
            break

        segments = segmenter.segment_frame(frame, frame_index, mask_dir, save_masks)
        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "segments": segments,
            }
        )
        if save_vis:
            save_segmentation_overlay(frame, segments, visualization_dir, frame_index)

    result = {
        "input": str(input_path),
        "model": segmenter.model_name,
        "type": "video",
        "frames": frames,
    }
    return save_json(result, output_path)


def main() -> int:
    args = parse_args()
    try:
        settings = resolve_settings(args)
        input_path = Path(settings["input_path"])

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input source does not exist: {input_path}\n"
                "Place a driving video at datasets/raw/sample.mp4 or pass another path with --input."
            )

        from src.segmentation import YOLOSegmenter
        from src.utils.video_utils import is_image_file, is_video_file

        segmenter = YOLOSegmenter(
            model_name=settings["model_name"],
            confidence_threshold=settings["confidence"],
            device=settings["device"],
        )

        output_path = Path(settings["output_path"])
        mask_dir = Path(settings["mask_dir"])
        visualization_dir = Path(settings["visualization_dir"])

        if is_image_file(input_path):
            saved_path = run_image(
                segmenter,
                input_path,
                output_path,
                mask_dir,
                visualization_dir,
                settings["save_masks"],
                settings["save_visualizations"],
            )
        elif is_video_file(input_path):
            saved_path = run_video(
                segmenter,
                input_path,
                output_path,
                mask_dir,
                visualization_dir,
                settings["save_masks"],
                settings["save_visualizations"],
                settings["max_frames"],
            )
        else:
            raise ValueError(f"Unsupported input type: {input_path.suffix}")

        print(f"Saved segmentations: {saved_path}")
        if settings["save_masks"]:
            print(f"Saved masks: {mask_dir}")
        if settings["save_visualizations"]:
            print(f"Saved segmentation overlays: {visualization_dir}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
