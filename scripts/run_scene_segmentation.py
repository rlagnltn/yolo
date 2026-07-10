"""Run SegFormer scene semantic segmentation for a driving video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scene semantic segmentation.")
    parser.add_argument("--config", default="configs/scene_segmentation.yaml")
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--output")
    parser.add_argument("--model")
    parser.add_argument("--device")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-class-maps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-color-maps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-vis", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-regions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    from src.utils.io_utils import load_yaml

    config = load_yaml(args.config) if Path(args.config).exists() else {}
    model = config.get("model", {})
    output = config.get("output", {})
    runtime = config.get("runtime", {})
    visualization = config.get("visualization", {})
    regions = config.get("regions", {})
    choose = lambda cli, key, default: cli if cli is not None else runtime.get(key, default)
    return {
        "input_path": args.input_path or config.get("input", {}).get("source", "datasets/raw/sample.mp4"),
        "output_path": args.output or output.get("scene_json", "outputs/scene_segmentation/scene_segmentation.json"),
        "model_name": args.model or model.get("name", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"),
        "device": args.device or model.get("device", "auto"),
        "class_map_dir": output.get("class_map_dir", "outputs/scene_segmentation/class_maps"),
        "color_map_dir": output.get("color_map_dir", "outputs/scene_segmentation/color_maps"),
        "visualization_dir": output.get("visualization_dir", "outputs/scene_segmentation/visualizations"),
        "region_dir": output.get("region_dir", "outputs/scene_segmentation/regions"),
        "max_frames": args.max_frames if args.max_frames is not None else runtime.get("max_frames"),
        "save_class_maps": bool(choose(args.save_class_maps, "save_class_maps", True)),
        "save_color_maps": bool(choose(args.save_color_maps, "save_color_maps", True)),
        "save_visualizations": bool(choose(args.save_vis, "save_visualizations", True)),
        "save_regions": bool(choose(args.save_regions, "save_regions", True)),
        "continue_on_error": bool(choose(args.continue_on_error, "continue_on_error", True)),
        "alpha": float(visualization.get("alpha", 0.45)),
        "drivable_labels": set(regions.get("drivable_classes", ["road"])),
        "non_drivable_labels": set(regions.get("non_drivable_classes", [])) or None,
    }


def main() -> int:
    try:
        args = parse_args()
        settings = resolve_settings(args)
        input_path = Path(settings["input_path"])
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input source does not exist: {input_path}\n"
                "Place a driving video at datasets/raw/sample.mp4 or pass another path with --input."
            )
        from tqdm import tqdm

        from src.scene_segmentation import SceneSegmenter
        from src.scene_segmentation.output import build_scene_frame_result
        from src.utils.io_utils import save_json
        from src.utils.video_utils import get_video_info, is_video_file, iter_video_frames

        if not is_video_file(input_path):
            raise ValueError(f"Scene segmentation input must be a supported video: {input_path}")
        segmenter = SceneSegmenter(settings["model_name"], settings["device"])
        info = get_video_info(input_path)
        total = int(info["frame_count"])
        progress_total = min(total, settings["max_frames"]) if total and settings["max_frames"] else total
        frames = []
        for frame_index, timestamp_sec, frame in tqdm(
            iter_video_frames(input_path), total=progress_total or None, desc="Scene segmenting"
        ):
            if settings["max_frames"] is not None and frame_index >= settings["max_frames"]:
                break
            try:
                class_map = segmenter.predict(frame)
                result = build_scene_frame_result(
                    frame, class_map, segmenter.id2label, frame_index, timestamp_sec,
                    class_map_dir=settings["class_map_dir"], color_map_dir=settings["color_map_dir"],
                    visualization_dir=settings["visualization_dir"], region_dir=settings["region_dir"],
                    save_class_maps=settings["save_class_maps"], save_color_maps=settings["save_color_maps"],
                    save_visualizations=settings["save_visualizations"], save_regions=settings["save_regions"],
                    alpha=settings["alpha"],
                    drivable_labels=settings["drivable_labels"],
                    non_drivable_labels=settings["non_drivable_labels"],
                )
                result["errors"] = []
            except Exception as exc:
                if not settings["continue_on_error"]:
                    raise
                result = {
                    "frame_index": frame_index, "timestamp_sec": timestamp_sec,
                    "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                    "class_statistics": [], "errors": [f"scene_segmentation: {exc}"],
                }
            frames.append(result)
        payload = {
            "metadata": {
                "input": str(input_path), "model": segmenter.model_name,
                "frame_count": total, "processed_frame_count": len(frames),
                "fps": float(info["fps"]), "width": int(info["width"]), "height": int(info["height"]),
                "device": segmenter.device,
            },
            "frames": frames,
        }
        saved = save_json(payload, settings["output_path"])
        print(f"Saved scene segmentation JSON: {saved}")
        print(f"Processed frames: {len(frames)} on {segmenter.device}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
