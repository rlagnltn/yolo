"""Run unified YOLO detection and instance segmentation on one video pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the unified perception pipeline.")
    parser.add_argument("--config", default="configs/perception.yaml")
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--output")
    parser.add_argument("--save-vis", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-masks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device")
    parser.add_argument("--iou-threshold", type=float)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-scene-segmentation", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    from src.utils.io_utils import load_yaml

    defaults: dict[str, Any] = {
        "models": {
            "detection": {"name": "yolov8n.pt", "confidence_threshold": 0.25},
            "instance_segmentation": {"enabled": True, "name": "yolov8n-seg.pt", "confidence_threshold": 0.25},
            "scene_segmentation": {"enabled": False, "name": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"},
            "device": "auto",
        },
        "fusion": {"iou_threshold": 0.5, "require_same_class": True},
        "input": {"source": "datasets/raw/sample.mp4"},
        "output": {
            "perception_json": "outputs/perception/perception.json",
            "mask_dir": "outputs/perception/masks",
            "visualization_dir": "outputs/perception/visualizations",
            "scene": {
                "class_map_dir": "outputs/perception/scene/class_maps",
                "color_map_dir": "outputs/perception/scene/color_maps",
                "region_dir": "outputs/perception/scene/regions",
            },
        },
        "runtime": {
            "max_frames": None, "save_masks": True, "save_visualizations": True,
            "save_scene_class_maps": True, "save_scene_color_maps": True,
            "save_scene_regions": True, "continue_on_error": True,
        },
    }
    config_path = Path(args.config)
    config = _merge(defaults, load_yaml(config_path)) if config_path.exists() else defaults
    models, fusion, output, runtime = config["models"], config["fusion"], config["output"], config["runtime"]
    instance_model = models.get("instance_segmentation", models.get("segmentation", {}))
    scene_model = models.get("scene_segmentation", {})
    scene_output = output.get("scene", {})
    return {
        "input_path": args.input_path or config["input"]["source"],
        "output_path": args.output or output["perception_json"],
        "mask_dir": output["mask_dir"],
        "visualization_dir": output["visualization_dir"],
        "detection_model": models["detection"]["name"],
        "detection_confidence": float(models["detection"]["confidence_threshold"]),
        "segmentation_model": instance_model["name"],
        "segmentation_confidence": float(instance_model["confidence_threshold"]),
        "scene_enabled": (
            args.enable_scene_segmentation
            if args.enable_scene_segmentation is not None else bool(scene_model.get("enabled", False))
        ),
        "scene_model": scene_model.get("name", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"),
        "scene_class_map_dir": scene_output.get("class_map_dir", "outputs/perception/scene/class_maps"),
        "scene_color_map_dir": scene_output.get("color_map_dir", "outputs/perception/scene/color_maps"),
        "scene_region_dir": scene_output.get("region_dir", "outputs/perception/scene/regions"),
        "save_scene_class_maps": bool(runtime.get("save_scene_class_maps", True)),
        "save_scene_color_maps": bool(runtime.get("save_scene_color_maps", True)),
        "save_scene_regions": bool(runtime.get("save_scene_regions", True)),
        "device": args.device or models["device"],
        "iou_threshold": args.iou_threshold if args.iou_threshold is not None else float(fusion["iou_threshold"]),
        "require_same_class": bool(fusion["require_same_class"]),
        "max_frames": args.max_frames if args.max_frames is not None else runtime["max_frames"],
        "save_masks": args.save_masks if args.save_masks is not None else bool(runtime["save_masks"]),
        "save_visualizations": args.save_vis if args.save_vis is not None else bool(runtime["save_visualizations"]),
        "continue_on_error": args.continue_on_error if args.continue_on_error is not None else bool(runtime["continue_on_error"]),
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

        from src.detection import YOLODetector
        from src.perception import PerceptionPipeline
        from src.segmentation import YOLOSegmenter
        from src.utils.io_utils import save_json
        from src.utils.video_utils import is_video_file

        if not is_video_file(input_path):
            raise ValueError(f"Perception input must be a supported video: {input_path}")
        detector = YOLODetector(settings["detection_model"], settings["detection_confidence"], settings["device"])
        segmenter = YOLOSegmenter(settings["segmentation_model"], settings["segmentation_confidence"], settings["device"])
        scene_segmenter = None
        if settings["scene_enabled"]:
            from src.scene_segmentation import SceneSegmenter

            scene_segmenter = SceneSegmenter(settings["scene_model"], settings["device"])
        pipeline = PerceptionPipeline(
            detector, segmenter,
            scene_segmenter=scene_segmenter,
            iou_threshold=settings["iou_threshold"],
            require_same_class=settings["require_same_class"],
            continue_on_error=settings["continue_on_error"],
        )
        result = pipeline.process_video(
            input_path,
            mask_dir=settings["mask_dir"],
            visualization_dir=settings["visualization_dir"],
            save_masks=settings["save_masks"],
            save_visualizations=settings["save_visualizations"],
            max_frames=settings["max_frames"],
            scene_output={
                "class_map_dir": settings["scene_class_map_dir"],
                "color_map_dir": settings["scene_color_map_dir"],
                "region_dir": settings["scene_region_dir"],
                "save_class_maps": settings["save_scene_class_maps"],
                "save_color_maps": settings["save_scene_color_maps"],
                "save_regions": settings["save_scene_regions"],
            },
        )
        saved_path = save_json(result, settings["output_path"])
        print(f"Saved perception JSON: {saved_path}")
        print(f"Processed frames: {result['metadata']['processed_frame_count']}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
