"""Run monocular metric depth estimation on a video."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monocular metric depth estimation.")
    parser.add_argument("--config", default="configs/depth.yaml")
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--output")
    parser.add_argument("--model")
    parser.add_argument("--device")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-depth-png", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-color-maps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-vis", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-depth", type=float)
    parser.add_argument("--max-depth", type=float)
    parser.add_argument("--depth-png-scale", type=float)
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    from src.depth import DEFAULT_MODEL
    from src.utils.io_utils import load_yaml

    config_path = Path(args.config)
    config = load_yaml(config_path) if config_path.exists() else {}
    model = config.get("model", {})
    input_config = config.get("input", {})
    output = config.get("output", {})
    runtime = config.get("runtime", {})
    depth = config.get("depth", {})
    visualization = config.get("visualization", {})
    choose = lambda cli, value, default: cli if cli is not None else value if value is not None else default
    return {
        "input_path": args.input_path or input_config.get("source", "datasets/raw/sample.mp4"),
        "output_path": args.output or output.get("depth_json", "outputs/depth/depth.json"),
        "model_name": args.model or model.get("name", DEFAULT_MODEL),
        "device": args.device or model.get("device", "auto"),
        "raw_depth_dir": output.get("raw_depth_dir", "outputs/depth/raw"),
        "depth_png_dir": output.get("depth_png_dir", "outputs/depth/depth_maps"),
        "color_map_dir": output.get("color_map_dir", "outputs/depth/color_maps"),
        "visualization_dir": output.get("visualization_dir", "outputs/depth/visualizations"),
        "max_frames": choose(args.max_frames, runtime.get("max_frames"), None),
        "save_raw_depth": bool(choose(args.save_raw, runtime.get("save_raw_depth"), True)),
        "save_depth_png": bool(choose(args.save_depth_png, runtime.get("save_depth_png"), True)),
        "save_color_maps": bool(choose(args.save_color_maps, runtime.get("save_color_maps"), True)),
        "save_visualizations": bool(choose(args.save_vis, runtime.get("save_visualizations"), True)),
        "continue_on_error": bool(choose(args.continue_on_error, runtime.get("continue_on_error"), True)),
        "min_depth_m": choose(args.min_depth, depth.get("min_depth_m"), None),
        "max_depth_m": choose(args.max_depth, depth.get("max_depth_m"), None),
        "png_scale": float(choose(args.depth_png_scale, depth.get("png_scale"), 1000.0)),
        "alpha": float(visualization.get("alpha", 0.45)),
        "percentile_min": float(visualization.get("percentile_min", 2.0)),
        "percentile_max": float(visualization.get("percentile_max", 98.0)),
    }


def main() -> int:
    try:
        args = parse_args()
        settings = resolve_settings(args)
        input_path = Path(settings["input_path"])
        if not input_path.exists():
            raise FileNotFoundError(f"Input source does not exist: {input_path}")
        from src.depth import DepthEstimator
        from src.depth.output import build_depth_frame_result
        from src.utils.io_utils import save_json
        from src.utils.video_utils import get_video_info, is_video_file, iter_video_frames

        if not is_video_file(input_path):
            raise ValueError(f"Depth input must be a supported video: {input_path}")
        estimator = DepthEstimator(
            settings["model_name"], settings["device"],
            settings["min_depth_m"], settings["max_depth_m"],
        )
        info = get_video_info(input_path)
        frames: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        started = time.perf_counter()
        for frame_index, timestamp_sec, frame in iter_video_frames(input_path):
            if settings["max_frames"] is not None and frame_index >= settings["max_frames"]:
                break
            try:
                prediction = estimator.predict(frame)
                frames.append(build_depth_frame_result(
                    frame, prediction, frame_index, timestamp_sec,
                    raw_depth_dir=settings["raw_depth_dir"],
                    depth_png_dir=settings["depth_png_dir"],
                    color_map_dir=settings["color_map_dir"],
                    visualization_dir=settings["visualization_dir"],
                    save_raw_depth=settings["save_raw_depth"],
                    save_depth_png=settings["save_depth_png"],
                    save_color_maps=settings["save_color_maps"],
                    save_visualizations=settings["save_visualizations"],
                    png_scale=settings["png_scale"], alpha=settings["alpha"],
                    percentile_min=settings["percentile_min"],
                    percentile_max=settings["percentile_max"],
                ))
            except Exception as exc:
                errors.append({"frame_index": int(frame_index), "error": str(exc)})
                if not settings["continue_on_error"]:
                    raise
        elapsed = time.perf_counter() - started
        result = {
            "metadata": {
                "input": str(input_path), "model_name": estimator.model_name,
                "depth_type": estimator.depth_type, "unit": estimator.unit,
                "device": estimator.device, "frame_count": int(info["frame_count"]),
                "processed_frame_count": len(frames), "fps": float(info["fps"]),
                "width": int(info["width"]), "height": int(info["height"]),
                "processing_time_sec": float(elapsed),
                "average_processing_fps": float(len(frames) / elapsed) if elapsed else 0.0,
            },
            "frames": frames, "errors": errors,
        }
        saved = save_json(result, settings["output_path"])
        print(f"Saved depth JSON: {saved}")
        print(f"Processed frames: {len(frames)}; errors: {len(errors)}")
        print(f"Processing time: {elapsed:.3f}s; average FPS: {result['metadata']['average_processing_fps']:.3f}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
