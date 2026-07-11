"""Project command-line entry points."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import cv2


def _boolean(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    video = commands.add_parser("video-plan", help="Run streaming image-space planning overlays.")
    video.add_argument("--input", required=True); video.add_argument("--output", required=True)
    video.add_argument("--goal-x", type=float, required=True); video.add_argument("--goal-y", type=float, required=True)
    video.add_argument("--normalized-goal", action="store_true")
    video.add_argument("--start-x", type=float); video.add_argument("--start-y", type=float); video.add_argument("--normalized-start", action="store_true")
    video.add_argument("--metadata"); video.add_argument("--config", default="configs/perception.yaml")
    video.add_argument("--potential-alpha", type=float, default=.4); video.add_argument("--trajectory-alpha", type=float, default=.5)
    video.add_argument("--max-reuse-frames", type=int, default=3); video.add_argument("--codec", default="mp4v")
    video.add_argument("--max-frames", type=int); video.add_argument("--frame-stride", type=int, default=1)
    video.add_argument("--heatmap-alpha", type=float, default=.35); video.add_argument("--fail-fast", action="store_true")
    _boolean(video, "--potential-smoothing", True); _boolean(video, "--trajectory-stabilization", True); _boolean(video, "--reuse-previous", True)
    _boolean(video, "--show-detections", True); _boolean(video, "--show-potential", True); _boolean(video, "--show-occupancy", False)
    _boolean(video, "--show-raw-path", True); _boolean(video, "--show-trajectory", True)
    return parser


def _video_size(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened(): raise FileNotFoundError(f"Could not open input video: {path}")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)); capture.release()
    return width, height


def _image_xy(args: argparse.Namespace, width: int, height: int, prefix: str) -> tuple[int, int]:
    from src.video_plan import normalized_to_pixel
    x, y, normalized = getattr(args, f"{prefix}_x"), getattr(args, f"{prefix}_y"), getattr(args, f"normalized_{prefix}")
    if prefix == "start" and x is None and y is None: return width // 2, height - 1
    if x is None or y is None: raise ValueError(f"Both {prefix}-x and {prefix}-y are required.")
    if normalized: return normalized_to_pixel(x, y, width, height)
    if not (0 <= x < width and 0 <= y < height): raise ValueError(f"{prefix} pixel coordinate is outside the video.")
    return round(x), round(y)


def _build_pipeline(args: argparse.Namespace, width: int, height: int):
    from src.bev import BEVConfig
    from src.detection import YOLODetector
    from src.perception import PerceptionPipeline
    from src.segmentation import YOLOSegmenter
    from src.utils.io_utils import load_yaml
    from src.video_plan import pixel_to_grid

    root = load_yaml(args.config); models = root["models"]; device = models.get("device", "auto")
    detector = YOLODetector(models["detection"]["name"], float(models["detection"].get("confidence_threshold", .25)), device)
    instance = models.get("instance_segmentation", models.get("segmentation")); segmenter = YOLOSegmenter(instance["name"], float(instance.get("confidence_threshold", .25)), device)
    scene_cfg, depth_cfg = models.get("scene_segmentation", {}), models.get("depth", {})
    if not (scene_cfg.get("enabled") and depth_cfg.get("enabled") and root.get("geometry", {}).get("enabled") and root.get("bev", {}).get("enabled") and root.get("mapping", {}).get("enabled") and root.get("potential", {}).get("enabled") and root.get("planner", {}).get("enabled")):
        raise ValueError("video-plan requires scene segmentation, depth, geometry, BEV, mapping, potential, and planner enabled in the perception config.")
    from src.scene_segmentation import SceneSegmenter
    from src.depth import DepthEstimator
    scene, depth = SceneSegmenter(scene_cfg["name"], device), DepthEstimator(depth_cfg["name"], device, None, None)
    pipeline = PerceptionPipeline(detector, segmenter, scene_segmenter=scene, depth_estimator=depth, continue_on_error=not args.fail_fast)
    bev_doc=load_yaml(root["bev"]["config"]); bev=BEVConfig.from_dict(bev_doc["bev"])
    start=pixel_to_grid(*_image_xy(args,width,height,"start"),bev.shape,width,height); goal=pixel_to_grid(*_image_xy(args,width,height,"goal"),bev.shape,width,height)
    geometry=root["geometry"]; camera=load_yaml(geometry.get("camera_config","configs/camera.yaml"))["camera"]
    mapping_doc=load_yaml(root["mapping"]["config"]); potential_doc=load_yaml(root["potential"]["config"]); planner_doc=load_yaml(root["planner"]["config"])
    trajectory_cfg=root.get("trajectory", {"config":"configs/trajectory.yaml","output":{}}); trajectory_doc=load_yaml(trajectory_cfg["config"])
    outputs=root.get("output", {}); scene_out=outputs.get("scene", {}); depth_out=outputs.get("depth", {})
    frame_options={
      "save_masks":False,
      "scene_output":{**scene_out,"save_class_maps":False,"save_color_maps":False,"save_regions":False},
      "depth_output":{**depth_out,"save_raw_depth":False,"save_depth_png":False,"save_color_maps":False,"save_visualizations":False},
      "geometry_output":{"enabled":True,"intrinsics":camera,"point_cloud_dir":geometry.get("point_cloud_dir","outputs/perception/geometry/point_clouds"),"stride":geometry.get("stride",4),"min_depth_m":geometry.get("min_depth_m"),"max_depth_m":geometry.get("max_depth_m")},
      "bev_output":{"enabled":True,"config":bev_doc["bev"],"id2label":scene.id2label,**bev_doc.get("runtime",{}),**root["bev"].get("output",{})},
      "mapping_output":{"enabled":True,"config":mapping_doc,**mapping_doc.get("runtime",{}),**root["mapping"].get("output",{})},
      "potential_output":{"enabled":True,"config":potential_doc,"goal":{"row":goal[0],"col":goal[1]},**potential_doc.get("runtime",{}),**root["potential"].get("output",{})},
      "planner_output":{"enabled":True,"config":planner_doc,"start":{"row":start[0],"col":start[1]},**planner_doc.get("runtime",{}),**root["planner"].get("output",{})},
      "trajectory_output":{"enabled":True,"config":trajectory_doc,**trajectory_doc.get("runtime",{}),**trajectory_cfg.get("output",{})},
    }
    return pipeline, frame_options


def video_plan(args: argparse.Namespace) -> int:
    from src.video_runner import VideoPlanOptions, run_video_plan
    input_path, output_path = Path(args.input), Path(args.output)
    width,height=_video_size(input_path); pipeline,frame_options=_build_pipeline(args,width,height)
    options=VideoPlanOptions(codec=args.codec,frame_stride=args.frame_stride,max_frames=args.max_frames,potential_smoothing=args.potential_smoothing,potential_alpha=args.potential_alpha,trajectory_stabilization=args.trajectory_stabilization,trajectory_alpha=args.trajectory_alpha,reuse_previous=args.reuse_previous,max_reuse_frames=args.max_reuse_frames,heatmap_alpha=args.heatmap_alpha,show_detections=args.show_detections,show_potential=args.show_potential,show_occupancy=args.show_occupancy,show_raw_path=args.show_raw_path,show_trajectory=args.show_trajectory,fail_fast=args.fail_fast)
    summary=run_video_plan(pipeline,input_path,output_path,frame_options=frame_options,metadata_path=args.metadata,options=options)
    print(f"Saved video: {summary['output_path']}");print(f"Processed: {summary['processed_frame_count']}, failed: {summary['failed_frame_count']}, average FPS: {summary['average_processing_fps']:.2f}")
    if summary["metadata_path"]: print(f"Metadata: {summary['metadata_path']}")
    return 0


def main() -> int:
    try:
        args=build_parser().parse_args(); return video_plan(args) if args.command=="video-plan" else 2
    except Exception as exc:
        print(f"Error: {exc}",file=sys.stderr);return 1


if __name__ == "__main__": raise SystemExit(main())
