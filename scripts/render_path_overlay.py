"""Render successful planner paths onto their original video frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.planner.image_overlay import merge_perception_chunks, render_path_overlay_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Render successful BEV planner paths over an original video chunk.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--perception", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback multiplier; 0.3 keeps frames at 30%% of source FPS.")
    args = parser.parse_args()
    perceptions = [json.loads(path.read_text(encoding="utf-8")) for path in args.perception]
    perception = merge_perception_chunks(perceptions)
    summary = render_path_overlay_video(args.input, perception, args.output, repository_root=ROOT, codec=args.codec, speed=args.speed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
