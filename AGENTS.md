# AGENTS

Repository guidance for Codex and other coding agents.

- Preserve the research-project structure. The main modules are `detection`, `segmentation`, `depth`, `bev`, `mapping`, `potential`, and `planner`.
- Keep segmentation code under `src/segmentation/`.
- Keep unified perception orchestration and fusion code under `src/perception/`.
- Do not duplicate detection or segmentation inference implementations in perception code.
- Read each input video once in the unified pipeline and do not initialize models per frame.
- When the perception JSON schema changes, update its tests and README documentation together.
- Describe YOLO segmentation as instance segmentation, not full-scene semantic segmentation.
- After completing a pipeline stage, update the relevant memory-bank documents.
- Keep scene semantic segmentation under `src/scene_segmentation/`; do not mix it with YOLO instance segmentation.
- Initialize the scene model once and pass each already-read frame to it without another video pass.
- Never store raw class-map arrays in JSON; distinguish class-ID maps from color visualizations.
- Treat only `road` as vehicle-drivable by default; never classify `sidewalk` as vehicle-drivable.
- Prefer model-config `id2label` over guessed label IDs, and safely handle unknown IDs.
- When scene schemas change, update tests, README, and memory-bank documents together.
- Scene segmentation must retain a functional CPU execution path.
- Keep metric-depth code under `src/depth/` and retain a functional CPU path.
- Never infer or guess depth units; accept only models whose config explicitly reports metric depth.
- Treat float32 NPY as the lossless source depth. uint16 PNG and color maps are derived artifacts.
- Never store visualization-normalized depth as source depth or embed full depth arrays in JSON.
- Initialize the depth model once, reuse the already-read frame, and validate depth/scene-map shapes before combining them.
- Do not store raw mask arrays directly in JSON. Store mask images under `outputs/` and put only paths and summary values in JSON.
- Keep generated output data out of Git. `outputs/` can become large quickly.
- Preserve existing detection code unless a requested change specifically requires modifying it.
- Keep later-stage modules as clean extension points unless explicitly asked to implement them.
- Do not delete existing code casually. If removal is necessary, explain why in the relevant docs or final report.
- Use clear file and function names. Prefer small modules with explicit responsibilities.
- After meaningful changes, update `README.md` and the relevant `memory-bank` documents.
- Keep runnable commands and verification results easy to find.
