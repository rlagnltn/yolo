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
- Geometry must use in-memory depth and class maps from the current frame, not reload saved artifacts for computation.
- Record geometry coordinate frame and units in metadata; do not conflate camera-coordinate point clouds with BEV.
- BEV must use in-memory geometry results directly; do not reload saved NPZ or PNG artifacts for computation.
- Do not conflate camera-centric BEV with world-coordinate BEV, and do not treat observed cells as obstacles by default.
- Reuse existing scene class mapping for BEV regions and record grid resolution, ranges, coordinate frame, and units in metadata.
- Do not store raw mask arrays directly in JSON. Store mask images under `outputs/` and put only paths and summary values in JSON.
- Keep generated output data out of Git. `outputs/` can become large quickly.
- Preserve existing detection code unless a requested change specifically requires modifying it.
- Keep later-stage modules as clean extension points unless explicitly asked to implement them.
- Do not delete existing code casually. If removal is necessary, explain why in the relevant docs or final report.
- Use clear file and function names. Prefer small modules with explicit responsibilities.
- After meaningful changes, update `README.md` and the relevant `memory-bank` documents.
- Keep runnable commands and verification results easy to find.

## Efficient Task Execution

Optimization of repeated commands, repository reads, logs, and token usage is the first priority unless it conflicts with correctness or an explicit user instruction.

- Start new work with one `git status --short`; do not run `git log` or revalidate completed features unless explicitly requested.
- Read only user-specified files and the code immediately connected to the functions being changed.
- Do not repeat the same file read, command, or successful test within one task.
- Do not output full files, repository trees, diffs, or successful test logs.
- Use `rg` to locate only required symbols and configuration keys.
- Reuse existing APIs, configuration structures, and test patterns after the smallest necessary inspection.
- Validate once in this order: compile, CLI smoke test, then the full test suite.
- On failure, inspect only the first failure and its relevant traceback, then rerun only the affected test.
- After targeted tests pass, run the full test suite only once at the end.
- Run model inference, video processing, or large artifact generation only when explicitly requested.
- Keep progress updates limited to state changes and failure causes; keep final reports limited to results.
- Record only the commit hash, concise test result, and push status after completion.
- Do not classify an observed BEV cell as occupied without semantic policy, or convert unknown cells to free.
- Keep occupancy NPY state values distinct from PNG display encoding; use cost-grid NPY for downstream computation.
- Build mapping directly from the current frame's in-memory BEV arrays, never by reloading saved BEV artifacts.
- Configure obstacle inflation radius in meters.
- Goal cells must be observed FREE cells; do not give UNKNOWN cells low potential.
- Use raw potential NPY arrays as planner inputs and exclude blocked cells from normalization.
- Keep BEV array-row direction distinct from the positive Z direction, and keep gradient generation separate from path planning.
- Planner uses in-memory potential and occupancy results directly; UNKNOWN and OCCUPIED cells are never paths.
- Diagonal moves enforce corner-cutting policy and planning remains deterministic for identical inputs.
- Local-minimum detection is separate from recovery logic; grid paths are not vehicle trajectories.
- Hybrid fallback runs only for defined gradient failure states, never configuration or invalid start/goal errors.
- A* must not traverse UNKNOWN or OCCUPIED cells, must use admissible heuristics, and must resolve equal priorities deterministically.
- Smoothing results must be collision-checked; never store trajectories through UNKNOWN or OCCUPIED cells.
- Keep grid paths distinct from camera_xz geometric reference trajectories, which are not vehicle-control commands.
- Stream video frames and JSONL records without accumulating the full video in memory.
- Reused or stabilized trajectories must pass collision validation against current occupancy.
- Do not concatenate a failed gradient partial path with a new A* path; A* replans from the original start.
