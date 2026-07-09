# AGENTS

Repository guidance for Codex and other coding agents.

- Preserve the research-project structure. The main modules are `detection`, `segmentation`, `depth`, `bev`, `mapping`, `potential`, and `planner`.
- Keep segmentation code under `src/segmentation/`.
- Do not store raw mask arrays directly in JSON. Store mask images under `outputs/` and put only paths and summary values in JSON.
- Keep generated output data out of Git. `outputs/` can become large quickly.
- Preserve existing detection code unless a requested change specifically requires modifying it.
- Keep later-stage modules as clean extension points unless explicitly asked to implement them.
- Do not delete existing code casually. If removal is necessary, explain why in the relevant docs or final report.
- Use clear file and function names. Prefer small modules with explicit responsibilities.
- After meaningful changes, update `README.md` and the relevant `memory-bank` documents.
- Keep runnable commands and verification results easy to find.
