# TASK-MB-MAP: Orchestrator Plan

## Scenario
Brownfield — substantial C++/CUDA codebase exists, no PRD.

## Subagent assignment

| Stage | Scope | Globs |
|-------|-------|-------|
| S-01 | Build/tooling/CI | CMakeLists.txt, pixi.toml, justfile, *.sh |
| S-02 | Core / cost functions | src/core/**, src/cost_functions/**, include/core/**, include/cost_functions/** |
| S-03 | GPU / CUDA layer | src/gpu/**, include/gpu/** |
| S-04 | GUI / Qt layer | src/gui/**, include/gui/** |
| S-05 | Tests + structure | test/**, src/shape_sensitivity/**, src/Study2Grid/** |

## Status
- [ ] S-01 complete
- [ ] S-02 complete
- [ ] S-03 complete
- [ ] S-04 complete
- [ ] S-05 complete
- [ ] Synthesis complete
- [ ] Review gate passed
