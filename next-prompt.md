# Next Agent Prompt — JTML Plan 012

## Mission

Implement **Plan 012 — CUDA-Graph Executor Admission and Lifecycle**, starting at **U1 only**.

Plan 011 U1–U5 are landed. Do **not** reimplement them. Do **not** implement Plan 011 U6. That remaining unit was deepened, then a CUDA review showed admission/lifecycle was still design work. 012 replaces 011 U6–U8.

## First turn

1. Read `AGENTS.md`.
2. `ctx_index` on `docs/` if needed; `ctx_search`:
   - `"Plan 012 BatchOutcome GraphAdmissionPolicy"`
   - `"ForceRelease LeavePoisoned CaptureCoordinator"`
3. Read in order:
   - `docs/handoff-2026-08-20-cuda-graph-executor-admission.md`
   - `docs/plans/2026-08-20-012-feat-cuda-graph-executor-admission-plan.md` — Key Technical Decisions + U1
   - `docs/solutions/workflow-issues/jtml-deepened-unit-not-implementation-ready-2026-08-20.md`
   - `docs/solutions/logic-errors/jtml-cuda-graph-stub-failure-2026-08-19.md`
4. `jj st`, `jj log --no-graph -r '@-::@-'`. Checkpoint leftover docs before U1 code.
5. `pixi run build`. No raw git / cmake / make / nvcc.

## U1 definition of done

- Typed CUDA-free `BatchOutcome` (`NotSubmitted` / `OrderedScores` / `PostLaunchAbort` / `WatchdogPoisoned`).
- `GraphAdmissionPolicy` default deny.
- Null-recipe branch no longer sets `useExecutor=true` or overwrites U12 with executor serial passthrough.
- `RunDirectStage` catches coordinator abort and `std::invalid_argument` → `OptimizerError`.
- Headless tests prove deny/null-recipe leaves U12/serial installed.
- No graph launch, no capture coordinator, no 8 GiB dummy pool init, no U7 oracle rewrite.

## Expected reds to leave alone

- `jtml.graph_throughput_oracle` (rev-1 300000) — 012 U7
- `jtml.qml_lint` — pre-existing
- Cut-0 golden mutation only with `JTML_UPDATE_GOLDEN=1`
