# Handoff / Status — JTML Testability + MVVM refactor (2026-08-07)

**Read first:** `AGENTS.md` (build/test/jj conventions), `docs/plans/2026-08-07-001-refactor-testability-mvvm-plan.md` (the plan), `golden_oracle.org` (oracle spec).

## Committed, verified progress (all `green` via `pixi run test`)

| Change | What landed |
|---|---|
| docs | requirements + plan + `golden_oracle.org` + `.gitignore` (`pi-subagents` noise) |
| U1 | Catch2 + CTest `headless` label/`oracle` label + `pixi run test` + CI workflow |
| U3 | removed dead `Point6D(Pose)` ctor + `gpu/render_engine.cuh` leak; fixed transitive-include header debt; CUDA-free data-structure test |
| U5 | pure `DirectOptimizer` (faithful ConvexHull/Trisect/Denormalize + cumulative budget) + Tier-1 analytic golden |
| U4 | headless `OptimizeCoordinator` + persistent worker thread + QtTest lifecycle suite |
| U2 | captured Qt5/GPU baseline (`fem_oracle.jtak`); appearance-based Tier-2 oracle spec + tolerances in `test/golden/baseline.json` |
| build | added `direct_optimizer.cpp` + `optimize_coordinator.cpp` to `jtml_core` (fixed the `file(GLOB)` AUTOMOC link breakage) |
| U6-a | `DirectOptimizer`: SetCallOffset (cumulative 20k/25k/30k) + iteration/improvement callbacks + hegel property-based tests |
| U6-b | `OptimizerManager::Optimize()` rewire to `RunDirectStage(range, stage_manager)` per trunk/branch/leaf with the real GPU DIRECT_DILATION cost |
| U6-c | Tier-2 GPU appearance oracle (`test/oracle/oracle_test.cpp`) — IoU 0.9936 vs 0.85 gate; headless still 6/6 |
| U7-a | extracted pure `pose_file_io` persistence service (round-trip + real-fixture tests) |
| U7-b | rewired MainScreen's 4 pose/kinematics slots to `pose_file_io` (strangle; 5806->5620 lines, ui. 837->833) |
| U8 | Qt5 -> Qt6 migration: pixi Qt6 (qt6-main/wayland 6.7.2), VTK-6 rebuild, Qt6 CMake + API fixes (QRegExp->QRegularExpression, QSurfaceFormat), oracle + headless green under Qt6 |

`pixi run test` → 6/6 pass (~0.1s, no GPU/GUI). This is the "fearlessly edit" headless seam. (U6 added hegel PBT tests; the Tier-2 GPU oracle runs separately under `ctest -L oracle`.)

## Status: 001 + 002 COMPLETE; 003 (layered directory restructure) is the ACTIVE plan

**Plan 001 (U1-U8)** — done. U1 harness/CI, U2 oracle baseline, U3 CUDA-ree decoupling, U4 headless coordinator, U5 pure DIRECT optimizer, U6 `OptimizerManager` -> `DirectOptimizer` rewire + Tier-2 GPU oracle, U7 phase-1 (persistence service + MainScreen strangle; full MVVM phase-gated/re-scoped), U8 Qt5 -> Qt6 migration (gated by the oracle).

**Plan 002 (U9-U11)** — done (code landed, checkboxes ticked). U9 `OptimizeIntentController` (headless AE4 gate), U10 `ModelListBuilder` + MainScreen shrink, U11 Tier-2 oracle expanded to all 3 Kneel_1 frames with per-frame correspondence + reconciled cumulative budget (`[20000,25000,30000,35000]`). Headless suite 11/11 green.

**Plan 003** — the current plan (`docs/plans/2026-08-07-003-refactor-layered-directory-restructure-plan.md`): restructure src/+include/ into `domain/services/coordinator/compute/view/app` layers. Six cuts (U1 cleanup of committed build artifacts, U2 core path renames, U3 layered lib split, U4 gpu+cost_functions->compute merge, U5 view+app isolation, U6 docs finalization). Pure reorg, zero runtime behavior (R15).

Current state is green under **Qt6**: `pixi run test` (headless) 7/7; `ctest -L oracle` (GPU) passes (IoU 0.9936 vs 0.85 gate); bounded GUI smoke runs the event loop on a real display. Full MVVM decomposition of `MainScreen` (session/model-list state) remains as intentionally re-scoped follow-on work, as does expanding the oracle beyond frame 0.

## Key decisions / gotchas to preserve

- **Golden oracle is a behavior-preservation gate, not a correctness check**; correctness comes from independent sources (Tier-1 analytic, known-good Labels).
- **Tier-2 oracle is appearance-based** (render-at-optimized-pose vs Labels), because DIRECT numeric convergence is noisy — don't gate on raw pose values.
- **Cumulative budget is load-bearing** (R15): effective 20k/25k/30k per stage (settings_constants: trunk 20000, branch 5000, leaf 5000; the docs text "10k/20k/30k" is stale); the original zeroes `cost_function_calls_` only before trunk.
- QTn **AUTOMOC**: add Q_OBJECT headers to `add_executable` sources (see `jtml_test_coordinator`).
- `OptimizeCoordinator` uses a **single persistent worker thread** (not per-run deleteLater threads) — per-run threads caused a dangling-pointer segfault in the destructor.
- `jj describe` then `jj new` per change (owner's workflow).
