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

`pixi run test` → 6/6 pass (~0.1s, no GPU/GUI). This is the "fearlessly edit" headless seam. (U6 added hegel PBT tests; the Tier-2 GPU oracle runs separately under `ctest -L oracle`.)

## Next: U7 - MainScreen MVVM decomposition

U6 is **DONE** (see the table above). The production `OptimizerManager` now runs each stage through the extracted `DirectOptimizer` behind the real GPU cost, and the Tier-2 appearance oracle gates it (IoU 0.9936 vs 0.85 gate; recovered-pose-vs-fem.jts within ~1mm/~0.14deg; headless suite still 6/6). Measured thresholds are recorded in `test/golden/baseline.json` and `golden_oracle.org`.

U7 is the MVVM decomposition of `MainScreen` (view vs app-state/command orchestration vs services). See the U7 unit in the plan: its entry gate is re-validating the human outcome (fast headless pass/fail) at the Phase-3 completion point; extract pose/kinematics persistence as pure tested functions (`src/core/pose_file_io.cpp`), crawl model-list state / pose storage out strangle-style, keep render binding in `Viewer`. Per-layer gate: logic/service/coordinator extractions get headless unit gates; presentation-only cuts get compile + a scheduled manual-visual check (no `MainScreen` characterization, R12). Track `MainScreen` line count + `ui.`-reference count down.

**U7 phase-1 done (persistence seam):** extracted the pure `pose_file_io` service (round-trip + real-fixture tests, headless 7/7) and rewired the four pose/kinematics persistence slots in `MainScreen` to it — preserved all dialogs + view updates; `MainScreen` 5806 -> 5620 lines, `ui.` refs 837 -> 833. Remaining U7 work is the heavier, widget-coupled session/model-list-state extraction, which is phase-gated (re-validate human outcome; the full MVVM commit is re-scoped if the headless seam already delivers the value).

## Key decisions / gotchas to preserve

- **Golden oracle is a behavior-preservation gate, not a correctness check**; correctness comes from independent sources (Tier-1 analytic, known-good Labels).
- **Tier-2 oracle is appearance-based** (render-at-optimized-pose vs Labels), because DIRECT numeric convergence is noisy — don't gate on raw pose values.
- **Cumulative budget is load-bearing** (R15): effective 20k/25k/30k per stage (settings_constants: trunk 20000, branch 5000, leaf 5000; the docs text "10k/20k/30k" is stale); the original zeroes `cost_function_calls_` only before trunk.
- QTn **AUTOMOC**: add Q_OBJECT headers to `add_executable` sources (see `jtml_test_coordinator`).
- `OptimizeCoordinator` uses a **single persistent worker thread** (not per-run deleteLater threads) — per-run threads caused a dangling-pointer segfault in the destructor.
- `jj describe` then `jj new` per change (owner's workflow).
