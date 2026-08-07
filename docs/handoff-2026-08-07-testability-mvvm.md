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

`pixi run test` → 4/4 pass (~0.05s, no GPU/GUI). This is the "fearlessly edit" headless seam.

## Next: U6 — rewire `OptimizerManager` to `DirectOptimizer` + build the Tier-2 oracle

U6 is the delicate validated-GPU-path change and is **NOT yet done**. Its full design:

1. **Extend `DirectOptimizer`** (backward-compatible) with:
   - `SetCallOffset(unsigned)` — cumulative budget across stages (trunk 10k → branch 20k → leaf 30k),
   - `SetIterationCallback(std::function<void()>)` — fired after each ConvexHull+Trisect (drives 30fps `UpdateDisplay`),
   - `SetImprovementCallback(std::function<void(6x double, double)>)` — fired on best-improvement (drives live `UpdateOptimum`).
2. **Add `OptimizerManager::RunDirectStage(range, stage_manager)`** that:
   - sets starting point + search range,
   - builds a `DirectOptimizer` with a GPU eval lambda (sets `gpu_principal_model_` pose A (+B if biplane), calls `stage_manager.callActiveCostFunction()`),
   - sets call offset = running `cost_function_calls_`, budget = cumulative `budget_`,
   - wires the callbacks to `emit UpdateOptimum` / 30fps `UpdateDisplay`,
   - updates `cost_function_calls_`/`current_optimum_*` from the result.
3. **Replace the three near-identical stage loops** (trunk/branch/leaf) in `OptimizerManager::Optimize()` with `RunDirectStage`, keeping the per-stage init/destruct/dilation/`UpdateDilationBackground` + `budget_ += stage_budget` intact.
4. **Build `test/oracle/oracle_test.cpp`** (GPU-labeled): load Kneel_1, optimize, **render the implant at the optimized pose, compare the silhouette to `Labels/fem/`** (pixel-diff / IoU) — the robust appearance-based gate; the recovered-pose-vs-`fem.jts` check is informational only.
   - Wire into `test/CMakeLists.txt` under the `oracle` label (never the `headless` default).
   - It links the GPU pipeline (jtml_gpu/libraries) — needs a GPU machine to run.

**Verification for U6:** `pixi run build` green; `pixi run test` (headless) green; Tier-2 oracle passes on a GPU machine against the captured `fem_oracle.jtak`/`fem.jts`. Because this rewires the validated production optimizer, confirm the GUI still optimizes Kneel_1 acceptably before moving on.

Then U7 (MainScreen MVVM decomposition) and U8 (Qt5→Qt6), both still pending.

## Key decisions / gotchas to preserve

- **Golden oracle is a behavior-preservation gate, not a correctness check**; correctness comes from independent sources (Tier-1 analytic, known-good Labels).
- **Tier-2 oracle is appearance-based** (render-at-optimized-pose vs Labels), because DIRECT numeric convergence is noisy — don't gate on raw pose values.
- **Cumulative budget is load-bearing** (R15): effective 10k/20k/30k; the original zeroes `cost_function_calls_` only before trunk.
- QTn **AUTOMOC**: add Q_OBJECT headers to `add_executable` sources (see `jtml_test_coordinator`).
- `OptimizeCoordinator` uses a **single persistent worker thread** (not per-run deleteLater threads) — per-run threads caused a dangling-pointer segfault in the destructor.
- `jj describe` then `jj new` per change (owner's workflow).
