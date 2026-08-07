---
title: "refactor: Testability + MVVM refactor for the JTML registration app"
type: refactor
status: active
date: 2026-08-07
origin: docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md
---

# Refactor: Testability + MVVM for JTML (Fearless Editing)

## Overview

JTML (`joint-track-machine-learning`) is a Qt5 + VTK 9.3 + CUDA 12.4 + OpenCV 2D-3D knee-implant registration GUI. Its two structural problems are a 5,806-line `MainScreen` god object (`src/gui/mainscreen.cpp`) and a 1,722-line `OptimizerManager` (`src/core/optimizer_manager.cpp`) that couples the pure DIRECT optimization algorithm to CUDA. There are no tests, no test framework, and no working CI.

This plan makes the app **safely editable** — a change to an algorithm or a button path is verifiable by a fast, headless pass/fail instead of launching the GUI and clicking to find a hang or regression. It does this in dependency-ordered, individually-shippable phases: (1) a test harness + golden-oracle foundation, (2) a headless-testable optimize-lifecycle seam, (3) extraction of the pure DIRECT optimizer behind an injected cost boundary, (4) MVVM decomposition of `MainScreen` (only after the first three are green), and (5) a Qt5→Qt6 migration gated by the oracle. The validated numerical core is preserved and gated, never silently re-derived.

---

## Problem Frame

The thesis is defended; this is now a maintainability and future-research ground. The app cannot currently be edited fearlessly because the bugs that hurt most — the "optimizer hangs after finishing", "button click hangs", "optimizer button doesn't fire" classes — live in the thread/orchestration seams that are impossible to exercise without running the GUI, and every change risks silently breaking them.

A prior agent attempt was abandoned; its lessons are load-bearing here (see `docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md`). It produced passing tests that were (a) **false-confidence** — they covered pure math and a GPU-bound "run the real MainScreen" test while never touching the thread/orchestration seams where the regressions lived, and (b) **circular** — test helpers re-derived the code-under-test's own math. This plan structurally rejects both: it tests the seams where hangs live, anchors on a behavior-preservation golden oracle plus independent ground truth, and forbids Qt-mocking wrappers and god-object characterization tests.

---

## Requirements Trace

- R1. Behavior-preservation baseline (Kneel_1) from the known-good Qt5 app before any refactor/migration.
- R2. No test re-derives the code-under-test's math; extract is exercised against the oracle or independent ground truth.
- R3. Deterministic cost/compute comparisons with documented tolerances; DRR/cost compare reference + tolerance resolved before the oracle phase.
- R4. Headless, widget-free coordinator owning the optimize state machine + worker thread, spyable under `QCoreApplication`.
- R5. Lifecycle test (stub cost, no GPU) drives launch→running→finished→re-launch, catches "hangs after optimizer finishes".
- R6. GUI binds to the coordinator as a thin caller.
- R7. Init-failure + stop paths covered headlessly (stub); real GPU-init-failure confined to the explicit GPU target.
- R8. Decompose `MainScreen`: presentation vs app-state/command orchestration vs services.
- R9. Each extraction lands with its gate green; per-layer gate (headless unit vs compile+manual-visual).
- R10. `MainScreen` shrinks measurably (line / `ui.`-reference count).
- R11. Testing enabled; default run headless (no GUI/VTK/GPU); GPU/real-VTK in a flagged target.
- R12. No Qt-mocking wrappers; no instantiate-the-real-`MainScreen` characterization.
- R13. Working CI runs the headless suite; lifecycle tests timeout-bounded; numeric fidelity on an explicit GPU gate.
- R14. Test framework declared in pixi.
- R15. Preserve validated DIRECT/cost logic; gate with oracle/convergence.
- R16. Qt5→Qt6: pixi pin, VTK rebuild, `find_package(Qt6)`, API revalidation, gated by pre-migration oracle + bounded manual GUI smoke.

**Origin actors:** A1 (developer/researcher), A2 (coding agent).
**Origin flows:** F1 (optimize lifecycle), F2 (golden-oracle regression gate), F3 (extraction-with-gate).
**Origin acceptance examples:** AE1 (Covers R4,R5,R7 — headless lifecycle returns to idle, timeout-bounded), AE2 (Covers R1,R2,R16 — Kneel_1 oracle within tolerance), AE3 (Covers R11,R12 — headless default, GPU flagged), AE4 (Covers R8,R9,R10 — seams out of MainScreen, per-layer gate), AE5 (Covers R13 — CI gates headless suite, timeout fails a hang).

---

## Scope Boundaries

- Not a from-scratch rewrite; the validated DIRECT/cost/CUDA core is preserved and gated.
- Not a Rust port; the optimizer seam is designed Rust-friendly but no Rust is written.
- No Qt-mocking wrappers (`QFileDialog`/`QMessageBox` function-pointer wrappers) and no "run the real `MainScreen`" characterization tests.
- No full UI overhaul of every dialog; focus is `MainScreen`, the optimize lifecycle, persistence, and the cost/compute seams.
- GUI and GPU/real-VTK tests are not part of the default headless run; numeric oracle fidelity is a separate, explicitly-triggered GPU gate.
- Regression safety currently spans a single oracle case (Kneel_1); expansion is a follow-on.

### Deferred to Follow-Up Work

- Expanding the golden oracle beyond Kneel_1: future iteration, triggered when a refactor touches changed-case behavior or before claiming fearlessness across cases.
- The ML-integration pathway (running a trained model over the original image): deferred; the binary-silhouette path is the primary oracle.
- Capturing the refactor outcome to a learnings doc via `/ce-compound`: after phases land.

---

## Context & Research

### Relevant Code and Patterns

- `src/core/optimizer_manager.cpp` — DIRECT core (`ConvexHull` 1495, `TrisectPotentiallyOptimally` 1553, `DenormalizeRange` 1623, `DenormalizeFromCenter` 1633, `SetSearchRange` 869, `SetStartingPoint` 879). Only `EvaluateCostFunction` (1437) touches CUDA; `cost_function_calls_` is the budget guard and is cumulative across stages (zeroed before trunk only, then `+=` per stage — effective 10k/20k/30k).
- `src/core/data_structures_6D.cpp` / `include/core/data_structures_6D.h` — `Point6D(gpu_cost_function::Pose)` ctor (`.h:36`, `.cpp:34`) pulls `gpu/render_engine.cuh` into a pure header. Grep confirms the ctor has **no active production call sites** (effectively dead code).
- `include/gui/viewer.h` — existing VTK render-wrapper abstraction (`Viewer`), the natural anchor for the view layer.
- `include/gui/mainscreen.ui` / `include/gui/drr_tool.ui` — already use `QVTKOpenGLNativeWidget` (Qt6-compatible); `main.cpp` does **not** call `QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat())` before `QApplication` (latent bug, louder in Qt6).
- `pixi.toml` — `qt = "5.*"`; no test framework; `configure` passes `-DQt5_DIR`; no `test` task. `vtk_installer.sh` forces `-DVTK_QT_VERSION=5 -DVTK_USE_QT6=OFF`.
- `CMakeLists.txt` — `find_package(Qt5)`, `#add_subdirectory(test)` (122), `#enable_testing()` (129). `test/CMakeLists.txt` adds `nfd` + `vtk` (no `BUILD_TESTING`, no registration).
- `test/vtk/test_case/` — left-knee (`KR_left_8_*`); **not** oracle fixtures. Oracle fixtures are `example_studies/Kneel_1/`.
- `golden_oracle.org` — oracle spec (now corrected: trunk 6px / branch 3px / leaf 1px dilation).

### Institutional Learnings

- No `docs/solutions/` or memory KB exists; the durable record is the brainstorm requirements doc + `golden_oracle.org`. Treat R1–R16/AE1–AE5 as normative.
- Lifecycle-seam-first, no-Qt-mocking, no-god-object-characterization, and anti-circularity (R2) are direct lessons from the abandoned attempt.

### External References

- Qt — Worker/controller `QThread` pattern, `threads-qobject`, `QSignalSpy` (use `wait(timeout)`, not fixed sleeps; observe on the test thread, QTBUG-2842), Qt Test offscreen, Qt 5→6 porting (`QRegExp`→`QRegularExpression` via Qt5Compat).
- VTK — `QVTKOpenGLNativeWidget` requires `QSurfaceFormat::setDefaultFormat`; VTK Qt6 build (`VTK_QT_VERSION=6`, `VTK_USE_QT6=ON`, delete stale Qt5 cache); conda-forge `vtk` package is built against `qt6-main`.
- Feathers, *Working Effectively with Legacy Code* — characterization tests at the seam you care to change (the optimizer boundary, not the GUI); seam + enabling point = constructor-injected `std::function`.
- CUDA/Warp determinism — GPU reductions/atomics are not bit-reproducible across hardware; expect CPU-vs-GPU float divergence. Oracle must be tolerance-based on the GPU tier.

---

## Key Technical Decisions

- **Hybrid test framework:** QtTest (`Qt5::Test` until U8, then `Qt6::Test` after the migration) for the QObject/QThread/QSignalSpy coordinator seam; Catch2 v3 for pure math (DIRECT, data structures, golden comparisons). Rationale: the lifecycle seam *is* Qt threading and QtTest gives the event loop + `QSignalSpy` for free; Catch2's `Approx`/matchers are far better for numerics and its `catch_discover_tests` registers into CTest.
- **Headless via `QCoreApplication`, not offscreen-as-default:** pure DIRECT and the coordinator need neither a widget nor a display; `QT_QPA_PLATFORM=offscreen` is reserved for the rare widget-level smoke cases. Zero GPU in the default run (R11/AE3).
- **Two-tier oracle (R3):** Tier 1 = pure-CPU analytic golden (bit-exact, CI) for the extracted DIRECT algorithm against an analytic cost function; Tier 2 = GPU tolerance-based Kneel_1 oracle (pose within per-axis tolerance + thresholded silhouette pixel-diff) on an explicit GPU target with the baseline captured on the same hardware. GPU is not bit-reproducible, so Tier 2 is never bit-exact.
- **Break the GPU header leak first:** remove the `Point6D(Pose)` ctor + `render_engine.cuh` include from `data_structures_6D`/`direct_data_storage` so those and the extracted DIRECT core compile CUDA-free (enables the headless target) before touching behavior.
- **Preserve cumulative budget semantics** when extracting DIRECT: the DIRECT loop's budget is cumulative (trunk 10k, then +branch, then +leaf → effective 10k/20k/30k guards); do not "fix" it to per-stage 10k during extraction. Treat the effective cumulative budget as the oracle gate's iteration cap.
- **Coordinator keeps real-GPU paths out of the headless surface:** `OptimizerManager` remains the CUDA initialization + eval adapter; a new headless `OptimizeCoordinator` owns the state machine + worker. The threaded stop uses a cooperative `std::atomic<bool>`/flag polled by the worker (replacing the current cross-thread `Qt::DirectConnection` to `onStopOptimizer`), with a bounded accept for "prompt".
- **MVVM is sequenced, not committed upfront (R8):** decompose `MainScreen` only after phases 1–3 are green and re-validated against the human outcome. Qt6 (R16) is deliberately last, gated by the pre-migration oracle; consider whether to migrate Qt6 before heavy MVVM decomposition to avoid double-touching the same god-object lines (re-evaluate at the Phase-3 completion gate).
- **`conda-forge vtk` as the Qt6 de-risk:** if available, add `vtk` (built against `qt6-main`) via pixi to replace the manual `vtk_installer.sh` build; otherwise flip `VTK_USE_QT6=ON`. Decide at Phase 5.

---

## Open Questions

### Resolved During Planning

- Test framework: QtTest (Qt seam) + Catch2 v3 (pure math). Both added to pixi (R14).
- Headless strategy: `QCoreApplication` + `QSignalSpy` for lifecycle; offscreen only for widget smoke.
- Oracle compare reference: two-tier (Tier-1 CPU analytic in CI; Tier-2 GPU tolerance on explicit target). Exact tolerance values recorded in `golden_oracle.org` at baseline capture (Phase 1, R3/D3).
- DIRECT extraction boundary: `std::function<double(const Point6D&)>`; keep cumulative budget.

### Deferred to Implementation

- Exact `DirectOptimizer`/`OptimizeCoordinator` class and method names (finalize while extracting).
- Exact tolerance numbers (translations, rotations, silhouette pixel threshold): recorded when the Qt5 GPU baseline is captured (a Phase-1 gate).
- The Tier-2 compare reference is **resolved** in the plan: known-good Labels (B) projections + `fem.jts` pose; no separate CPU DRR reference is built.

- Qt5→Qt6 OpenCV Qt-binding coordination (OpenCV ships a Qt5 variant in the lockfile — ripple to confirm at Phase 5).

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

State machine owned by the headless coordinator (the R4/R6 seam the GUI binds to):

```text
            user.start()
   ┌─────────────┐
   ▼             │
 [IDLE] ──▶ [RUNNING] ──▶ [FINISHED] ──▶ [IDLE]  (re-launchable)
   ▲                │
   │                ├── failure ──▶ [ERROR] ──▶ [IDLE]   (error surfaced then consumed)
   │                └── user.stop ─▶ (cooperative, bounded) ─▶ [IDLE]
   └─────────────────────────────────────────────┘
```

The two extraction seams:

```text
GUi  --"optimize intent"-->  OptimizeCoordinator (headless, QObject)
                                 │  owns QThread + worker
                                 ▼
                            worker: DIRECT loop (extracted DirectOptimizer)
                                 │  double eval(Point6D)   ← injected
                                 ▼
                    OptimizerManager::EvaluateCostFunction (real GPU cost adapter)
                                 │  ← or stub cost in tests
                                 ▼
                    CostFunctionManager → GPUModel/GPUMetrics → CUDA kernels
```

The pure-data seam (U3) removes the `Point6D(Pose)` ctor so `data_structures_6D`/`direct_data_storage` and `DirectOptimizer` compile without CUDA.

---

## Output Structure

    example_studies/Kneel_1/          # (existing) oracle source fixtures (right knee)
    golden_oracle.org                 # (modify) record compare reference + tolerances
    test/
      CMakeLists.txt                  # (rewrite) register unit + lifecycle + oracle targets
      unit/                           # Catch2 pure-math tests (no Qt event loop)
        test_harness_smoke.cpp
        test_data_structures.cpp
        test_direct_optimizer.cpp
        test_pose_file_io.cpp
      lifecycle/                      # QtTest coordinator tests
        coordinator_test.cpp
      oracle/                         # GPU-labeled Tier-2 oracle + R7b GPU-init-failure
        oracle_test.cpp
      golden/                         # mirrored oracle fixtures + captured baseline
        pose/           fem.jts  (golden pose)
        images/         (known-good projections / Labels)
        baseline.json   (settings + tolerance values)
    src/core/
      direct_optimizer.h/.cpp         # (new) pure DIRECT, std::function<double(Point6D)>
      optimize_coordinator.h/.cpp     # (new) headless state machine + worker
    include/core/
      direct_optimizer.h              # (new)
      optimize_coordinator.h          # (new)
      data_structures_6D.h            # (modify) drop Point6D(Pose) + render_engine.cuh
    src/core/data_structures_6D.cpp   # (modify) same
    src/core/optimizer_manager.cpp    # (modify) retain GPU eval; rewire to DirectOptimizer
    .github/workflows/cmake.yml       # (rewrite) pixi headless CI / or new ci.yml

---

## Implementation Units

- [ ] U1. **Test harness, frameworks, and CI**

**Goal:** Enable CTest; add Catch2 + QtTest; register the headless suite with a default run that has no GPU/display; add a real pixi/CI gating job.

**Requirements:** R11, R13, R14, AE3, AE5.

**Dependencies:** None.

**Files:**
- Create: `pixi.toml` (add `catch2` dev dep), `.github/workflows/cmake.yml` (rewrite or new `ci.yml`)
- Modify: `CMakeLists.txt` (un-comment `enable_testing()`/`add_subdirectory(test)`, gate on `BUILD_TESTING`/`JTML_BUILD_TESTS`), `test/CMakeLists.txt` (register targets + `add_test`/`catch_discover_tests`, add `headless` default label and `gpu`/`oracle` labels), `pixi.toml` (`[tasks.test]`)
- Test: `test/unit/test_harness_smoke.cpp`

**Approach:**
- Add Catch2 to `pixi.toml` dev deps. Wire `find_package(Qt5 COMPONENTS Core Gui Widgets Test)` + `Qt5::Test` for now (the project is on Qt5 until U8; only U8 renames to `Qt6::Test`). The goal at U1 is a *runnable* harness, not the Qt6 migration.
- Bind the headless default suite with a run-level timeout (`ctest --timeout <N>` / `set_tests_properties(... TIMEOUT n)` / `CTEST_TEST_TIMEOUT`) so a hang in *any* test fails the job, not just `QSignalSpy::wait`-guarded ones (R13/AE5).
- Introduce a test-output convention: binaries under `test/`, ctest labels `headless` (default) and `gpu`/`oracle` (explicit-flag only).
- Add `pixi run test` mapping to `ctest -L headless` (+ `ctest -L gpu` behind an explicit flag).
- Rewrite CI to configure + build + run the headless suite via pixi.

**Test scenarios:**
- Integration: `pixi run test` configures, builds, and runs the headless suite and reports pass/fail (`Covers AE5`).
- Happy path: a trivial smoke assertion compiles and passes in a Catch2 target and a QtTest target.
- Edge case: `ctest -L headless` runs with no GPU present and no display.
- Error path: a deliberately failing assertion yields a non-zero ctest exit (CI fails).

**Verification:** `pixi run test` is green; CI job runs the headless suite (timeout-bounded); a GPU-labeled test is *not* run by the default.

---

- [ ] U2. **Golden-oracle definition and Qt5 baseline capture** *(a pre-flight gate — see Notes)*

**Goal:** Decide the DRR/cost compare reference + tolerance, record them and the copied Kneel_1 fixtures as committed ground truth, and capture the known-good pose from the Qt5/GPU app.

**Requirements:** R1, R2, R3, AE2, D3 (pre-flight).

**Dependencies:** U1 (harness to run the oracle target). U2 defines the oracle spec and captures the baseline; the Tier-1 analytic CI golden is implemented in **U5** (it needs the extracted `DirectOptimizer`), and the Tier-2 GPU gate lands in **U6**.

**Files:**
- Create: `test/golden/pose/fem.jts` (copied known-good), `test/golden/images/` (known-good projections from `Labels/`), `test/golden/baseline.json` (settings + tolerance values), `test/oracle/oracle_test.cpp` (Catch2, GPU-labeled)
- Modify: `golden_oracle.org` (record compare reference + tolerance values)

**Approach:**
- Mirror the right-knee `example_studies/Kneel_1` fixtures (silhouettes (A), `Labels` (B), `fem.jts` (C), `KR_right_7_fem.stl` (D)) into `test/golden/`.
- Set the compare reference: Tier-2 compares the GPU-rendered projection against the committed known-good **Labels (B)** projections and the recovered pose against `fem.jts` (no separate CPU DRR reference is needed — Tier-1 covers headless DIRECT correctness analytically; Tier-2 is the GPU tolerance gate). Tier-2 runs only under the `oracle` label on a GPU machine, within documented per-axis pose tolerance and a documented silhouette pixel-diff threshold.
- Capture the known-good pose from the current Qt5/GPU app and record the tolerance values in `golden_oracle.org`.
- U2 owns oracle *definition* + baseline capture only; the Tier-1 CI golden body is delivered in U5, and the Tier-2 gate in U6.

**Test scenarios:**
- Happy path: `ctest -L oracle` on a GPU machine runs Kneel_1 and the recovered pose is within tolerance of `fem.jts` (`Covers AE2`).
- Edge case: the Tier-2 gate is skipped (or labeled) so a default CI run never launches CUDA.
- Error path: a manually-corrupted golden pose fails the gate — proving the oracle catches drift (it is a behavior-preservation gate, not a correctness check: a latent bug in the current app is *preserved*, which is expected and documented).

**Verification:** `test/golden/` committed; tolerance values documented; the Tier-2 oracle passes on the user's GPU machine; default CI does not execute it.

**Notes (pre-flight gate):** This unit cannot truly pass until the user captures the Qt5 GPU baseline (D3). If no GPU baseline is obtainable, defer the Tier-2 gate and (per requirements) defer the Qt6 migration (R1/R16). Log this as an explicit phase-1 gate requiring A1.

---

- [ ] U3. **Decouple pure data structures from GPU**

**Goal:** Remove the `Point6D(Pose)` constructor and the `gpu/render_engine.cuh` include from `data_structures_6D`/`direct_data_storage` so those, and everything depending on them, compile without CUDA.

**Requirements:** R11 (headless compile target), R15 (preserve behavior), unblocks U4/U5.

**Dependencies:** U1.

**Files:**
- Modify: `include/core/data_structures_6D.h` (drop ctor + include), `src/core/data_structures_6D.cpp`, and any downstream TU (sym_trap, optimizer_manager, mainscreen) only where a compile error actually requires it.
- Test: `test/unit/test_data_structures.cpp` (Catch2)

**Approach:**
- Grep verified `Point6D(gpu_cost_function::Pose)` has **no active production call sites** — it is effectively dead code (and carries a latent bug: it assigns `xa` twice and never sets `za`). So U3 simply **deletes the ctor** from the pure struct and drops `#include "gpu/render_engine.cuh"` from `data_structures_6D.h`. Any TU that then fails to compile is fixed at its exact failing site.
- No behavior change beyond removing the unused ctor; verified via the CUDA-free unit target.

**Test scenarios:**
- Integration: a pure test target that includes `data_structures_6D.h` and `direct_data_storage.h` compiles and links with **no** CUDA/`.cu` sources (`Covers AE3`).
- Happy path: `Point6D` construction/comparison/`GetLargestDirection` behave as before.
- Edge case: after deletion, no remaining code path constructs a `Point6D` from a `Pose` (grep-clean), and the pure unit target compiles CUDA-free.

**Verification:** `test/unit/test_data_structures.cpp` builds as a CUDA-free target; existing app still compiles and behaves identically.

---

- [ ] U4. **Headless optimize-coordinator + worker**

**Goal:** Extract the optimize orchestration (state machine + worker thread) into a widget-free `OptimizeCoordinator` testable under `QCoreApplication` + `QSignalSpy` with an injected stub cost; fix the cross-thread stop.

**Requirements:** R4, R5, R6, R7, AE1.

**Dependencies:** U1, U3.

**Files:**
- Create: `include/core/optimize_coordinator.h`, `src/core/optimize_coordinator.cpp`, `test/lifecycle/coordinator_test.cpp` (QtTest)
- Modify: `src/gui/mainscreen.cpp` (bind thin caller to coordinator; remove inline thread orchestration)

**Approach:**
- `OptimizeCoordinator` is a `QObject` on the test/main thread owning a `QThread` + a worker `QObject` moved to it; re-emits progress/result/error signals on the test thread.
- The worker runs the optimize loop over an injected `std::function<double(const Point6D&)>`. Until U5, the worker invokes the existing `OptimizerManager::Optimize()` body through that boundary so the coordinator is headless-testable now; at U5 that inline body is replaced verbatim by the extracted `DirectOptimizer` (not duplicated), and at U6 it is bound to the real GPU eval. This pins the U4 worker's production path and removes the "eventual DirectOptimizer" ambiguity.
- Stop via a cooperative flag polled by the worker, with a bounded `promptly` accept; replace the current cross-thread `Qt::DirectConnection` to `onStopOptimizer`.
- All lifecycle tests use `QCoreApplication` (never `QApplication`/widgets), observe the coordinator's signals (not the worker thread), and are timeout-bounded (`QSignalSpy::wait`), so a hang fails the test.

**Test scenarios:**
- Happy path: stub cost → coordinator transitions idle → running → finished → idle and accepts a second launch (`Covers AE1`).
- Happy path: progress/result signals are re-emitted on the test thread (no `QTBUG-2842` worker-thread spy).
- Edge case: launching while already running is refused (no double-start).
- Error path: injected cost-init failure → `ERROR` state surfaced then back to idle, re-launchable, no deadlock (headless, stub — real GPU-init-failure handled in the GPU target, per R7b).
- Error path: a `stop()` mid-run returns to idle within the timeout bound.
- Integration: a deliberately-stuck worker timeout fails the test (AE5 behavior).

**Verification:** `test/lifecycle/coordinator_test.cpp` runs headless (no display, no GPU) and passes under `pixi run test`; the GUI still optimizes.

---

- [ ] U5. **Extract the pure DIRECT optimizer**

**Goal:** Move `ConvexHull`/`TrisectPotentiallyOptimally`/`Denormalize*`/`SetSearchRange`/`SetStartingPoint` and the cumulative-budget loop into a CUDA-free `DirectOptimizer` that takes an injected cost function and returns the argmin.

**Requirements:** R2, R15; owns the Tier-1 analytic golden (not AE2 — AE2 is the Tier-2 Kneel_1 GPU oracle, delivered via U2/U6).

**Dependencies:** U1, U3 (CUDA-free `data_structures_6D`).

**Files:**
- Create: `include/core/direct_optimizer.h`, `src/core/direct_optimizer.cpp`, `test/unit/test_direct_optimizer.cpp` (Catch2)
- Modify: `src/core/optimizer_manager.cpp` (delegate the loop to `DirectOptimizer`; keep CUDA eval)

**Approach:**
- `DirectOptimizer` owns `DirectDataStorage` + `budget_` + range/start + denormalization, exposes `Point6D optimize(std::function<double(const Point6D&)>)` running the `ConvexHull`/`TrisectPotentiallyOptimally` loop.
- **Preserve cumulative budget semantics** (10k/20k/30k effective) and make it explicit.
- `OptimizerManager::EvaluateCostFunction` becomes the real GPU-backed callback (`OptimizerManager` retains CUDA init/eval). This is a behavior-preserving rewire, not a re-implementation (R15).

**Test scenarios:**
- Happy path: analytic sphere/quadratic cost (e.g. `f = Σ(p−c)²` on the unit cube) converges to the known min within tolerance; returns the argmin (Tier-1 independent ground truth per R2).
- Happy path: budget reached terminates the loop and returns the best found (`Covers` R5-adjacent determinism).
- Edge case: zero-range in one dimension terminates cleanly.
- Edge case: budget accounting matches current code — the initial center evaluation consumes one budget unit (mirroring the current `cost_function_calls_++` on the seed eval), so the extracted loop's effective cap equals the pre-extraction cumulative cap (10k/20k/30k); assert equality with the golden to avoid an off-by-one divergence.
- Error path: a cost that returns `NaN`/`inf` doesn't corrupt the optimum selection.

**Verification:** `test/unit/test_direct_optimizer.cpp` runs bit-end-to-end in CI (no GPU); `OptimizerManager` produces identical results to pre-extraction on the existing pipeline.

---

- [ ] U6. **Rewire the real GPU cost into the extracted optimizer**

**Goal:** Make the production app run the extracted `DirectOptimizer` with the real `OptimizerManager::EvaluateCostFunction` adapter — behavior-preserving, gated by the oracle/convergence tests.

**Requirements:** R15, AE2.

**Dependencies:** U5.

**Files:**
- Modify: `src/core/optimizer_manager.cpp`, `include/core/optimizer_manager.h`

**Approach:**
- Wire `DirectOptimizer` with the CUDA-backed eval callback; keep the GPU init/upload path and the per-frame trunk/branch/leaf orchestration intact around it.
- Emit progress/optimum signals from the coordinator layer as before.

**Test scenarios:**
- Integration: on a GPU machine, running the pipeline with the extracted optimizer yields a pose within tolerance of the pre-extraction result / the Tier-2 oracle (`Covers AE2`).
- Edge case: the cumulative-budget behavior matches the pre-extraction run (no off-by-stage).
- Error path: on a GPU machine, forcing CUDA/cost init failure surfaces a clean `ERROR` → idle without deadlock (owns origin **R7b**; GPU-labeled target, not the headless default).

**Verification:** GPU-labeled `oracle` target still green post-rewire; existing app behavior unchanged.

---

- [ ] U7. **MainScreen MVVM decomposition (view vs state vs services)** *(sequenced after phases 1–3 are green)*

**Goal:** Extract app-state/command orchestration and services (frames/models/selection, pose & kinematics persistence, optimize intent) out of `MainScreen`, leaving a thinner view layer that binds to the coordinator and services.

**Requirements:** R8, R9, R10, AE4.

**Dependencies:** U2, U4, U6 (oracle baseline captured, coordinator + DIRECT in place).

**Files:**
- Create: `include/core/pose_file_io.h`, `src/core/pose_file_io.cpp` (persistence service), controller/app-state headers under `src/core/` (e.g. `session_state.h`)
- Modify: `src/gui/mainscreen.cpp`, `include/gui/mainscreen.h` (shrink; wire to coordinator + services)
- Test: `test/unit/test_pose_file_io.cpp` (Catch2)

**Approach:**
- **Entry gate (R8):** before U7 work is authorized, re-validate the human outcome (fast headless pass/fail for a representative change) at the Phase-3 completion point; if phases 1–3 already deliver it, the MVVM commit is re-scoped. MVVM stays phase-gated, not a foregone deliverable.
- Extract persistence (pose/kinematics load-save) as pure functions tested against real fixture files (round-trips), not re-derived format logic (R2).
- Crawl outward strangle-style: model-list state → a service, pose storage → a service, render binding stays in the view layer via `Viewer`.
- Per-layer gate (R9): logic/service/coordinator extractions get headless unit gates; presentation-only cuts get a successful compile + a scheduled manual-visual check (no `MainScreen` characterization, R12).
- Track coupling coarsely: `MainScreen` line count and `ui.`-reference count move down (R10).

**Test scenarios:**
- Happy path: save→load a pose file round-trips to the same values.
- Happy path: selecting a model/frame updates session state with no widget.
- Error path: loading a malformed pose file yields a clean error, not a hang.
- Integration: a controller triggers `OptimizeCoordinator` intent → coordinator runs (headless stub) → state updates (`Covers AE4`).

**Verification:** `MainScreen` coupling metrics trend down; every extraction's headless gate is green before the next cut; presentation-only cuts are compile+manual-visual-gated.

---

- [ ] U8. **Qt5 → Qt6 migration (gated by the oracle)**

**Goal:** Move the whole build to Qt6: pixi pin, VTK rebuild against Qt6, `find_package(Qt6)`, API revalidation across `MainScreen`, plus the `QSurfaceFormat::setDefaultFormat` fix and a bounded manual GUI smoke step.

**Requirements:** R16, AE2.

**Dependencies:** U2 (pre-migration oracle captured), U7 (ideally) or sequenced after it.

**Files:**
- Modify: `pixi.toml` (`qt = "5.*"` → `qt >= 6` and `-DQt6_DIR` in `configure`; optionally add `vtk` from conda-forge per Key Decision), `vtk_installer.sh` (`VTK_QT_VERSION=6`, `VTK_USE_QT6=ON`, delete stale Qt5 cache, or replace with conda-forge `vtk`), `CMakeLists.txt` (`find_package(Qt6)`, `Qt5::`→`Qt6::`, and the test targets' `Qt5::Test` → `Qt6::Test`), `src/gui/main.cpp` (`QSurfaceFormat::setDefaultFormat(...)`), `src/gui/mainscreen.cpp` + `include/gui/mainscreen.h` (the R16 Qt5-API revalidation pass lands here), `src/Study2Grid/main.cpp` (`QRegExp`→`QRegularExpression` if built)
- Test: existing headless suite must remain green post-migration.

**Approach:**
- Run only after the Qt5 baseline (U2) is captured; the oracle + headless suite act as the regression net.
- Fix the missing `QSurfaceFormat::setDefaultFormat` before/at migration (latent bug).
- Because a Qt port most plausibly breaks GUI/render/UI-binding that the headless suite can't see, include a bounded manual GUI smoke step (open app, load Kneel_1, run optimizer) per R16.
- If adopting conda-forge `vtk`, drop `_deps/vtk` and `build-vtk`.

**Test scenarios:**
- Integration: the entire headless suite (units + coordinator) passes unchanged under Qt6 (`Covers AE2` numerics/lifecycle).
- Edge case: `find_package(Qt6)`, VTK builds/links against Qt6, and `QVTKOpenGLNativeWidget` renders (manual smoke).
- Error path: a Qt6-incompatible API (e.g. `QRegExp`) is caught by the build, not by runtime.

**Verification:** `pixi run build` on Qt6 is green; headless suite green; manual GUI smoke passes on a GPU/display machine.

---

## System-Wide Impact

- **Interaction graph:** `MainScreen`'s `LaunchOptimizer` and its ~9 signal wirings move behind the coordinator; `OptimizerManager`'s `Optimize()` becomes the worker; cost-manager/cost-function wiring is unchanged.
- **Error propagation:** coordinator `ERROR` state → surfaced to the view and consumed → back to idle; never a hung thread.
- **State lifecycle risks:** double-start guarded; stop is cooperative (flag polled), so a mid-kernel stop may wait one iteration — accept and bound; `QThread` teardown (`quit(); wait(); deleteLater`) avoids crashes.
- **API surface parity:** `OptimizerManager`'s public signal/slot surface is preserved for compatibility during extraction; no public API breaks.
- **Integration coverage:** coordinator→worker→real-GPU-cost path is only fully exercised on a GPU machine (Tier-2 oracle); the headless suite covers the state machine + DIRECT with stubs.
- **Unchanged invariants:** validated DIRECT math, cost-function parameter semantics, and CUDA kernels are preserved; extraction is rewire-not-reimplement (R15). `Point6D`/`HyperBox6D`/`DirectDataStorage` values are unchanged.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Golden oracle is GPU/VTK-bound and can't run in default CI | Two-tier: Tier-1 CPU analytic golden runs in CI; Tier-2 GPU tolerance oracle runs only under the explicit `oracle` label (R13 split). |
| Oracle baseline captured from code-under-test is not a correctness check | Document explicitly (R2): it is a behavior-preservation gate; independent ground truth (analytic cost, known-good Labels) provides correctness. |
| Qt5 baseline (D3) unobtainable | Phase-1 gate: if not captured, defer Tier-2 oracle + the Qt6 migration (R1/R16 fallback). |
| GPU/CUDA nondeterminism breaks bit-exact assertion | Tier-2 uses per-axis pose tolerance + thresholded silhouette pixel-diff, never bit-exact; capture on the same hardware. |
| Qt6 migration scope (VTK rebuild, OpenCV Qt binding, API revalidation) | Sequence last, gated by pre-migration oracle; prefer conda-forge `vtk` to kill `VTK_QT_VERSION` fragility; bounded manual GUI smoke. |
| Cumulative DIRECT budget semantics misread | Key Decision + U5 test scenario pin it (10k/20k/30k) and the oracle uses the effective cap. |
| Over-engineering test scaffolding (repeat of abandoned attempt) | Thin, behavior-focused tests; no Qt-mocking wrappers; no god-object characterization (R12). |
| Cross-thread stop deadlock | Cooperative flag + bounded accept + timeout-bound lifecycle test converts any hang into a failing test (AE5). |

---

## Documentation / Operational Notes

- Record the oracle compare reference + tolerance values in `golden_oracle.org` (U2) and mirror them in `test/golden/baseline.json`.
- Document the `pixi run test` usage in `readme`/`justfile`; add a `test` task to `just`.
- After phases land, capture the outcome via `/ce-compound` (no memory KB exists today).
- The previous attempt's `big-main-diff.diff`, `context.md`, and rejected wrapper tests should be deleted/ignored rather than resurfaced.

---

## Sources & References

- **Origin document:** [docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md](docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md)
- Oracle spec: `golden_oracle.org`
- Relevant code: `src/core/optimizer_manager.cpp`, `src/core/data_structures_6D.cpp`, `src/gui/mainscreen.cpp`, `include/gui/viewer.h`, `include/gui/mainscreen.ui`
- External: Qt `QThread`/`QSignalSpy`/QtTest docs; VTK `QVTKOpenGLNativeWidget` + Qt6 build; Feathers *WELC*; CUDA/Warp determinism guidance.
