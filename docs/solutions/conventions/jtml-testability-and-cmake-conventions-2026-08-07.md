---
title: Headless Testing and Testability-Refactor Conventions for a Qt/GPU Desktop App (JTML)
date: 2026-08-07
last_updated: 2026-08-08
category: conventions
module: JTML
problem_type: convention
component: testing_framework
severity: medium
related_components:
  - tooling
  - development_workflow
applies_when:
  - Adding new test targets to a Qt/CMake project (QtTest or Catch2)
  - Introducing a shared Q_OBJECT header that is #included but not a listed source
  - Extending a CMake target that globs headers but lists sources explicitly
  - Writing tests that must run headless with no GPU, VTK render, or widgets
  - Refactoring a pure algorithm or thread-seam out of a Qt and GPU app for testability
tags:
  - qt
  - cmake
  - automoc
  - ctest
  - headless-testing
  - testing-framework
  - jtml
  - oracle
---

# Headless Testing and Testability-Refactor Conventions for a Qt/GPU Desktop App (JTML)

## Context

JTML is a validated Qt6 (qt6-main/wayland 6.7.2) + VTK 9.3 rebuilt against Qt6 + CUDA 12.4 + OpenCV C++20 desktop app for 2D-3D knee-implant registration (a DIRECT global optimizer over a GPU cost function). Before the refactor it could not be edited fearlessly:

- `src/view/mainscreen.cpp` is a ~5,800-line god object mixing UI wiring, app state, compute, I/O, and optimize orchestration.
- `src/coordinator/optimizer_manager.cpp` (~1,722 lines) coupled the pure DIRECT algorithm directly to CUDA; only `EvaluateCostFunction` touched the GPU, but the loop lived inline next to it.
- Testing was effectively disabled (`#add_subdirectory(test)` + `#enable_testing()` commented out in `CMakeLists.txt`); there was no working CI (the `.github/workflows/cmake.yml` was inert boilerplate), and no test framework was declared in `pixi.toml`.

A **prior agent attempt was abandoned**, and its two failure modes are load-bearing lessons:

1. **False confidence** — passing tests covered pure math and a GPU-bound "run the real MainScreen" test while never touching the thread/orchestration seams (idle→running→finished→re-launch) where the actual hangs lived.
2. **Circular tests** — test helpers re-derived the code-under-test's own math (pose-file and denormalization tests reimplemented production logic), so a green suite proved nothing.

This session's outcome (all green via `pixi run test`, no GPU/GUI): a headless harness, a pure extracted `DirectOptimizer`, a headless `OptimizeCoordinator` thread seam, a two-tier golden oracle, extracted `pose_file_io`/`session_state` services, and the Qt5->Qt6 migration — the foundation the MVVM refactor is sequenced onto.

## Guidance

### 1. Hybrid test framework + headless CTest labels (U1/R11/R13/R14)

Use **Catch2 v3 for pure logic** (no Qt event loop) and **QtTest for QObject/QThread/QSignalSpy seams**. Wire both into CTest, declare them in `pixi.toml`, and add a `test` task. Make headless the **default**, keep any GPU/VTK/GUI test behind an explicit label:

```toml
# pixi.toml
[tasks.test]
cmd = ["ctest", "--test-dir", ".build", "--output-on-failure", "--timeout", "600", "-L", "headless"]
env = {"QT_QPA_PLATFORM" = "offscreen"}
```

CTest registration pattern (from `test/CMakeLists.txt`):

```cmake
find_package(Catch2 REQUIRED)
add_executable(jtml_test_direct_optimizer
    unit/test_direct_optimizer.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/direct_optimizer.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/direct_data_storage.cpp)
target_include_directories(jtml_test_direct_optimizer PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_direct_optimizer PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.direct_optimizer COMMAND jtml_test_direct_optimizer)
set_tests_properties(jtml.direct_optimizer PROPERTIES LABELS "headless" TIMEOUT 300)
```

- A **run-level timeout** (`ctest --timeout 600` + per-test `TIMEOUT`) converts any hang into a failing test, not a hung job (R13/AE5).
- The default `headless` label means **no GUI/display dependency, VTK render window, or widget**. Fast compute-only CUDA tests may initialize a GPU; expensive fixture/render-fidelity/performance gates remain under separate `oracle`/`gpu` labels.
- Pure-data and DIRECT tests remain compiled **CUDA-free** directly from their `.cpp` files rather than linking the CUDA-heavy compute target; allowing focused CUDA lifecycle tests does not weaken the pure-domain boundary.

### 2. CMake AUTOMOC gotcha: list Q_OBJECT headers in `add_executable`

AUTOMOC does **not** auto-moc an included shared header. If a Q_OBJECT class lives in a header you only `#include`, its moc is never generated → undefined-symbol link error. Fix: list the header in the target's sources. See the `jtml_test_coordinator` target (`test/CMakeLists.txt`) — note `include/coordinator/optimize_coordinator.h` explicitly listed.

**Sibling trap (disambiguation, 2026-08-11):** a `multiple definition of <Class>::<accessor>` between a `.cpp.o` and `mocs_compilation.cpp.o` is a DIFFERENT moc failure — a `signals:` section placed mid-class turns every following accessor into a signal (moc generates emitter bodies for them). Undefined symbol = header never moc'd (this section); duplicate definition = mis-scoped `signals:` region (see `docs/solutions/build-errors/jtml-moc-signals-section-placement-duplicate-definition-2026-08-11.md`).

### 3. The `file(GLOB)` trap in the layered lib `CMakeLists.txt`

Each layered lib (`src/{domain,services,coordinator,view}/CMakeLists.txt` — the former
single `src/core/CMakeLists.txt` was deleted in 003 U3) uses `file(GLOB ...)` for headers
**and** an explicit `.cpp`/`.cu` source list. Because GLOB only catches headers, a new
header globbed without its implementation in the explicit source list causes an
AUTOMOC undefined-symbol link error. The observed breakage: adding `direct_optimizer.h`
+ `optimize_coordinator.h` without their `.cpp` files. Fix — list every implementation
explicitly:

```cmake
add_library(jtml_domain STATIC
    data_structures_6D.cpp
    direct_data_storage.cpp
    direct_optimizer.cpp       # must be added explicitly
    ...
    ${HEADER_FILES})
```

Generalize: **any new core `.cpp` reached via the header GLOB must be added to the explicit source list**, or the build fails with a confusing AUTOMOC/moc link error.

### 4. Extract the pure algorithm behind an injected `std::function` (U5, R2/R15)

Extract the DIRECT loop (`ConvexHull`, `TrisectPotentiallyOptimal`, `DenormalizeRange`, `DenormalizeFromCenter`) into a CUDA-free class with the GPU touchpoint replaced by an injected cost callback (`include/domain/direct_optimizer.h`):

```cpp
class DirectOptimizer {
public:
    using CostFunction = std::function<double(const Point6D&)>;
    DirectOptimizer(CostFunction cost, Point6D range, Point6D starting_point,
                    unsigned int budget);
    bool Run();
    unsigned int GetCostFunctionCalls() const;
    Point6D GetOptimumLocation() const;
    double GetOptimumValue() const;
    void Stop();   // cooperative early-stop
};
```

Correctness rules to preserve during extraction (R15):

- **Cumulative budget is load-bearing.** The original zeroes `cost_function_calls_` only *before trunk*, then `+=` per stage. Reconciled authoritative caps (verified against `src/services/optimizer_settings.cpp` defaults + `OptimizerManager::Optimize`, not prose): **20k → 25k → 30k → 35k** = trunk 20,000 + **2 branches** 5,000 each + leaf 5,000 (`settings_constants`: trunk 20000, branch 5000, number_branches 2, leaf/z-search 5000). Do NOT "fix" it to per-stage caps, and ignore stale "~10k/stage" / "20k/25k/30k" prose in older handoff/golden text. The seed (center) evaluation consumes one budget unit — preserve that off-by-one exactly.
- **Cost is injected in denormalized physical space** — the callback receives the denormalized `Point6D`, not the unit-cube point.
- The `DirectOptimizer` may carry cumulative-offset + iteration/improvement callbacks for a throttled production UI (plan U6 design); the pure Tier-1 test uses defaults.

### 5. One persistent worker thread, not per-run threads (U4)

The headless `OptimizeCoordinator` owns the state machine and a **single persistent worker thread** reused across runs via a queued `RunRequested` signal (`src/coordinator/optimize_coordinator.cpp`). Do **not** create/`deleteLater` a thread per run — that caused a dangling-pointer segfault in the destructor. Teardown: `RequestStop(); quit(); wait(5000); delete worker_`.

Threading rules that make it spy-able and crash-free:

- The Q_OBJECT coordinator lives on the **main/test thread** and **re-emits** Succeeded/Failed/StateChanged there, so `QSignalSpy` observes on the test thread (QTBUG-2842 — never spy on the worker thread).
- Stop is **cooperative** (a flag polled by the worker); a mid-kernel stop may wait one iteration — accept and bound it. No cross-thread `Qt::DirectConnection`.
- Tests run under plain `QCoreApplication` (no widget/display), drive Idle→Running→Finished→re-launch, refuse double-start, handle injected cost-init failure, and are timeout-bounded (`QSignalSpy::wait`).

### 6. Two-tier, appearance-based golden oracle (U2/U6, R1/R2/R3)

The oracle is a **behavior-preservation gate, not a correctness check** (the baseline is captured from the current app, so it proves refactors preserved behavior, not that behavior is correct). Correctness comes from independent sources (Tier-1 analytic, known-good `Labels`).

- **Tier 1 (CI, headless, CPU):** the extracted `DirectOptimizer` converges on an analytic cost (e.g. `f = Σ(p−c)²`). Deterministic, bit-exact, default CI.
- **Tier 2 (GPU-labeled, GPU machine):** run the real pipeline on Kneel_1, **render the implant at the final optimized pose and compare the rendered silhouette to `Labels/fem/`** via pixel-diff / IoU. Do **not** gate on raw recovered pose values — DIRECT numeric convergence is noisy (CUDA reductions/atomics are not bit-reproducible; captured baseline vs `fem.jts` showed a ~6.3 mm z_trans gap while silhouette agreement is the load-bearing check). Pose-vs-`fem.jts` is informational only.

Authoritative fixtures under `example_studies/Kneel_1/` (mirrored to `test/golden/`): base silhouettes, `Labels/fem/`, `fem.jts` (JT_EULER_312), `KR_right_7_fem.stl`, `calibration.txt` (JT_INTCALIB), and `fem_oracle.jtak` (the captured Qt5/GPU baseline recorded in `test/golden/baseline.json`).

### 7. MVVM strangle: extract widget-free controllers + pure builders (U9/U10, AE4/R9/R15)

Shrink the `MainScreen` god object one seam at a time. For each extraction, pull the
widget-free **decision/state/packaging** out into a pure service; leave the **view
(render/color/widget binding)** and the **real production binding** in `MainScreen`.
Per-layer gate (R9): headless unit/coordinator for logic; compile + scheduled
manual-visual for presentation-only cuts. Two worked examples from this session:

- **Controller — `OptimizeIntentController` (`include/domain/optimize_intent_controller.h`):**
  owns the "can I optimize / what do I need" predicate + `Initialize` argument packaging
  (primary model = first selected row, current frame) that used to live inline in
  `MainScreen::LaunchOptimizer`. It takes **plain values** (no widgets) and returns a typed
  `Intent{status, primary_model_index, current_frame}`. `MainScreen` keeps only the error
  presentation + the real GPU `OptimizerManager` binding (R15 — do NOT rewire to the stub
  `OptimizeCoordinator`).
- **Builder — `ModelListBuilder` (`include/domain/model_list_builder.h`):** a pure `std::string`
  service owning the model name-dedup (two-pass "scan-restart" quirk preserved verbatim —
  `["A","A","A"]` → `["A(2)","A(3)","A"]`) + the frame same-size checks. View keeps only
  `addItem(...)` + VTK binding.

Key rules: keep production binding intact (R15); reproduce quirks exactly and pin them
with deterministic unit tests before "normalizing" behavior; back extracted pure logic with
a hegel property-based test next to its Catch2 cases (`test/HEGEL-PBT-GUIDE.md`); add any
new `.cpp` to the **explicit** source list of its layer's `CMakeLists.txt` (GLOB header-trap, §3).

### 8. Tier-2 oracle: multi-frame + per-frame empirical label (U11)

The Tier-2 appearance oracle is no longer frame-0-only. `test/oracle/oracle_test.cpp` now
loops all **3 Kneel_1 frames** (`1024/2806-2808.tif`), each with its **own** `fem.jts` start
pose (from `test/golden/baseline.json` `expected_pose_per_frame`) and its **own**
empirically-resolved label (start-pose IoU pick; never by filename) — all pass IoU > 0.85
(measured 0.9936 / 0.9910 / 0.9946 on the RTX 3090). Keep the silhouette/IoU gate (not raw
pose — DIRECT convergence is noisy), the vertical label flip, and the repo-root
`WORKING_DIRECTORY`.

## Why This Matters

- **Fearlessness:** the watched failure classes — "optimizer hangs after finishing", "button click hangs", "optimizer button doesn't fire" — live in thread/orchestration seams that used to be impossible to exercise without running the GUI. A headless coordinator seam + timeout-bounded tests turn each into a fast, non-interactive pass/fail.
- **A behavior-preservation net before touching validated numerics:** the Tier-1 analytic golden + Tier-2 appearance oracle let extracting DIRECT and (later) migrating Qt5→Qt6 be verified against recorded baselines, not silently re-derived.
- **The two abandoned-attempt anti-patterns are structurally prevented:** tests target seams (not god objects), and independent ground truth (analytic cost, known-good projections) replaces circular re-derivation.
- **Every change is gated:** no logic/service/coordinator extraction ships untested; `std::function` injection makes a future pure-core port (e.g., Rust) or MVVM decomposition safe to sequence change-by-change.

## When to Apply

Apply when you have a **C++ desktop app coupling pure algorithms to GPU and/or GUI** and you want to (a) start testing at all, (b) establish a regression net before a refactor or major upgrade, or (c) shrink a god object. It is the right call when:

- There is no test framework, CTest is disabled, or the only "tests" require a display/GPU.
- The expensive/validated math (optimizer, geometry, pose IO) is the part you most need to protect while editing.
- Hangs and thread-orchestration bugs are the dominant failure mode and only manifest by clicking the GUI.
- CUDA/numeric output is not bit-reproducible — so you need a tolerance/appearance-based gate, not an equality assertion.

The hybrid Catch2/QtTest split (pure math vs threading seam) is key: don't force pure numerics into QtTest, and don't put GUI/VTK-render or expensive fixture/performance oracles in the default suite.

## Examples

- `pixi run test` → `ctest -L headless --timeout 600`: no GUI/display or VTK render window; fast compute-only CUDA tests are allowed. `ctest -L oracle` runs the expensive Tier-2/render/performance gates.
- Tier-1 golden — `test/unit/test_direct_optimizer.cpp` (Catch2): converges an analytic quadratic to its known min; asserts budget accounting (seed consumes one unit) matches the cumulative cap.
- Lifecycle seam — `test/lifecycle/coordinator_test.cpp` (QtTest): stub cost drives Idle→Running→Finished→Idle, re-launches, refuses double-start, recovers from injected cost-init failure, and a stuck worker fails via timeout (AE5).
- MVVM controller+builder — `test/unit/test_optimize_intent_controller.cpp`, `test/unit/test_model_list_builder.cpp` (+ hegel PBT `test/unit/test_model_list_builder_properties.cpp`): lock the widget-free intent predicate and the dedup quirks.
- Tier-2 oracle (U11, multi-frame) — `test/oracle/oracle_test.cpp` (Catch2, `oracle` label): loops all 3 Kneel_1 frames, each empirically mapped to its label, optimizes, renders at the optimized pose, compares the silhouette to `Labels/fem/` (IoU > 0.85 gate). Labels are vertically flipped (bottom-left vs top-left y-origin); the oracle runs from the repo root (fixture-relative paths).
- Seam boundary — `include/domain/direct_optimizer.h` + `include/coordinator/optimize_coordinator.h`: `DirectOptimizer(std::function<double(const Point6D&)>, range, start, budget)`; `OptimizeCoordinator` re-emits Succeeded/Failed/StateChanged on the main thread for `QSignalSpy`.

## Related

- Requirements (normative): `docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md` (R1–R16, AE1–AE5)
- Plan (Units U1–U8, stable U-IDs): `docs/plans/2026-08-07-001-refactor-testability-mvvm-plan.md`
- MVVM controller-oracle continuation (U9/U10/U11): `docs/plans/2026-08-07-002-refactor-mvvm-controller-oracle-expansion-plan.md`
- Layered-dir restructure (domain/services/coordinator/compute/view/app) + its adversarial review:
  `docs/plans/2026-08-07-003-refactor-layered-directory-restructure-plan.md`,
  `docs/reviews/ce-adversarial-003-layered-restructure.json`
- Status / next steps: `docs/handoff-2026-08-07-testability-mvvm.md`
- Oracle spec: `golden_oracle.org`; baselines in `test/golden/`
- Working conventions: root `AGENTS.md`
