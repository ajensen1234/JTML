I have completed a thorough investigation of the repository. Here is my research report.

## Repository Research Summary — JTML (Joint Track Machine Learning)

### Technology & Infrastructure

**Stack (all resolved via pixi/prefix.dev, conda-forge + nvidia channels):**
- **C++20** (CUDA 20), CMake ≥3.26, Ninja generator, pixi env named `joint-track-machine-learning`
- **Qt 5.x** — pinned `qt = "5.*"` in pixi.toml; `find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)`; RPATH points at conda prefix
- **VTK 9.3** — NOT a pixi dep; built separately via `vtk_installer.sh` which hard-forces `-DVTK_USE_QT6=OFF` and `-DVTK_QT_VERSION=5` (line 76–77). Found under `_deps/vtk/install/lib{,64}/cmake/vtk-9.3/`
- **OpenCV 4.10** (qt5 build — see pixi.lock), **Torch** (pytorch-gpu ≥2), **Eigen3**, **CUDA 12.4** (cuda, cuda-version, nvtx, ccache, gcc, clang-tools 19.x, ninja, glew, libopengl, sdl2)

**Deployment model:** single desktop GUI application (monolithic, not serverless/multi-service). Entry: `src/gui/main.cpp` → `QApplication` + `MainScreen`.

**API style:** none external — it's a native GUI + in-process libraries, no REST/gRPC/GraphQL surface.

**Data/async patterns:** CUDA GPU kernels (`src/gpu/*.cu`, compiled into shared `jtml_gpu`); a QThread-based optimizer worker (`OptimizerManager` moved to a `QThread` inside `MainScreen::LaunchOptimizer`). OpenCV for image dilation; nested-vector `DirectDataStorage` for the DIRECT search.

**Module organization:**
```
src/
  core/          -> jtml_core (STATIC): optimizer_manager, model, frame, stl_reader, optimizer_settings, machine_learning_tools, DirectDataStorage, sym_trap...
  gpu/           -> jtml_gpu (SHARED): all .cu kernels (metrics, rendering, dilation, distance maps, heatmaps, model)
  cost_functions -> JTA_Cost_Functions (SHARED): CostFunction{Manager}, DIRECT_DILATION, DIRECT_MAHFOUZ, sym_trap, pole-constraint variants
  gui/           -> executable joint-track-machine-learning: mainscreen, viewer, controls, settings_control, drr_tool, about
  Study2Grid, shape_sensitivity -> auxiliary (not wired into the app; shape_sensitivity commented out)
include/{core,gpu,cost_functions,gui}/  -> all headers
```
Not a monorepo; single CMake project with subdirectories. Build output dirs `.build/bin`, `.build/lib`.

### Build / Test Wiring

**pixi tasks (`pixi.toml`):** `build-vtk`, `configure`, `build`, `run`, `check-display`, `format`, `tidy`. **There is NO `test` task.** Environments: `build` and `dev`.
- `configure` uses `-GNinja -S. -B.build` with `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX`, conda CUDA/OpenCV/Torch/Qt5 dirs, `-DCMAKE_BUILD_TYPE=Release`, exports compile_commands.
- `build` = `cmake --build .build`, depends on configure+build-vtk.

**CMake target graph:**
- `jtml_gpu` (SHARED) — all `.cu` kernels; links OpenCV + `CUDA::cublas/cudart/curand`; `CUDA_SEPARABLE_COMPILATION ON`.
- `JTA_Cost_Functions` (SHARED) — cost functions; links Qt5 libs + Torch + OpenCV + `CUDA::cudart/cublas` + **`jtml_gpu`**.
- `jtml_core` (STATIC) — includes `frame.cu`, `optimizer_manager.cpp`, etc.; links Qt5 + Torch + VTK + OpenCV + CUDA + `JTA_LIBS` (= `jtml_gpu JTA_Cost_Functions`).
- `joint-track-machine-learning` (executable) — GUI; links Qt5 + `jtml_core` + `JTA_LIBS` + VTK + Torch + OpenCV + Eigen3.

**Test target wiring: ABSENT.** In root `CMakeLists.txt`: `#add_subdirectory(test)` and `#enable_testing()` are **commented out**. No `ctest`, no test-framework package in pixi (verified: no Catch2/gtest/doctest in pixi.lock). `test/nfd/` and `test/vtk/` each carry their own (unused) CMakeLists; `test/vtk/test_case` is a **left-knee** fixture (`KR_left_8_*` per D2 in the brainstorm doc) — not usable for the Kneel_1 oracle.

**CI: INERT.** `.github/workflows/cmake.yml` triggers only on branch literally named `actions-test`, runs plain `cmake -B build` (no pixi/CUDA/Qt dirs — would fail), and `ctest` with nothing enabled. No pixi setup, no test runner.

### optimizer_manager.cpp — private methods & data flow (`src/core/optimizer_manager.cpp`, 1722 lines)

**Private methods and where the DIRECT algorithm lives:**
| Method | Line | Role |
|---|---|---|
| `SetSearchRange(Point6D)` | 869 | stores `range_`, sets `valid_range_` if nonzero |
| `SetStartingPoint(Point6D)` | 879 | stores `starting_point_` |
| `Optimize()` (slot, runs on QThread) | 883 | per-frame trunk→branch→leaf orchestration loop |
| `CalculateSymTrap()` | 1336 | fidelity-analysis cost sweep |
| `EvaluateCostFunctionAtPoint(Point6D, stage)` | 1411 | non-denormalized eval for sym-trap |
| `EvaluateCostFunction(Point6D)` | 1437 | **the GPU-coupled cost entry** used by the DIRECT loop |
| `ConvexHull()` | 1495 | Jarvis-march (gift-wrapping) over `data_` columns → `potentially_optimal_col_ids_` |
| `TrisectPotentiallyOptimal()` | 1553 | populates/get/delete hyperboxes, trisects largest side, calls `EvaluateCostFunction` |
| `DenormalizeRange(Point6D)` | 1623 | unit → side length (×2×range_) |
| `DenormalizeFromCenter(Point6D)` | 1633 | unit → real (from `starting_point_`, ×2×range_) |
| `onStopOptimizer()` | 1643 | sets `error_occurrred_ = true` |

**Data flow of the DIRECT loop** (Trunk section, lines ~1020–1075; identical pattern repeated for branch/leaf):
1. `SetSearchRange(stage.range)`; `SetStartingPoint(...)`
2. reset `cost_function_calls_ = 0` (before trunk) then `budget_ = trunk_budget`
3. `DirectDataStorage(current_optimum_value_)` seeded with unit hyperbox at `(.5,.5,.5,.5,.5,.5)`
4. `while (cost_function_calls_ < budget_) { ConvexHull(); TrisectPotentiallyOptimal(); …30fps UpdateDisplay; }`
   - `ConvexHull()` reads `data_.GetNumberColumns/GetMinimumHyperboxValue/GetSizeStoredInColumn`
   - `TrisectPotentiallyOptimal()` picks largest **denormalized** side (`DenormalizeRange(...).GetLargestDirection()`), `TrisectSide`, and calls `EvaluateCostFunction` at the two new centers
5. emits `OptimizedFrame(optimum…, error, directive)`, updates `pose_storage_`, loops to next frame; then `finished()`.

**End-to-end trigger:** `MainScreen::LaunchOptimizer` (`src/gui/mainscreen.cpp` ~4652) creates `OptimizerManager`, `moveToThread(optimizer_thread)`, calls `Initialize(...)`, wires ~9 signals (UpdateDisplay, OptimizerError, UpdateOptimum, OptimizedFrame, StopOptimizer*, UpdateDilationBackground, onUpdateOrientationSymTrap), then `optimizer_thread->start()` → queued slot `Optimize()`.

### Implementation seams — what DIRECT *actually depends on* vs GPU-only

**Pure / GPU-independent DIRECT core (extraction candidates — safe to unit-test with a stub eval):**
- `ConvexHull()` — touches only `data_`, `potentially_optimal_col_ids_`, and sets `error_occurrred_`
- `TrisectPotentiallyOptimal()` mechanics (hyperbox trisection, center updates) *except* its two `EvaluateCostFunction(...)` calls
- `DenormalizeRange` / `DenormalizeFromCenter` / `SetSearchRange` / `SetStartingPoint` — pure math on `Point6D`
- `stage_orchestration`: the trunk/branch/leaf loops and budget logic
- Data structures: `DirectDataStorage`, `HyperBox6D`, `Point6D` (`include/core/direct_data_storage.h`, `data_structures_6D.h`, `src/core/direct_data_storage.cpp`)

**GPU-coupled (what must stay behind an injected `double eval(Point6D)` boundary):**
- `EvaluateCostFunction(...)` (1437) — calls `DenormalizeFromCenter`, then `gpu_principal_model_->SetCurrentPrimaryCameraPose(pose)`, biplane `convert_Pose_A_to_Pose_B`, then `{trunk,branch,leaf}_manager_.callActiveCostFunction()`
- `Initialize(...)` / destructor — builds all GPU frames/models/metrics (`GPUIntensityFrame/GPUEdgeFrame/GPUDilatedFrame/GPUFrame/GPUHeatmap/GPUModel/GPUMetrics`), uploads distance maps & heatmaps

**Cost function → CUDA mapping** (`CostFunctionManager::callActiveCostFunction` → `costFunctionDIRECT_DILATION` in `src/cost_functions/DIRECT_DILATION.cpp`):
```
gpu_principal_model_->RenderPrimaryCamera(pose)
score = pinned_sum_white_pix_A + gpu_metrics_->FastImplantDilationMetric(rendered, dilated_frame_A, dilation)
      + gpu_metrics_->DistanceMapMetric(rendered, distance_map, dilation)
(+ biplane: render secondary, squared dist score)
```
All heavy lifting is `GPUModel::Render*` + `GPUMetrics::FastImplantDilationMetric/DistanceMapMetric/ComputeSumWhitePixels` — 100% GPU. So the seam is clean: the loop consumes `double eval(Point6D)`; the manager layer is the only GPIO touchpoint.

**Seam-shaping observations:**
- `double OptimizerManager::EvaluateCostFunction(Point6D)` (already signature-compatible with a `std::function<double(Point6D)>` / `double(Point6D)` boundary) is the natural injection point.
- The DIRECT loop also reads two members: `cost_function_calls_` (for the budget guard) and `current_optimum_value_`, and emits `UpdateOptimum`/`UpdateDisplay`. An extracted pure core would take an eval callback plus an iteration-budget integer and return the argmin — later re-emit signals at the coordinator.
- **Budget is cumulative across stages, not per-stage reset:** in trunk `budget_ = trunk_budget`; branch does `budget_ += branch_budget`; leaf `budget_ += leaf_budget`; and `cost_function_calls_` is only zeroed before trunk. So with a 10k-per-stage setting the guard is `calls < 10000` (trunk), `< 20000` (branch), `< 30000` (leaf). **Plan/planner must not assume per-stage 10k budgets** — the effective iteration budget is cumulative. An implementation seam worth clamping in the extracted core.
- **Coupling to break for a pure DIRECT extraction:** `include/core/data_structures_6D.h` and `include/core/direct_data_storage.h` both `#include "gpu/render_engine.cuh"` solely for the `Point6D(gpu_cost_function::Pose)` constructor and `Direction` enum — a GPU include reaching into pure data structures. Extracting DIRECT cleanly requires removing/relocating that Pose-cast constructor.
- `optimizer_manager.h` drags in 6 CUDA `.cuh` headers + Qt + VTK + cost-function headers — the header itself is the heavy coupling; the pure core should live in a new dep-light TU.
- Cross-thread stop uses `Qt::DirectConnection` (`main.cpp`-adjacent `connect(this, StopOptimizer, optimizer_manager, onStopOptimizer, Qt::DirectConnection)` at mainscreen.cpp ~4742): a known-thread-safety smell near the exact seam (R5/R7 hang/stop paths).

### Architecture & Structure

- **God-object `MainScreen`** (`src/gui/mainscreen.cpp`, 5806 lines + header `include/gui/mainscreen.h`) mixes: UI wiring (`Ui::MainScreenClass ui`), VTK render binding (vtkSmartPointer actors/renderers/viewers), app state (frames, models, calibration, pose storage), persistence slots (`on_actionSave_Pose/Load_Pose/Load_Kinematics…`), ML/segmentation slots, **and** optimizer thread orchestration (`LaunchOptimizer`, the manager, all its signal wiring) in one class.
- **Viewer abstraction exists:** `Viewer` (via `std::shared_ptr<Viewer> vw`, `coronal_vw`) already wraps VTK render-window operations (`include/gui/viewer.h`) — a useful existing seam to build on for the MVVM view layer.
- Cost/optimizer orchestration layer is `OptimizerManager` + `CostFunctionManager` + `CostFunction`; the three managers (`trunk_manager_/branch_manager_/leaf_manager_`, one per DIRECT stage) are created in `MainScreen` and passed by value into `Initialize`.
- The `OptimizerManager` is already positioned as a QObject that could be morphed into/behind a headless coordinator — but it currently *owns* the worker thread lifecycle wiring (`connect(thread started → Optimize)`, finished → quit/deleteLater) inside `Initialize`, i.e., orchestration is entangled with CUDA init and the GUI wiring lives in the slot. A coordinator extracting the state machine out of `Optimize()` is the R4 seam.

### Implementation Patterns

- **Existing seam candidates to follow:** `Viewer` (VTK render wrapper), `MainScreen_size_constants.h`/`settings_*_constants.h` (constant extraction precedent), `LocationStorage`/`pose_matrix` (persistence data classes), `OptimizerSettings` (plain struct of settings — easy to seed an oracle), `Stage` enum (`Stages.h`: Trunk/Branch/Leaf), `search_stage_flag_`.
- **Naming/org conventions:** PascalCase classes/methods; `_`-suffix private members (e.g., `frames_A_`, `optimizer_settings_`); regional header comment banners; `namespace jta_cost_function` / `gpu_cost_function`; SPDX + copyright header on every file (`// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab`); AGPL-3.0 license.
- **Coding style enforced:** `.clang-format`/`.clang-tidy`/`.clangd` present; `format.sh`/`tidy.sh` wrap them; clang-tools pinned 19.x. `just` recipes: `format`, `tidy`, `all`, `format-check`, `tidy-fix`, `fix`.
- **Signal/slot style:** old-style `SIGNAL(...)`/`SLOT(...)` string macros everywhere (Qt5) — a Qt6 migration concern (much of the `connect` uses string-based connections which Qt6 still supports but deprecates).

### Documentation / Guidelines & project constraints (AGENTS.md / justfile)

- **No AGENTS.md or CLAUDE.md exists** anywhere in the tree (checked root and recursively to depth 2). `.claude/skills/` and `.memory-bank/**` are empty scaffolding directories. The system-level AGENTS/skill instructions you inherit are the only agent guidance.
- **The authoritative plan/reference documents are:**
  - `golden_oracle.org` — the oracle spec (Kneel_1 fixtures, optimizer settings, DIRECT_DILATION primary).
  - `docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md` — the requirements doc **R1–R16**, acceptance examples AE1–AE5, and scope boundaries. **The plan must honor this contract.**

**Constraints from R1–R16 that directly affect the plan (from the brainstorm doc — treat as binding):**
- R1/R2: oracle baseline from the *known-good Qt5 app* first, before any refactor; **no test may re-derive the code-under-test's math (anti-circularity)** — process the prior abandoned attempt's lesson (18 test files + `qt_wrappers.h` + characterize-real-god-object tests were rejected).
- R4/R5/R6/R7: headless, widget-free coordinator owning the state machine (`idle→running→finished→idle` + error/cancelled) + worker thread, tested under `QCoreApplication` + `QSignalSpy` with a stub cost and a **timeout so hangs fail tests**; covers init-failure and stop paths headlessly; real GPU-init-failure confined to the explicit GPU target only.
- R11: enable CMake testing; **default run headless (no GUI/VTK/GPU)**; real-VTK/GPU cases in a separate explicitly-triggered target/ctest label.
- R12: **no Qt-mocking wrappers, no instantiate-the-real-MainScreen tests** (explicit rejects).
- R13: working CI on the pixi build running the headless suite, all lifecycle tests timeout-bounded.
- R14: add the test framework **as a declared pixi dependency** (currently absent — verified).
- R15: preserve validated DIRECT/cost logic; gate with oracle/convergence, don't silently re-derive.
- R16: Qt6 migration = update `qt` pin to `6.*`, rebuild VTK against Qt6 (`vtk_installer.sh` line 76–77 must flip `VTK_USE_QT6=ON`/`VTK_QT_VERSION=6`), switch to `find_package(Qt6)`, revalidate Qt5 APIs in MainScreen; gated by pre-migration baseline; bounded manual GUI smoke step.
- R8–R10: MVVM decomposition is sequenced **after** oracle + lifecycle + DIRECT/cost seams are green — not a prerequisite of fast headless pass/fail. Each extraction lands with its gate green before the next cut.

**justfile constraints:** only formatting/tidying recipes (`format`, `tidy`, `all`, `format-check`, `tidy-fix`, `fix`). No test/build recipes in `just`; build is via `pixi run build`. No policy that blocks adding tests, but the format/tidy gates imply new code must pass clang-format/clang-tidy.

**Residual risks / open items the plan should carry (from the brainstorm doc "Outstanding Questions" + my findings):**
- Pre-flight gate D3: the Qt5 GPU baseline must be captured by the user on a GPU machine before Phase 1; fallback = defer Qt6 migration.
- Oracle reference/tolerances (CPU vs GPU DRR/cost, numeric tolerance) are unresolved — load-bearing precondition of the golden gate (R3/AE2).
- `golden_oracle.org` optimizer-settings correctness (the brainstorm doc flags an inconsistency in the dilation list; the current file reads trunk 6px/branch 3px/leaf 1px — matches your stated settings, but confirm against the app defaults).
- Test-framework choice (Catch2 vs QtTest) is a deferred planner decision; must be added to pixi (R14).
- OpenCV is a **qt5** build variant in pixi.lock — a Qt6 migration may need to coordinate OpenCV's Qt binding, another migration ripple beyond the pin/VTK.

---

## Recommendations for the plan (given the consumer's acceptance contract)

1. **Gold: extract `DirectOptimizer` as a pure, dep-light `double eval(Point6D)` loop** — move `ConvexHull`/`TrisectPotentiallyOptimal`/`Denormalize{FromCenter,Range}` plus the stage budget loop into a new TU (e.g. `src/core/direct_optimizer.{h,cpp}`) that owns `DirectDataStorage`/`HyperBox6D`. Feed it a `std::function<double(Point6D)>` so DIRECT is testable with an analytic cost (R2's independent-correctness source). The existing `OptimizerManager::EvaluateCostFunction` becomes the real GPU-backed eval callback — **no rewiring of the validated math**, honoring R15.
2. **Break the GPU header leak first:** remove `#include "gpu/render_engine.cuh"` from `data_structures_6D.h`/`direct_data_storage.h` (relocate the `Point6D(Pose)` ctor). This is the single move that lets DIRECT compile/test without CUDA for the headless target (R11/AE3).
3. **Preserve the cumulative-budget semantics** when extracting the loop (trunk→`budget_=trunk`, then `+=branch`, `+=leaf`, cumulative `cost_function_calls_`) — clamp/assert it so the oracle gate uses the *effective* cumulative budget.
4. **Coordinator seam:** refactor `Optimize()`-body orchestration (per-frame stage selection, budget, start/stop, completion) out of `OptimizerManager` into a headless `OptimizeCoordinator` QObject owning the state machine and a worker so it can be driven under `QCoreApplication`+`QSignalSpy` (R4/R5). Keep `OptimizerManager` as the GPU/CUDA initialization + eval adapter. Fix the `Qt::DirectConnection` StopOptimizer wiring as part of this (R7 stop path).
5. **Test/build wiring:** add a Catch2 (or QtTest) dev dependency in pixi (R14), un-comment `enable_testing()`/`add_subdirectory(test)`, add a `headless` ctest label for the pure-DIRECT + coordinator suite (no GPU, `QT_QPA_PLATFORM=offscreen`), and a separate `gpu`/`oracle` label behind an explicit flag for the Kneel_1 numeric gate. Bind every lifecycle test with a timeout (R13). Add a `pixi run test` task.
6. **MVVM only after seams green** (R8–R10): lean on the existing `Viewer` abstraction for the view layer; the coordinator + cost + persistence services come out of `MainScreen` first.
7. **AVOID entirely (R12 + prior-attempt lesson):** Qt-mocking wrappers (the old `qt_wrappers.h` pattern), and any "instantiate the real `MainScreen`" characterization test.

---