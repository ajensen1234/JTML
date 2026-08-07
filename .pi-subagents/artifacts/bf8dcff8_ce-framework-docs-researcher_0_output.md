I now have comprehensive research from official Qt/VTK docs, VTK Discourse, conda-forge, and codebase analysis. Here is my findings report.

---

# Framework Documentation Research Report — JTML Refactor

## 1. Summary
JTML is a Qt5 + VTK(9.3) + CUDA + OpenCV + libtorch C++20 GUI app for 3D knee-implant registration (DIRECT optimizer over a GPU cost function). The refactor needs: (1) a golden-oracle regression gate, (2) a headless-testable optimize-lifecycle seam, (3) a pure DIRECT optimizer behind an injected cost-function boundary, (4) MainScreen decomposition toward MVVM, (5) Qt5→Qt6 migration, plus a real test framework + CI. This report covers the Qt6/VTK migration and headless Qt/QObject/QThread testing best practices, grounded in concrete codebase findings.

I did **not** find the abandoned 18-test-file/wrapper attempt in the working tree (only clangd index artifacts matched "wrapper"); it is not recoverable from `jj st` output, so I flagged the deprecated APIs it would have touched (see Residual Risks).

---

## 2. Version Information (from codebase + lockfile)
| Dependency | Current | Notes |
|---|---|---|
| Qt | **5.15.8** (`pixi.lock`: `qt-5.15.8`, `qt-main-5.15.8`) | pinned `qt = "5.*"` in `pixi.toml:14` |
| VTK | **9.3.0** (manual build via `vtk_installer.sh`) | `find_package(VTK 9.3 ...)` in `CMakeLists.txt` |
| CUDA | 12.4 (`pixi.toml`) | |
| C++ standard | 20 (`CMakeLists.txt`) | |
| CMake | ≥3.31.4 | |
| Tests | none (none declared; `test/CMakeLists.txt` only has nfd + vtk-tester) | |
| CI | inert boilerplate (`.github/workflows/cmake.yml` triggers on `actions-test` branch, bare `cmake -B build`) | |

**Key app build facts (concrete):**
- Top-level `CMakeLists.txt` uses `find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)` and `Qt5::Core/Gui/Widgets` targets.
- GUI is already on **`QVTKOpenGLNativeWidget`** (the modern Qt5/Qt6-compatible widget) in `include/gui/mainscreen.ui` (lines 2151, 2470, 4225–4227) and `include/gui/drr_tool.ui` (625, 638). This is the *good* widget for Qt6 — no `QVTKWidget`→`QVTKOpenGLWidget` swap is needed.
- `src/gui/main.cpp` does **NOT** call `QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat())` before `QApplication`. This is required for correct VTK-in-QOpenGLWidget rendering (see Finding M-1, high severity).

---

## 3. Qt5 → Qt6 Migration Findings

### Qt library/CMake changes
- **M-1 [HIGH] `src/gui/main.cpp`:** Must add, before constructing `QApplication`:
  ```cpp
  QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
  ```
  `QVTKOpenGLNativeWidget` is a `QOpenGLWidget` subclass; Qt takes over OpenGL-context creation, and a VTK-appropriate `QSurfaceFormat` must be applied via `QSurfaceFormat::setDefaultFormat` (or `QOpenGLWidget::setFormat`) before any widget is created (VTK QVTKOpenGLNativeWidget class reference; official minimal example). Missing this is a latent bug in the current Qt5 build and becomes louder under Qt6.

- **M-2 [HIGH] `CMakeLists.txt` (find_package):** Replace `find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)` with `find_package(Qt6 COMPONENTS Core Gui Widgets REQUIRED)`, and rename `Qt5::Core`→`Qt6::Core`, `Qt5::Gui`→`Qt6::Gui`, `Qt5::Widgets`→`Qt6::Widgets` (all `Qt5::` references, incl. tests). Qt6 recommends `qt_standard_project_setup()` / `qt_add_executable` (Qt6-only); the version-agnostic `Qt::Core` imported targets also work with `find_package(Qt6)`.
  - The app and VTK must resolve the **same** Qt. Because `find_package(Qt6)` runs in a pixi env, pass `-DQt6_DIR=$CONDA_PREFIX/lib/cmake/Qt6` or set `Qt6_ROOT` (Qt docs: Making Qt available in CMake projects).
  - `pixi.toml` `[tasks.configure]` currently passes `-DQt5_DIR=$CONDA_PREFIX/lib/cmake/Qt5` → change to `-DQt6_DIR`.

- **M-3 [HIGH] `vtk_installer.sh`:** To build VTK 9.3 against Qt6, change:
  - `-DQt5_DIR=` → `-DQt6_DIR=$CONDA_PREFIX/lib/cmake/Qt6`
  - `-DVTK_USE_QT6=OFF` → `-DVTK_USE_QT6=ON`
  - `-DVTK_QT_VERSION=5` → `-DVTK_QT_VERSION=6`
  - `-DVTK_GROUP_ENABLE_Qt6=NO` → `=YES` (or remove; VTK 9 defaults the Qt group to Qt6 when Qt6 is found)
  - Keep `-DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES`.
  - VTK ≥9.0 supports Qt6; the Qt group auto-selects Qt6 when `Qt6_DIR` is found (VTK Discourse "Building VTK with Qt6"). Recommendation: **delete the Qt5 cache** (`rm -rf _deps/vtk/build _deps/vtk/install`) before reconfiguring so stale `Qt5*` cache variables don't pin Qt5 (discourse reports repeated "Could not find a configuration file for package Qt6 compatible with 5.9" when Qt5_DIR leaks in).

- **M-4 [MEDIUM, OPTIONAL — biggest risk-reducer]:** Prefer replacing the entire manual VTK build with the **conda-forge `vtk` package** (`pixi add vtk`), which ships VTK 9.x (up to 9.6) built against `qt6-main` on Linux. This removes `vtk_installer.sh`, `build-vtk` task, and all `VTK_QT_VERSION`/`VTK_USE_QT6` fragility at once; the trade-off is depending on conda-forge's VTK config (its `GUISupportQt` is Linux/macOS-only — Windows no-qt caveat, irrelevant here). Decide before writing migration work.

### Qt5-only API usage that breaks under Qt6
- **M-5 [MEDIUM] `src/Study2Grid/main.cpp` (lines 209, 284, 299):** uses `QRegExp(...)` with `Qt::SkipEmptyParts`. `QRegExp` was retired to the **Qt5Compat** module in Qt6 (Qt "Changes to Qt Core"). If Study2Grid is part of the Qt6 build, port to `QRegularExpression` (or add `Qt5::Core5Compat`). `StrUtil` aside, this is an auxiliary tool.
- **M-6 [LOW] `src/Study2Grid/*` and `src/core/*STLReader*.cpp`:** use `QTextStream` and `qPrintable` — both still valid in Qt6. No change needed.
- **M-7 [LOW] `Qt::SkipEmptyParts`, `QString::fromStdString`, `QAction`, `QMainWindow`:** all available in Qt6. No change.
- **M-8 [INFO] `#include "QVTKOpenGLNativeWidget.h"`** (in autogen `ui_mainscreen.h`/`ui_drr_tool.h`, triggered by the `.ui` `<header>` tag): resolves from VTK's `GUISupportQt` include dir regardless of Qt version — no change needed, but the generate must have the VTK Qt module + matching Qt include paths.

---

## 4. Headless Qt / QThread / QSignalSpy Testing Best Practices

### Recommended architecture (matches the task's optimize-lifecycle seam plan)
Use the canonical **worker-object + coordinator** pattern (Qt6 `QThread` class docs; KDAB "Eight Rules of Multithreaded Qt"):
- **Coordinator** is a `QObject` living in the **test/main thread**. It owns the `QThread`, constructs the worker, and is the object you spy on.
- **Worker** is a `QObject` subclass moved to the thread via `moveToThread` that runs the blocking DIRECT loop and re-emits progress/result signals.
- Connections: `QThread::started → Worker::Optimize`, `Worker::finished → QThread::quit`, `QThread::finished → Worker::deleteLater`, and `Worker::(raw signals) → Coordinator` (auto/queued), with the Coordinator re-emitting `stateChanged`/`finished`/`errorOccurred` **on the test thread**.
- Teardown in the coordinator destructor: `workerThread.quit(); workerThread.wait();` (Qt6 `QThread` class reference).

### Testing recipe (headless, zero GPU)
- Use **`QCoreApplication`** (not `QApplication`) for lifecycle tests — no widgets, no display needed for QObject + QThread + signal/slot + `QSignalSpy::wait()`.
- Use **`QSignalSpy`** on the **coordinator's** signals (main thread). Do **not** spy directly on a signal emitted from the worker thread: `QSignalSpy` connects via a direct connection, and signals emitted from another thread caused a known crash/UB (`QTBUG-2842`, "QSignalSpy crashes if signal is emitted from worker thread"; SO "QSignalSpy can not be used with threads"). Keeping the observed signal on the test thread entirely sidesteps this.
- Wait with `QSignalSpy::wait(timeout)` (Qt ≥6.6: `wait(std::chrono::milliseconds)`, default 5s) which runs a nested event loop, or `QTRY_VERIFY`/`QTRY_COMPARE`. **Avoid `QTest::qWait(N)` fixed sleeps** (Qt Test Best Practices) — they are flaky on slow/fast machines; especially relevant with a 10k iteration budget.
- **`QT_QPA_PLATFORM=offscreen`: only required if you instantiate `QApplication`/`QWidget`/`QVTK` widgets.** For pure `QCoreApplication`+worker state-machine tests it is unnecessary — a genuine advantage of the coordinator seam. Reserve `offscreen` (or Xvfb if code paths call the real X render stack) for any future smoke test that instantiates `QWidget` off a display (Qt Test Overview; QGIS headless CI guide; VoiceFlow CI example).

### Framework choice: QtTest vs Catch2 (recommendation)
- **Use QtTest** (`Qt::Test`) for the QObject/QThread/QSignalSpy tests because the seam *is* Qt threading — `QSignalSpy`, `QTRY_VERIFY`, and `QCoreApplication` are the native, well-trodden mechanisms (Qt Test Overview / Best Practices).
- **Catch2 is a valid complement for the PURE parts** — the extracted DIRECT solver (objective 3) and any pure math/data-structure tests — where no QObject event loop is needed and where Catch2's richer assertion/matcher ergonomics help. It must not be used to drive QThread/QSignalSpy; do that through QtTest.
- Concretely: declare `find_package(Qt6 COMPONENTS Core Gui Widgets Test REQUIRED)` for the test targets and add unit tests via `add_executable`+`target_link_libraries(... Qt6::Test)`, enableable under `BUILD_TESTING`/`JTML_BUILD_TESTS` (currently `test/CMakeLists.txt` has no linkage to a test lib nor any `BUILD_TESTING` guard).

---

## 5. DIRECT Optimizer Extraction (objective 3) — findings from the code

`src/core/optimizer_manager.cpp`'s DIRECT core is **already mostly GPU-pure**; only the cost-evaluation step couples to CUDA. Concretely:
- `ConvexHull()` (line 1495) — operates only on `data_` (`DirectDataStorage`), `potentially_optimal_col_ids_`. **No GPU.**
- `TrisectPotentiallyOptimal()` (line 1553) — uses `DenormalizeRange`, `HyperBox6D`, and calls `EvaluateCostFunction` at the two new centers. Otherwise **pure.**
- `DenormalizeRange` (1623) / `DenormalizeFromCenter` (1633) — pure arithmetic on `range_`/`starting_point_`.
- `SetSearchRange` (869) / `SetStartingPoint` (879) — pure.
- **`EvaluateCostFunction(Point6D)` (line 1437) is the ONLY GPU-touching step.** It (a) denormalizes, (b) builds `Pose`, calls `gpu_principal_model_->SetCurrentPrimaryCameraPose(...)`, does the biplane A→B conversion + secondary pose set, (c) calls `trunk_manager_/branch_manager_/leaf_manager_.callActiveCostFunction()` (which hit `GPUMetrics`/`GPUModel`/`GPUFrames`), (d) increments `cost_function_calls_`, tracks `current_optimum_*`, and emits `UpdateOptimum`.

**Boundary design (recommended):** Extract a pure class, e.g. `DirectSolver<6>` (owning `DirectDataStorage` + `budget_` + `range_`/`valid_range_` + the unit↔physical denormalization), exposing `Point6D optimize(std::function<double(const Point6D& physicalPoint)> evaluate)` or `double` convergence loop that iterates `ConvexHull()`/`TrisectPotentiallyOptimal()`. The eviction point is a `double eval(Point6D)` — the optimizer calls it with the **denormalized physical** point; the GPU-pose-setting + `CostFunctionManager` call remains in a concrete adapter *outside* the solver. This makes the solver unit-testable with a stub (e.g. paraboloid `f(x)=Σ(p−c)²` on the unit cube; assert it reaches a min near `c` and terminates under budget). Move `data_structures_6D.{h,cpp}`/`direct_data_storage` into the pure dependency set (currently `data_structures_6D.h` pulls in `gpu/render_engine.cuh` via `Point6D(gpu_cost_function::Pose)` — a **severity-MEDIUM** head: strip that CUDA ctor from the pure model to break the transitive GPU dependency).
- Objective 2's lifecycle test needs only the coordinator + a stub `eval` (no DIRECT at all); objective 3's pure test covers the solver itself.

---

## 6. Golden-Oracle Regression Gate (objective 1) — notes
- `golden_oracle.org` and `example_studies/Kneel_1` (femoral `KR_right_7_fem.stl`, `fem.jts`, `Labels/` projections) are the fixture; optimizer settings: trunk(±30)/branch(±20)/leaf(±3, z≈100) with DIRECT_DILATION, dilations 6/3/1px, ~10k budget each. 
- Practical gate: run the extracted `DirectSolver` with a **CPU-side eval stub** that reproduces DIRECT_DILATION against the Kneel_1 silhouettes (or record known-good optimum from the Qt5 run) and assert the recovered `Point6D` matches `fem.jts` within the leaf tolerance. Do not launch the full GPU GUI to gate — that's the anti-pattern the prior attempt rightly rejected.
- `test/vtk/test_case` is the separate LEFT-knee case; keep it out of the gate to avoid coupling.

---

## 7. CI + Test Tooling (concrete)
- `.github/workflows/cmake.yml` is inert boilerplate (triggers on `actions-test`, bare `cmake -B build`, `ctest`). Recommendation: gate on `find_package(Qt6 ... Test)`, set `QT_QPA_PLATFORM=offscreen` as a job env (harmless safety), install headless X/GL libs, and run `ctest`. For the pure solver + coordinator tests, offscreen is not strictly required (QCoreApplication path).
- pixi: add `qt`→Qt6 via `pixi add qt>=6` (or `qt6-main`) and `pixi add vtk` if adopting M-4; declare a test runner + a `[tasks.test]` / `test` feature. There is currently **no test framework in `pixi.toml`** and `test/CMakeLists.txt` adds executables but no test registration (`add_test`/`ctest`).

---

## 8. Best Practices / Anti-Boundary Summary (what to avoid)
- **Avoid** Qt-mocking wrappers and run-the-real-MainScreen characterization tests (prior attempt rejected — reaffirmed as correct).
- **Prefer** QtTest for thread/event-loop seams; Catch2 for pure math only.
- Keep all QSignalSpy observations on the main/test thread; coordinator re-emits worker progress.
- Short, deterministic waits via `QSignalSpy::wait`/`QTRY_`; no fixed sleeps.
- `QCoreApplication` is enough for the lifecycle seam (zero GPU, zero display); `offscreen` only for widget-level tests.

---

## References
- Qt6 QThread class (worker/controller pattern): https://doc.qt.io/qt-6/qthread.html
- Threads and QObjects: https://doc.qt.io/qt-6/threads-qobject.html
- QSignalSpy class (incl. `wait` since 6.6): https://doc.qt.io/qt-6/qsignalspy.html
- Qt Test Overview (offscreen): https://doc.qt.io/qt-6/qtest-overview.html
- Qt Test Best Practices (avoid fixed timeouts): https://doc.qt.io/qt-6/qttest-best-practices.html
- Porting to Qt 6 / Changes to Qt Core (QRegExp→Qt5Compat): https://doc.qt.io/qt-6/portingguide.html, https://doc.qt.io/qt-6/qtcore-changes-qt6.html
- Qt 5/6 CMake compatibility & find_package: https://doc.qt.io/qt-6/cmake-qt5-and-qt6-compatibility.html, https://doc.qt.io/qt-6/cmake-making-qt-available.html
- QVTKOpenGLNativeWidget class ref (defaultFormat requirement): https://vtk.org/doc/nightly/html/classQVTKOpenGLNativeWidget.html
- VTK Qt6 build issues (Qt5_DIR leak): https://discourse.vtk.org/t/building-vtk-with-qt6/12534
- QTBUG-2842 (QSignalSpy + worker thread crash): https://bugreports.qt.io/browse/QTBUG-2842
- conda-forge vtk (Qt6 packages): https://anaconda.org/conda-forge/vtk ; conda-forge Qt6 status: https://conda-forge.org/blog/2026/07/01/qt6-status-in-conda-forge/

---

## Acceptance Report

I did not modify any files (research-only task). Evidence is based on direct file reads and cited official sources.