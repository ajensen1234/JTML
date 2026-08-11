All research phases complete. Here is the full report for the scoped phases (`technology`, `architecture`, `patterns`).

---

## Repository Research Summary

**Repo:** JTML — Qt6 + VTK 9.3 + CUDA 12.4 knee-implant registration app. This is a research-only pass; no files were modified.

### Technology & Infrastructure

**Stack (from `AGENTS.md`, `pixi.toml`, root `CMakeLists.txt`):**
- C++20 + CUDA 20; Qt **6.7.2** (`qt6-main`/`qt6-wayland`, conda-forge via pixi); VTK **9.3 built in-tree against Qt6** (`_deps/vtk/install/` via `vtk_installer.sh`, `GUISupportQtQuick` included); CUDA 12.4, OpenCV, Torch, Eigen3, Catch2 3.x, hegel PBT (FetchContent).
- All build/test entry points are pixi tasks (`pixi run configure/build/test/run`) — never raw cmake. `pixi run test` = `ctest --test-dir .build --output-on-failure --timeout 600 -L headless` with `QT_QPA_PLATFORM=offscreen` (pixi.toml `[tasks.test]`). `pixi run run` forces `QT_QPA_PLATFORM=xcb`.
- **Qt6::QuickTest availability — CONFIRMED AVAILABLE (net-new, unused):**
  - `.pixi/envs/default/lib/cmake/Qt6QuickTest/Qt6QuickTestConfig.cmake` + `Qt6::QuickTest` target (verified in `Qt6QuickTestTargets.cmake`), `libQt6QuickTest.so.6.7.2` in `.pixi/envs/default/lib/`.
  - Tooling in `.pixi/envs/default/bin/`: `qmltestrunner`, `qmllint` (1.0), `qmlprofiler`, `qmlformat`, `qmlcachegen`, `qmlimportscanner`.
  - Offscreen platform plugin present (`libqoffscreen.so` in `.pixi/envs/default/plugins/platforms/`); `qmltestrunner -help` runs cleanly under `QT_QPA_PLATFORM=offscreen` — so a headless 2D-chrome QuickTest suite (no VTK render item) is viable under the existing test-task env. Caution: the env also ships Qt5 (`Qt5QuickTest` cmake dirs) — do not link the Qt5 flavor.
- **QML is shipped via qrc + qmlRegisterType, not `qt_add_qml_module`** (the planning context's claim verified):
  - `src/app/experimental/resources.qrc` → prefix `/`: `main.qml`, `SettingsPanel.qml`, `Theme.qml`, `PoseCell.qml`, `qmldir`. `renderer.qrc` → `renderer.qml` (smoke scene).
  - `qmldir` (2 lines): `singleton Theme 1.0 Theme.qml` + `PoseCell 1.0 PoseCell.qml`.
  - `main.cpp`: `qmlRegisterType<QmlVtkRenderer>("jtml.experimental", 1, 0, ...)` + `qmlRegisterUncreatableType<OptimizerBridge>(...)`; all bridges exposed via `rootContext()->setContextProperty` (appBridge, fileDialogBridge, studyBridge, settingsBridge, optimizerBridge, mlBridge, poseBridge). AUTORCC is global (root CMakeLists.txt), so listing the .qrc in sources is sufficient.
- **Deployment model:** single desktop executable per front-end; CI = `.github/workflows/cmake.yml` (setup-pixi → configure → build → headless test, 90-min timeout). No container/serverless/IaC.

**QML surface metrics (files in scope of the improvement pass):**

| File | Lines | Role |
|---|---|---|
| `src/app/experimental/main.qml` | **1016** (grew past the ~700 estimate) | Window shell: toolbar, 9 FileDialogs, 2 message/replace Dialogs, settings+pose Dialogs, 2 ListViews, ML strip, progress bar, 5 `Connections` glue blocks |
| `src/app/experimental/SettingsPanel.qml` | 361 | Dialog form; inline `component RangeField`/`IntField` (legal here — not the engine root) |
| `src/app/experimental/PoseCell.qml` | 40 | Editable pose cell; note: inline components are NOT supported in engine-root documents (main.qml loaded by URL) |
| `src/app/experimental/Theme.qml` | 19 | `pragma Singleton` palette (12 tokens) |
| `src/app/experimental/renderer.qml` | 23 | Smoke scene |

**C++ side (all in `src/app/experimental/`):** `AppBridge` hub owning StudyBridge/SettingsBridge/OptimizerBridge/MlBridge/PoseBridge + `FileDialogBridge`, `QmlVtkRenderer` (QQuickVTKItem subclass), `ExperimentalScene`/`ExperimentalSession`, `DelegateSelection`. FrameListModel/ModelListModel are direct-compiled from `src/view/` (widget-free QAbstractListModels).

### Architecture & Structure

**Layered-lib rules (AGENTS.md "003 layered layout" + `src/CMakeLists.txt`):** `src/domain` (pure, Qt/GPU-free) → `src/services` (non-pure headless) → `src/coordinator` (QObject orchestration) → `src/compute` (GPU/CUDA, single SHARED `jtml_compute` = `JTA_LIBS`) → `src/view` (STATIC `jtml_view`, QWidgets, QML-swappable by design) → `src/app` (composition root only). Each layered lib CMakeLists uses `file(GLOB)` for headers **plus an explicit .cpp source list** — new .cpp files must be added explicitly or AUTOMOC produces undefined-vtable link errors (documented gotcha).

**QML decision record — `src/view/CMakeLists.txt:6-14` (verbatim substance):** seams are QML-ready; the full QML front-end is **DEFERRED**; `src/app/experimental` (`jtml_experimental`) is the **sanctioned parallel vehicle**; revisit a full QML front-end on a concrete UI need QWidgets cannot deliver, or Qt 6.8+/VTK 9.4+ integration maturity. The improvement pass runs inside that sanctioned vehicle.

**`jtml_experimental` target mechanics (`src/app/experimental/CMakeLists.txt`):**
- Directory-scoped `find_package(Qt6 COMPONENTS Quick Qml QuickControls2 REQUIRED)` + re-find of VTK 9.3 with `GUISupportQtQuick` (root find_package deliberately untouched); autoinit MODULES must stay explicit against the directory-scoped `VTK_LIBRARIES`.
- Link set = widgets-app set **minus `jtml_view`**; `jtml_coordinator` PUBLIC-pulls services→domain→compute. `vtk_module_autoinit(TARGETS jtml_experimental MODULES ${VTK_LIBRARIES})` is mandatory (missing → base-class `vtkRenderWindow::New()`, segfaults).
- rpath recipe (conda toolchain overrides BUILD_RPATH): `-Wl,-rpath,$ORIGIN/../lib` + `$ENV{CONDA_PREFIX}/lib` + `${VTK_LIB_DIR}`.
- Every Q_OBJECT header listed in `add_executable` (AUTOMOC gotcha).

**Render-thread contract (`QmlVtkRenderer.h` file comment, load-bearing for any test/profiling work):** all VTK objects created in `initializeVTK()` and reachable only from `initializeVTK`/`destroyingVTK`/`dispatch_async` bodies; GUI-thread slots copy scene state to locals and capture **by value** (no `this`, no member reads in lambdas); scene-graph node recreation re-runs `initializeVTK` from an app-thread mirror. `setScene` is non-owning (app-owned `ExperimentalScene` outlives the item — that is what makes destroy-mid-update safe). Pose sync-back uses direct by-value signal emit (queued `invokeMethod` functor **silently fails** — documented in `docs/solutions/ui-bugs/jtml-qml-model-pose-sync-queued-functor-never-delivered-2026-08-11.md`).

**Docs conventions (all under `docs/`):**
- Plans: `docs/plans/YYYY-MM-DD-NNN-<type>-<name>-plan.md` — type ∈ `feat`/`refactor`; YAML frontmatter (`title`, `type`, `status`, `date`, `origin`); body: Overview → Problem Frame → Requirements Trace (R#/F#/AE#) → Scope Boundaries → per-Unit sections with Verification gates. Numbers 001–006 used; **the improvement pass would be 007**.
- Brainstorms/requirements (normative contract): `docs/brainstorms/YYYY-MM-DD-<name>-requirements.md` (R1..Rn, F1..Fn, AE1..AEn).
- Handoffs: `docs/handoff-YYYY-MM-DD-<name>.md` — "Read first" block, status, next phase. Latest: `docs/handoff-2026-08-11-vm-layer-extraction.md` (plan 006 complete, 46/46 headless; NEXT = Follow-On Optimization Phase, synthesis items 1→3→2→5→8 — an unrelated arc, so the QML polish pass is a new plan, not a continuation).
- Solutions: `docs/solutions/<category>/jtml-<topic>-YYYY-MM-DD.md` with YAML frontmatter (`date`, `last_updated`, `module`, `tags`, `problem_type`, `severity`, `applies_when`). Categories: `conventions/`, `tooling-decisions/`, `ui-bugs/`, `logic-errors/`, `build-errors/`. **Directly relevant: `docs/solutions/conventions/jtml-qml-experimental-frontend-2026-08-11.md`** (QML compound: render-thread contract, sync pattern, Theme/qmldir conventions, versionless imports, Material Dark, `grabWindow` capture, xdg-portal lesson) and `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md` (headless/AUTOMOC/CTest guidance).
- Reviews: `docs/reviews/ce-adversarial-<NNN>-<topic>.json` (adversarial review artifacts).

**AGENTS.md guidance that materially affects the plan:** headless-label rule (never GPU/VTK-render/widget), AUTOMOC gotcha, Q_OBJECT-header-in-source-list, layered-lib explicit-source-list rule, QtTest-for-seams/Catch2-for-pure split, hegel-PBT-complements rule, jj workflow (`jj describe -m "<scope>: <msg>"` then `jj new`), never raw git/cmake/pip.

### Implementation Patterns

**Test registration pattern (`test/CMakeLists.txt` — single flat file, 1447 lines; `test/nfd` + `test/vtk` deliberately NOT registered):**
```
add_executable(jtml_test_<name> unit/<name>_test.cpp [direct-compiled sources + Q_OBJECT headers])
target_include_directories(... PRIVATE ${PROJECT_SOURCE_DIR}/include [${PROJECT_SOURCE_DIR}/include/compute] ...)
target_link_libraries(... PRIVATE Catch2::Catch2WithMain [Qt6::Core Qt6::Test] [hegel dl] [VTK/OpenCV/Torch/CUDA])
target_link_options(... PRIVATE "-Wl,-rpath,...")
add_test(NAME jtml.<name> COMMAND jtml_test_<name>)
set_tests_properties(jtml.<name> PROPERTIES LABELS "headless" TIMEOUT <60..300>
    [WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"])   # fixtures: example_studies/Kneel_1, test/golden
```
- **Labels:** `headless` (default run; must never touch GPU/VTK-render/widget), `oracle` (GPU, TIMEOUT 3600), `render` (always paired `"oracle;render"`, forced xcb env). QtTest targets use `find_package(Qt6 COMPONENTS Core Test REQUIRED)` near the top of test/CMakeLists.txt; the QML smoke uses a `block()`-scoped `find_package(Qt6 COMPONENTS Quick Qml Test REQUIRED)` + VTK re-find (see `jtml_test_qml_render_smoke`, ~line 1380) — the established precedent for adding a Qt6 component locally without touching the root.
- **AUTOMOC gotchas (documented, repeated in every QtTest target):** Q_OBJECT headers MUST be in the `add_executable` source list (e.g. `include/view/frame_list_model.h` listed alongside `src/view/frame_list_model.cpp`); sibling trap — `signals:` mid-class → duplicate-definition (docs/solutions/build-errors/...). `QSignalSpy` must observe on the main/test thread (QTBUG-2842).
- **Direct-compile pattern:** headless tests compile the relevant `.cpp` files directly (CUDA-free) rather than linking `jtml_compute`/`jtml_view`; the experimental-bridge suites (`jtml_test_experimental_selection/settings/optimizer_gate/ml_bridge/pose_bridge`) direct-compile the whole bridge stack + `test/unit/frame_headless.cpp` (pure-OpenCV Frame twin) and link `jtml_coordinator`. Fixtures via repo-root `WORKING_DIRECTORY`.
- **Naming:** targets `jtml_test_<snake_case>`; ctest names `jtml.<snake_case>`; Catch2 tests in `test/unit/<x>_test.cpp`, QtTest in `test/lifecycle/<x>_test.cpp`, hegel PBT `<x>_properties.cpp` ("deterministic twin + PBT" house rule), oracle in `test/oracle/`. Test dirs: `unit/`, `lifecycle/`, `golden/`, `oracle/` (nfd/vtk dirs are dead).
- **Qt Quick Test wiring implications (no precedent in repo — net-new):** the natural fit is `find_package(Qt6 COMPONENTS QuickTest)` (component verified available) in a `block()` inside `test/CMakeLists.txt`, a `tst_*.qml` dir (e.g. `test/qml/`), and either `qmltestrunner` or a C++ runner linking `Qt6::QuickTest`. Headless-safe **iff** the test scene never instantiates `QmlVtkRenderer`/VTK items; offscreen plugin verified present. qmlprofiler requires the app to enable QML debugging (`-qmljsdebugger`/`QQmlDebuggingEnabler`) — a profiling harness would be a separate instrumented run, not a ctest.

**QML code patterns (observed, evidence for the review pass):**
- **Theme token compliance is PARTIAL (severity: medium):** `Theme.qml` defines 12 tokens (`bg/panel/surface/border/fg/fgMuted/fgDim/selection/ok/accent/badge`), but `SettingsPanel.qml` hardcodes `"#cfd3da"` (header label — not a token) and `"#8b929c"` (== `Theme.fgMuted`); the dirty-badge palette (`#e5b567`/`#3a4a3d`/`#2a2118`/`#8fbf96`) is duplicated verbatim in both `main.qml` (pose dialog) and `SettingsPanel.qml`; `main.qml` readout uses `color: "white"`.
- **No typography scale:** `font.pixelSize: 10` / `11` inline everywhere (~20+ sites); no `font` tokens in Theme.
- **Layout/binding smells:** `Column` inside `ScrollView` with `width: poseScroll.availableWidth` / `width: root.width - 18` (scrollbar-gutter magic); fixed column widths (44/78px) in header + cells; `Qt.callLater` index-sync hack in `onDatasetChanged`; `ButtonGroup` `buttons:` lists; both ListViews use raw `Rectangle`+`MouseArea` delegates (no keyboard nav, no focus handling — accessibility gap for track 2).
- **Structure patterns:** versionless imports; `qsTr` everywhere; secondary surfaces as `Dialog`/`Popup`; bridge→view glue exclusively via `Connections` hops (5 blocks); per-seam `onMessageRequested` funneled to one `showMessage()` dialog; inline `component` definitions only in non-root documents (SettingsPanel) — main.qml is engine-root so refactors must extract files, not inline components.
- **Commit conventions (jj log):** conventional commits with scope + plan refs — `fix(qml): multi-select via Qt's native QFileDialog...`, `feat(coordinator): OptimizerRunController + drive seam (plan 006 U5)`, `refactor(services): ... (plan 006 U7)`, `docs: ...`.

---