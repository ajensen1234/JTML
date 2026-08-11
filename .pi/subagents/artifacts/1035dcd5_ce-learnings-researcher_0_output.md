All candidates read and scored. The knowledge base is small (10 files) and I've examined every potentially relevant entry. Composing the final report.

## Institutional Learnings Search Results

### Search Context
- **Feature/Task**: Structured review + best-practice improvement pass over the experimental QML front-end (`src/app/experimental/`, `jtml_experimental`) of the Qt6+VTK9.3+CUDA desktop app — QML review, UI/UX audit, Qt Quick Test coverage, qmlprofiler profiling.
- **Keywords Used**: qml, qtquick, qquickvtkitem, qt quick test, qmltest, qmllint, qmlprofiler, automoc, cmake, xcb, qvtk, rendering, moc, signals, layered-lib, view-model, ui styling, theme, headless testing
- **Files Scanned**: 10 total (`docs/solutions/` — build-errors/, conventions/, logic-errors/, tooling-decisions/, ui-bugs/)
- **Relevant Matches**: 9 files; 5 detailed below, 4 noted as additional matches

### Critical Patterns
`docs/solutions/patterns/critical-patterns.md` does not exist in this repo — no critical-patterns section.

### Relevant Learnings

#### 1. QML experimental front-end: what landed, what it proved, what it costs (plan 005)
- **File**: `docs/solutions/conventions/jtml-qml-experimental-frontend-2026-08-11.md`
- **Module**: jtml_view
- **Problem Type**: `convention` (severity: medium)
- **Relevance**: The charter document for the exact code under review. Every structural convention the review must respect is here.
- **Key Insight**:
  - **Render-thread contract**: `dispatch_async` lambdas run on the Qt Quick render thread, not the GUI thread — copy state on the GUI thread, capture by value, never read app-owned mutable state inside lambdas. Scene-graph node recreation re-runs `initializeVTK` — rebuild the full pipeline from an app-thread mirror. Any review fix touching the viewport must respect this or it reintroduces races.
  - **Known defect, not a regression**: the pinned VTK 9.3 `QQuickVTKItem` never calls `QVTKInteractorAdapter::SetDevicePixelRatio` — DPR≈2 box ⇒ drag/pick sensitivity halved. A "best-practice fix" candidate, but flag it as a known, pinned limitation; verify before "fixing" (VTK in-tree version).
  - **Conventions to preserve**: versionless imports (`import QtQuick`, never `2.15`); Theme singleton requires `pragma Singleton` + `qmldir` entry (plain file gives `[undefined]` colors); Material Dark for Controls; `Dialog`/`Popup` for secondary surfaces; pure-QML `MultiFilePicker.qml` (checkboxes, zero native/portal variance) is the sanctioned multi-select solution — don't regress to native file dialogs.
  - **Capture path**: `QQuickWindow::grabWindow` is the ONLY capture path; `vtkWindowToImageFilter` is a documented segfault (VTK_USE_X=OFF). Gate first grab on a rendered frame (expose/afterRendering).

#### 2. QML model-pose sync: queued invokeMethod functor silently never delivered
- **File**: `docs/solutions/ui-bugs/jtml-qml-model-pose-sync-queued-functor-never-delivered-2026-08-11.md`
- **Module**: jtml_view (component: frontend_stimulus)
- **Problem Type**: `ui_bug` (severity: **high**)
- **Relevance**: The only high-severity QML-specific bug on record — exactly the failure class a UI/UX audit of this app must guard against.
- **Key Insight**: `QMetaObject::invokeMethod(obj, functor, Qt::QueuedConnection)` posted (returned `true`) but the functor **never executed** in this harness — silent delivery failure. The working pattern: plain thread-safe reporter method that `emit`s a signal with by-value data; AutoConnection resolves Direct/Queued per receiver. Also: when a queued delivery is suspect, instrument the **receiver** side with a counter, not the sender; stale-binary trap — after any compile fix, rebuild-and-rerun (a failed build makes the previous binary suspect).

#### 3. Shared VM layer (plan 006): the conventions that make the two front-ends one
- **File**: `docs/solutions/conventions/jtml-shared-vm-layer-2026-08-11.md`
- **Module**: jtml_coordinator
- **Problem Type**: `convention` (severity: medium)
- **Relevance**: Directly constrains the UI/UX audit: the QML bridges (`StudyBridge`/`OptimizerBridge`/etc.) are deliberately thin shells over shared controllers — behavior lives in the seams.
- **Key Insight**: **"Pin, don't unify"** — save-last-pose is 4 deliberately divergent behaviors (widgets canonical, camera-A inline, camera-B inline, QML mirror), each pinned by a call-site table test; the review must not "fix" QML behavior to match widgets (e.g., QML keeps "stays Error" runState while widgets unlock at the terminal relay; QML ignores `messageRequested` severity). **QML-side policies stay QML-side** (SingleModelOnly pre-check, Dialog mapping in `OptimizerBridge`). R13: QML thinnings are "verbatim-behavior extraction with signature adaptation" gated by characterization test + behavior diff, not literal diff. Follow-on note: optimizer-backend work is deliberately out of scope for any QML pass.

#### 4. Headless Testing and Testability-Refactor Conventions (JTML)
- **File**: `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md`
- **Module**: JTML (component: testing_framework)
- **Problem Type**: `convention` (severity: medium)
- **Relevance**: The template for the planned Qt Quick Test coverage — hybrid framework split, CTest labels, and the two moc/AUTOMOC traps that will bite any new QML-bridge test target.
- **Key Insight**: Hybrid Catch2 (pure logic) + QtTest (QObject/QThread/QSignalSpy seams); headless is the default (`ctest -L headless`, `QT_QPA_PLATFORM=offscreen`), GPU/render cases behind explicit `oracle`/`render` labels with run-level timeouts (`--timeout 600`, per-test `TIMEOUT`). **AUTOMOC gotcha**: a Q_OBJECT header only `#include`d (not listed in the target's sources) never gets moc'd → undefined-symbol link error — new QML bridge headers in a test target must be listed explicitly. **GLOB trap**: headers are globbed but `.cpp` sources are explicit lists; a new `.cpp` reached via the header GLOB must be added to the explicit list. QtTest rules: spy on the main thread only (QTBUG-2842), cooperative stop, timeout-bounded `QSignalSpy::wait`. No prior learning exists specifically on Qt Quick Test (`qmltest`) or `qmllint` — that coverage is new territory; capture what you learn with `/ce-compound` after.

#### 5. JTML rendering runtime: xcb default, QVTK smoke, VTK standalone-window limitation
- **File**: `docs/solutions/tooling-decisions/jtml-rendering-runtime-xcb-qvtk-2026-08-10.md`
- **Module**: render (component: tooling)
- **Problem Type**: `tooling_decision` (severity: **high**)
- **Relevance**: Governs how the app (and any profiling/test session) must be run, and why QQuickWindow-grab tests are the only viable render-verification path.
- **Key Insight**: Run under `QT_QPA_PLATFORM=xcb` (Wayland/EGL incomplete on this box → blank GL view); historical working platform is xcb. **Never use a standalone `vtkRenderWindow`**: this VTK build has `VTK_USE_X=OFF`, `VTK_OPENGL_HAS_EGL=OFF`, `VTK_OPENGL_HAS_OSMESA=OFF` — `vtkRenderWindow::New()` returns the base class, `vtkTextRenderer::GetInstance()` is null, `vtkWindowToImageFilter` segfaults; only the QVTK path works. Render verification = QVTK-mirror smoke capturing via `widget->grab()`/`QQuickWindow::grabWindow` (mirrors learning #1). For qmlprofiler: profile under xcb with the same env as production, and be aware DISPLAY is inherited, never baked into CMake.

**Additional matches (brief)**:
- `docs/solutions/build-errors/jtml-moc-signals-section-placement-duplicate-definition-2026-08-11.md` (`build_error`, medium) — `signals:` mid-class turns following accessors into signals → `multiple definition of StudyBridge::hasCalibration()` link errors. Any review fix adding signals/Q_INVOKABLEs to bridge classes must keep `signals:` last or follow with an access specifier. Distinct from the AUTOMOC undefined-symbol trap (#4).
- `docs/solutions/conventions/jtml-layered-lib-split-2026-08-08.md` (`convention`, medium) — per-layer CMakeLists: `file(GLOB ... CONFIGURE_DEPENDS)` for headers + explicit `.cpp` list; include prefixes standardized (`view/`, `domain/`) and grep-gated; new `Q_OBJECT` headers must be globbed into their owning target; `jtml_view` AUTOUIC autogen is PUBLIC and bakes the target name. Applies to any new CMake wiring for QML test targets.
- `docs/solutions/conventions/jtml-mainscreen-decomposition-patterns-2026-08-10.md` (`convention`, medium) — the widgets-side "silent-wiring-death" class (by-name auto-connect dies with zero compile error; `setModel()` must precede connects) — the same failure class as the QML pose-sync bug; useful context for the UI/UX audit's wiring review, and stylesheet-selector drift after widget swaps.
- `docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md` (`tooling`, medium) — Qt6 environment recipes: pin `qt6-main`+`qt6-wayland` 6.7.2 (not `qt`), `QSurfaceFormat::setDefaultFormat(...)` must run BEFORE `QApplication`/engine creation (latent, louder on Qt6 — relevant to any QML harness main()), hegel PBT recipes (FetchContent, `-Wl,-rpath` append for conda toolchain) if property tests are added alongside Qt Quick Tests.

**Not relevant**: `docs/solutions/logic-errors/cost-function-parameter-double-truncation-2026-08-08.md` — cost-function numeric bug, no QML/UI/CMake applicability.

### Recommendations
- **Read learning #1 in full before touching any viewport code** — the render-thread contract is the load-bearing discipline; review fixes that add state access inside `dispatch_async` lambdas will reintroduce the exact race the architecture avoids.
- **Frame the UI/UX audit around "pin, don't unify" (#3)**: QML-vs-widgets behavioral deltas are deliberate and test-pinned — the audit should surface deltas but gate any unification as a separately-scoped cut with its own characterization tests.
- **Design Qt Quick Test coverage on the #4 template**: headless default label, explicit `render`/`oracle` labels, timeout-bounded tests, receiver-side instrumentation counters (per #2's smoke leg 3.5 pattern); list every Q_OBJECT header explicitly in new test targets and add any new `.cpp` to its layer's explicit source list (#4, layered-lib doc).
- **Run qmlprofiler and any render tests under `QT_QPA_PLATFORM=xcb` (#5)**, and use `QQuickWindow::grabWindow`-based capture only; a blank result means check xcb → nvidia-smi → smoke, in that order.
- **Add qmllint/Qt Quick Test/qmlprofiler learnings to `docs/solutions/` after the pass lands** — nothing exists on those tools yet; the absence is itself the signal that this territory is uncaptured.