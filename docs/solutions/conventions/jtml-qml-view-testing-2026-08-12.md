---
date: 2026-08-12
last_updated: 2026-08-12
last_updated: 2026-08-12
module: jtml_view
tags: [qml, qtquick, testing, qmltest, qmllint, qmlprofiler, conventions]
problem_type: convention
severity: medium
---

# QML view testing, qmllint gate, and profiling recipes (plan 007 U6/U7)

Plan 007 landed the experimental app's first QML-side test harness, a
repeatable qmllint gate, and a profiling build. This entry captures the
recipes so future QML work reuses them instead of re-deriving them.

## The Qt Quick Test harness (`test/qml/`)

- **Layout:** `test/qml/main.cpp` (`quick_test_main`), `tests.qrc` which
  **aliases the REAL `src/app/experimental/*.qml` sources**
  (`<file alias="components/MlStrip.qml">../../src/app/experimental/MlStrip.qml</file>`)
  plus a test-owned `components-qmldir` — tests exercise the shipped
  files, no drift. `ViewportPanel` is deliberately absent (VTK territory).
- **Registration** (`test/CMakeLists.txt`, additive): block-scoped
  `find_package(Qt6 COMPONENTS QuickTest)`, `add_executable(jtml_test_qml_view)`
  with `main.cpp` + `tests.qrc` only — **no bridges, no VTK, no
  `jtml_coordinator`/`jtml_compute`** — `LABELS "headless"`, repo-root
  `WORKING_DIRECTORY`, env `QT_QPA_PLATFORM=offscreen` +
  `QT_QUICK_CONTROLS_STYLE=Material` (the app pins Material Dark; tests
  must run the same style).
- **Run mechanism:** the ctest target is the SOLE runner. Never invoke
  bare `qmltestrunner`/`qmllint` — the conda env's `bin/` names are
  **Qt 5.15.8** (from `qt-main`, pulled by opencv's qt5 build); the Qt 6
  tools live at `$CONDA_PREFIX/lib/qt6/bin/` (verified 2026-08-12). A Qt 5
  runner silently runs tst files on the Qt 5 engine and can pass while
  validating nothing.
- **Fake bridges:** `FakeStudyBridge.qml` etc. are plain QObjects
  exposing the property/signal surface the components read; components
  receive them via **injected `required property var <bridge>`** (plan 007
  D1) — the injection refactor also removed ~300 qmllint unqualified
  warnings. The app wires the real bridges in `main.qml` via root-level
  aliases (`readonly property var studyBridgeRef: studyBridge`) to avoid
  same-name self-referential binding loops.
- **Mouse-path pins are mandatory.** The frame-picker regression (see
  below) survived because an earlier pin skipped the mouse path as
  "ListView-standard". `mouseClick(delegate, x, y)` on
  `list.itemAtIndex(row)` is the pattern that would have caught it.

## The qmllint gate (`jtml.qml_lint`)

- `test/qml_lint.cmake` runs `$CONDA_PREFIX/lib/qt6/bin/qmllint --json`
  over the app's QML and **fails on warnings outside a documented accepted
  set** (each accepted id carries its removal path in the script's
  comment): `import` (Qt5/Qt6 module ambiguity — environmental),
  `unqualified` (rootContext bridges in main.qml — removed by the D1
  injection in components), `unresolved-type`/`missing-property`
  (QmlVtkRenderer is C++-registered without qmltypes), `use-proper-function`
  (deliberate `property var commit` pass-through glue).
- The gate is a headless ctest, so `pixi run test` enforces it.

## The profiling build (`configure-profiling` / `build-profiling`)

- Additive pixi tasks mirroring `configure` into `.build-prof` with
  `-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS=-DQT_QML_DEBUG`
  (qmlprofiler needs it for binding/JS/signal events). Never edit
  `CMakeLists.txt` for profiling.
- Run: `QT_QPA_PLATFORM=xcb $CONDA_PREFIX/lib/qt6/bin/qmlprofiler -o
  profiler/traces/<name>.qtd -- .build-prof/bin/jtml_experimental` —
  the trace saves on app exit. Parse with the qt-qml-profiler skill's
  script; report per the skill's format (two standalone reports, no
  delta framing).
- First verdict (2026-08-12): zero jank (7.04 ms frames); the dominant
  cost is the in-process QFileDialog open (~3.2 s) — the evidence for the
  queued QML path-bar picker.

## QML gotchas confirmed on this stack (Qt 6.7.2)

- **`ListView.view` is null in NESTED delegate items.** The attached
  `view` property reaches the delegate root only. `onClicked:
  ListView.view.currentIndex = index` inside a nested MouseArea throws a
  silent TypeError per click — the frame picker appeared dead. Use the
  list's id (`frameList.currentIndex = index`) — ids are in file scope
  and visible from delegates.
- **Pooling does not drop focus in Qt 6.7.** The commit-on-pool contract
  cannot rely on focus-loss `editingFinished` when a delegate is
  recycled (hidden items keep activeFocus) — the landed mechanism is an
  `edited` flag + explicit `commitIfEditing()` flushed from the
  **attached** `ListView.onPooled` handler (delegate-root `onPooled:`
  never fires — it's a ListView-attached signal). See the D8 correction
  in plan 007.
- **`required property double x/y/z` on a delegate collides with Item's
  FINAL geometry props** (`Cannot override FINAL property`) — the app was
  unloadable until the pose-table delegate switched to
  `model.roleName` access. Name roles anything but geometry names.
- **Trace-driven debugging works:** the qmlprofiler trace of the failing
  session (13 `onClicked` firings, 1 `onCurrentIndexChanged`) pinpointed
  the frame-picker regression without a debugger. Full method + case:
  `docs/solutions/ui-bugs/jtml-qml-trace-driven-debugging-2026-08-12.md`.
- **Required properties leave the implicit `model` context stale on pool
  reuse (2026-08-12 review round):** a required `frameIndex` re-binds
  correctly while `model.x` keeps the previous row's value — declare
  `required property var model` alongside. Test-proven
  (`frameIndex=3, model.x=0`). Full pattern set incl. keys contracts:
  `docs/solutions/conventions/qml-listview-delegate-patterns-2026-08-12.md`.
- **Keys.onPressed on a QQC2 TextField root captures arrows/Escape before
  the internal editor** — the enabler for the D7 cell-navigation contract
  (Left/Right/Up/Down/Esc via a `navRequested` signal; cursor positioning
  becomes click-only). Same pattern doc as above.

## Known drift (historical entries, not edited)

- The 2026-08-11 conventions entry (`jtml-qml-experimental-frontend-...`)
  references a `MultiFilePicker.qml` that no longer exists — superseded
  by `FileDialogBridge` (in-process QFileDialog, DontUseNativeDialog).
  The owner's queued follow-up is a full QML path-bar picker.
- `profiler/` output (traces + reports) is a local artifact directory —
  gitignore it; traces exceed jj's snapshot size limit.

## Related

- Plan: `docs/plans/2026-08-12-007-refactor-qml-experimental-improvement-pass-plan.md`
- Review: `docs/reviews/2026-08-12-qml-experimental-review.md`
- Profile: `profiler/reports/profile-report-jtml_experimental-2026-08-12.md`
- `docs/solutions/conventions/jtml-qml-experimental-frontend-2026-08-11.md`
