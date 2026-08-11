---
date: 2026-08-11
topic: qml-experimental-frontend
---

# JTML QML Experimental Front-End (Parallel App)

## Problem Frame

The research tool's UI (`src/view/mainscreen.cpp`, ~5,115 lines) is a hand-built
QWidgets mainscreen developed by a long-gone student; changing its layout or redoing
it in Qt Designer is a chore the current owner won't invest in. Meanwhile the
backend — `jtml_domain/services/coordinator/compute` — is now fully widget-free
(plans 001–004), the two list view-models are `QAbstractListModel` (QML-native), and
the repo's pinned VTK 9.3 build already compiles `libvtkGUISupportQtQuick` in-tree
(panoptes angle 04: zero VTK rebuild needed). The owner wants to explore QML — not as
a rewrite of the working app, but as a **resident experimental front-end**: a new,
parallel executable with its own QML main that links the same compute libraries and
provides a cleaner surface for running optimizer/cost experiments, coexisting with
the widgets app indefinitely.

The exploration is grounded in the panoptes run `jtml-research-horizons`, angle 04
(`.panoptes/jtml-research-horizons/angles/04-qml-vs-widgets.org`): a full QML
rewrite of the mainscreen is a NO-GO for now, but a parallel QML app exercising the
already-compiled `QQuickVTKItem` integration is the sanctioned first increment, and
QML islands on the mainscreen (QQuickWidget) remain a future option, not this phase.

## Actors

- A1. **Developer/researcher (owner):** runs experiments; wants a UI they actually
  enjoy using for optimizer/cost work; the success bar is *their preference*.
- A2. **Coding agent (pi):** builds the parallel app under the repo's gate
  discipline; must not touch the widgets app.

## Key Flows

- F1. **Study-load flow** — Trigger: user opens the QML app and picks a study's
  calibration/image/model files. Actors: A1, A2. Steps: (1) QML file dialogs gather
  paths; (2) `SessionController` parses them (existing headless seam, U6) and the
  app populates its dataset containers (frames/models/`LocationStorage`); (3) the
  QML lists (over `FrameListModel`/`ModelListModel`) update; (4) the viewport
  renders the first frame + models. Outcome: the dataset is loaded and inspectable
  with zero changes to the backend. **Covered by:** R3, R5.

- F2. **Experiment-run flow** — Trigger: user sets settings (ranges/budgets/dilation,
  cost variant) and hits run. Actors: A1, A2. Steps: (1) settings panel writes
  `OptimizerSettings` + selects the cost variant via `CostFunctionManager`; (2) the
  app drives the real `OptimizerManager` (widget-free QObject) directly; (3)
  progress signals (calls, current min, pose updates, stage) flow into QML
  bindings; (4) on completion the optimized pose is written to `LocationStorage`
  and the viewport re-renders the implant at the result pose over the fluoro
  frame. Outcome: a complete experiment run inside the QML app, oracle-arbitrated
  (R12). **Covered by:** R4, R6, R7, R12.

- F3. **ML-initialized run flow** — Trigger: user supplies `.pt` models and runs
  segment/estimate before optimizing. Actors: A1, A2. Steps: (1) `SegmentationController`
  segments the frame (existing per-frame API); (2) `ImplantEstimator` produces an
  initial pose; (3) the pose seeds the optimizer; (4) refinement proceeds as F2.
  Outcome: the full pipeline the widgets app runs, minus biplane, in the QML app.
  **Covered by:** R8, R12.

## Requirements

**[Parallel architecture]**
- R1. A new executable target (QML app) lives in `src/app/` alongside the widgets
  app, linking `jtml_domain` + `jtml_services` + `jtml_coordinator` + `jtml_compute`
  + `Qt6::Quick/Qml/QuickControls2` + `VTK::GUISupportQtQuick` (with
  `vtk_module_autoinit`). `jtml_view` (MainScreen + Viewer + dialogs) is NOT linked.
- R2. The widgets app and all backend code stay byte-identical across this phase
  (R13): the QML app only *links* existing seams (`OptimizerManager`,
  `CostFunctionManager`, `SessionController`, `SettingsService`, `pose_file_io`,
  `pose_copy`, `SegmentationController`, `ImplantEstimator`, `FrameListModel`,
  `ModelListModel`, `SessionState`) — zero modifications to them. `jj diff` against
  the pre-phase baseline proves additivity.
- R3. Study loading reuses `SessionController`'s parsing; the QML app owns its
  dataset containers (frames/models/`LocationStorage`) exactly as the widgets view
  does today.

**[UI surfaces]**
- R4. An optimizer settings panel edits trunk/branch/leaf ranges, budgets, and
  dilation (backed by `OptimizerSettings` + `settings_constants.h` defaults).
- R5. Cost-variant selection exposes the registered cost functions
  (`CostFunctionManager::listCostFunctions`) — the experiment vehicle's core knob.
- R6. Live optimization progress: stage, cost-call count, current minimum, pose
  updates, driven by the `OptimizerManager` signals the widgets app currently
  binds in `LaunchOptimizer` (same signal surface, new QML wiring).
- R7. A single main viewport: a `QQuickVTKItem` rendering the implant STL silhouette
  at the current pose over the fluoro background frame; pose updates flow through
  the render-thread discipline (see R10). No coronal/biplane viewport in v1.
- R8. ML path: `.pt` model selection (segment + estimate), `SegmentFrame` /
  `EstimateImplantPose` reused; the estimate seeds the optimizer (initial pose).
- R9. Pose save/load/edit: `pose_file_io` save/load of pose and kinematics files,
  and copy-previous/copy-next via the `pose_copy` domain seam, exposed in the QML
  UI (per-frame pose values editable/displayed).
- R10. Settings persistence through the existing `SettingsService` (registry parity
  contract — same org/app/groups/keys), so the experiment setup survives restarts.

**[Rendering seam]**
- R11. A `QmlVtkRenderer`-style class owns the minimal VTK pipeline (renderer,
  actors, mappers, background image) inside `QQuickVTKItem::initializeVTK`; ALL VTK
  state is QML-render-thread-owned and reachable only via
  `initializeVTK`/`destroyingVTK`/`dispatch_async` (the QQuickVTKItem contract —
  the widgets `Viewer` plumbing does not transfer and is not reused). Model loading
  (STL via existing services), silhouette rendering, and background display follow
  the compute layer's existing render paths.

**[Gates]**
- R12. Oracle integrity: the oracle tests link no UI today and nothing in this
  phase changes that — `test/oracle/` binaries stay UI-free, and the existing gates
  (`pixi run test` 32/32, `ctest -L oracle`, `ctest -L render` under xcb) remain the
  phase's regression gate, green at every cut.
- R13. Oracle-arbitrated parity is the end-to-end gate: the QML app run on Kneel_1
  with the oracle's configuration (DIRECT_DILATION, budget 3000, range
  (12,12,15,15,15,15), Canny 3/0/150, dilation 6) must land within the oracle's
  band (silhouette IoU ≥ 0.85; pose within the informational tolerance of
  `fem.jts` per `test/golden/baseline.json`). This verifies the app's *wiring*,
  not the backend (which the oracle already gates directly).
- R14. Per-cut gates: pure/new-logic extractions get headless tests; UI cuts get
  compile + scheduled manual-visual check under xcb; the QML render path gets a
  QML render smoke (see R15). The widgets app + oracle suite green at every cut.
- R15. A QML render smoke replaces the `widget->grab()` pattern for this app:
  capture via `QQuickWindow::grabWindow` or `vtkWindowToImageFilter` under forced
  `QT_QPA_PLATFORM=xcb`, registered as its own test (analog of `jtml.render_smoke`).
- R16. `.pt` torch models remain user-provided (no training, no committed weights
  in this phase); the ML path degrades gracefully (clear message) when absent.

**[Scope of v1]**
- R17. v1 covers exactly: study loading, optimizer settings panel, cost-variant
  selection, progress display, ML segmentation/estimate, pose save/load/edit,
  settings persistence, single main viewport. Explicitly deferred: biplane +
  coronal viewport, an IoU results table, QML islands on the widgets mainscreen,
  any MainScreen rewrite.

## Acceptance Examples

- AE1. **Covers R1, R2, R13.** Given the phase is complete, `jj diff` against the
  pre-phase baseline shows only added files (the new app target + its sources +
  tests); the widgets app, backend, and oracle suite are byte-identical and green.
- AE2. **Covers R4, R5, R6, R13.** Given Kneel_1 loaded in the QML app with the
  oracle configuration, when the user runs the optimizer, progress updates appear
  (stage/calls/min) and the final viewport shows the implant at the optimized pose;
  the pose lands within the oracle's band (IoU ≥ 0.85 vs `Labels/`).
- AE3. **Covers R5.** Given two cost variants, when the user runs the same study
  with each, the runs execute with the selected variant (visible via settings state
  and differing progress), demonstrating the experiment knob works.
- AE4. **Covers R8.** Given user-provided `.pt` models, when the user runs
  segment→estimate→optimize, the segmented frame displays and the estimated pose
  seeds a successful refinement; without models, a clear message appears and the
  plain-optimize path still works.
- AE5. **Covers R9, R10.** Given a completed run, the user saves the pose file,
  restarts the app, and the settings + pose round-trip via `SettingsService` and
  `pose_file_io` (registry keys and file format unchanged).

## Success Criteria

- **Human outcome:** the owner *prefers* the QML app for optimizer/cost experiments
  (the widgets app remains the full-featured fallback); a full experiment
  (load → configure → run → inspect → save) is pleasant end-to-end.
- **Handoff quality:** a plan can enumerate units with per-cut gates, reusing the
  seam inventory above; no unit requires inventing backend behavior (everything
  called already exists and is tested); the oracle-integrity argument (R12) is
  structurally true, not aspirational.

## Scope Boundaries

- NOT a MainScreen rewrite, NOT QML islands on the widgets app (deferred; the
  islands path is a separate future decision).
- No changes to `OptimizerManager`, `SessionController`, `CostFunctionManager`,
  compute kernels, or any existing seam (R2 — reuse only).
- No biplane/coronal viewport and no IoU results table in v1.
- No ML model training or committed weights; `.pt` models stay user-provided.
- No new backend test surface beyond what the phase's own new code requires — the
  oracle, its fixtures, and tolerances are untouched.
- Not the production-shaped multi-stage oracle (the oracle's single-stage config is
  used as the parity baseline; closing the production-vs-oracle gap is the
  optimizer-backend workstream).

## Key Decisions

- **Parallel resident app, not a rewrite:** a second executable in `src/app/`
  linking the same libs; the widgets app is untouched forever unless a later
  decision supersedes it. This matches the owner's "wholly parallel" instinct and
  the angle-04 verdict (full QML rewrite = NO-GO now; spike = sanctioned first
  increment; the resident app IS the spike plus the experiment surfaces).
- **Real optimizer, real seams:** the QML app drives `OptimizerManager` (widget-free
  QObject) directly — no stubs, no coordinator seam, no reimplementation. This is
  what makes oracle-arbitrated parity meaningful.
- **Models reused, selection simplified:** `FrameListModel`/`ModelListModel` are
  reused as-is; selection is QML-delegate-based (current frame via `currentIndex`,
  model multi-select via model-level state) — a NEW, simpler contract that does not
  reuse `QItemSelectionModel` and therefore does not touch the widgets app's pinned
  selection semantics (R13).
- **Render pipeline reimplemented under the render-thread contract:** the minimal
  VTK pipeline (R11) replaces `Viewer`'s widget-specific plumbing; this is the
  phase's main technical risk and the reason the render smoke (R15) exists.
- **The oracle arbitrates the wiring, not the backend:** parity (R13) uses the
  oracle's own config and band because it is the R2-validated independent ground
  truth; the backend itself is already gated directly by the oracle tests.

## Dependencies / Assumptions

- VTK 9.3 `GUISupportQtQuick` is already compiled in-tree (`_deps/vtk/install/lib64/libvtkGUISupportQtQuick-9.3.so`) — no VTK rebuild (verified, angle 04).
- The pixi env ships QtQuick/QtQuickControls2/QtQml headers (verified, angle 04).
- `QQuickVTKItem` renders and picks correctly under this box's xcb + Qt 6.7.2 + VTK
  9.3 — the phase's first unit is the spike that verifies this; known risks: the
  DPR/mouse-position bug (reported Qt 6.5.3, user-patched), draw-pick-style
  interactor failures. The spike result is the go/no-go for the rest.
- The user has `.pt` models available for the ML path (or the graceful-degradation
  path is exercised instead).
- Kneel_1 fixtures (`example_studies/Kneel_1/` + `test/golden/`) are the parity
  baseline.
- jj is the VCS; pixi is the build (`pixi run build`); xcb is the only working
  render platform.

## Outstanding Questions

### Resolve Before Planning

None.

### Deferred to Planning

- [Affects R4][Technical] Exact app/target naming (e.g. `jtml_experimental`) and
  where in `src/app/` it lives; CMake wiring details (new target vs subdirectory).
- [Affects R4][Technical] Whether the settings panel writes the registry through
  `SettingsService` on every change (parity with the widgets app's behavior) or
  applies session-local settings with an explicit save.
- [Affects R9][Technical] How per-frame pose display/editing maps to QML (a table
  with editable cells vs a pose readout) — the values themselves come from the same
  seams either way.
- [Affects R5][Technical] How cost-variant selection interacts with the
  `CostFunctionManager` registry guard blocks (the widgets app selects via
  `SettingsControl`; the QML app calls the same API — verify there is no
  UI-coupled state).
- [Affects R8][Needs research] The QML FileDialog vs a thin QWidgets-bridge for the
  study/model file dialogs (QML's FileDialog is fine for the spike; parity of
  directory persistence is an open detail).
- [Affects R15][Needs research] `QQuickWindow::grabWindow` vs
  `vtkWindowToImageFilter` for the QML render smoke — spike-time discovery.

## Next Steps

-> /ce-plan for structured implementation planning
