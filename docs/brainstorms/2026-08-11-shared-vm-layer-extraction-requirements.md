---
date: 2026-08-11
topic: shared-vm-layer-extraction
---

# Shared VM-Layer Extraction (both front-ends, one controller layer)

## Problem Frame

The repo now has two front-ends: the widgets app (`src/view/mainscreen.cpp`,
~5,115 lines, fuses View + ViewModel in its slots) and the QML experimental
app (`src/app/experimental/`, whose bridges — StudyBridge/SettingsBridge/
OptimizerBridge/MlBridge/PoseBridge behind AppBridge — ARE a working
view-model layer). Plan 005 built the bridges as thin pass-throughs by rule,
but it was R13-forced to *replicate* three pieces of widgets logic instead of
sharing them: the `LaunchOptimizer` drive sequence, the
`BuildCostFunctionRegistryEntries` mapping, and `matToVTK`/the render
pipeline. Meanwhile MainScreen still holds a large VM-shaped surface the QML
app mirrors (session sync, save-last-pose bookkeeping, load orchestration,
camera A/B, segment/estimate slots).

The owner's goal: extract the VM-shaped surface into a **shared controller
layer both apps call**, so MainScreen shrinks to minimal glue ("only the
minimal set of elements needed as basic glue in place") and future
experiment surfaces land once instead of twice. The optimizer-backend
modularity is explicitly a later session. This phase executes R8 of
`docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md` (the
full-MVVM commitment) and the deferred follow-up work of plan 005.

**Design constraint (applies to the whole layer, not a deliverable):** the
shared layer must also be drivable by a *headless* consumer — a future
config-file-driven batch driver that never displays anything, but still needs
to set the session (selection, poses, settings) and drive the run. Controllers
must therefore be view-agnostic: no view/scene/renderer pointers, and the full
drive + observe surface must work under `QCoreApplication` alone. Nothing
headless is built this phase (Scope Boundaries); the shape must simply not
preclude it.

The agreed shape, at a glance:

```
                     ┌──────────────────────────────┐   ┌─────────────────────────────┐
                     │  Widgets app (MainScreen)    │   │  QML app (experimental)     │
                     │  view-only residue           │   │  QML views                  │
                     │  + thin slot glue            │   │  + thin bridge shells       │
                     └──────────────┬───────────────┘   └──────────────┬──────────────┘
                        QItemSelectionModel         delegate selection
                        writes derived facts        writes derived facts
                     ┌──────────────▼───────────────┬──────────────────▼──────────────┐
                     │  SHARED VM LAYER                                               │
                     │  coordinator (QObject shells): OptimizerRunController,         │
                     │    SessionStateController, load/ML orchestration               │
                     │  services (plain): SettingsService(+registry mapping),         │
                     │    RenderPipelineBuilder                                       │
                     │  domain (pure, Qt-free): SessionState (the one core),          │
                     │    OptimizeIntentController                                    │
                     └──────────────┬──────────────────────────────────────────────────┘
                        controllers read only SessionState + plain values
                     ┌──────────────▼──────────────────────────────────────────────────┐
                     │  Existing seams (untouched): OptimizerManager, SessionController│
                     │  CostFunctionManager, SegmentationController, ImplantEstimator  │
                     └─────────────────────────────────────────────────────────────────┘
```

*(Future: a headless driver — config file in, no display — drives the same
controllers and observes the same signals; nothing in the shared layer assumes
a view exists.)*

---

## Actors

- A1. **Owner/developer:** runs optimizer/cost experiments; wants MainScreen as
  minimal glue and the QML app as the preferred experiment surface; the
  success bar is their workflow.
- A2. **Coding agent (pi):** executes the extraction under the repo's gate
  discipline (R13 relocation, per-cut gates); must keep both apps + the
  oracle suite green at every cut.

---

## Key Flows

- F1. **Optimizer-run flow (both apps)** — Trigger: the user hits Run with a
  frame + model selected. Actors: A1, A2. Steps: (1) `OptimizerRunController`
  evaluates the intent gate on SessionState + settings; (2) SaveLastPose
  mirror runs through the session core; (3) a fresh `OptimizerManager` +
  thread is created, initialized with the app's containers by value and the
  selection as plain rows; (4) the 7 binds relay progress (stage/calls/min/
  pose) to the view; (5) on completion the optimized pose lands in
  `LocationStorage` and the viewport re-renders. Outcome: one shared drive
  sequence; MainScreen's `LaunchOptimizer` and `OptimizerBridge` are thin
  callers. **Covered by:** R7, R13, R14.
- F2. **Study-load flow (both apps)** — Trigger: the user picks a study's
  calibration/image/model files. Actors: A1, A2. Steps: (1) the shared load
  orchestration parses via `SessionController` into caller-owned containers;
  (2) dataset-replace/calibration-one-use semantics apply (the widgets
  behavior is the spec); (3) counts + labels update through the session core;
  (4) the viewport renders the first frame + models. Outcome: one load path;
  the widgets load slots and `StudyBridge` thin onto it. **Covered by:** R11,
  R13.
- F3. **ML-initiated run flow (both apps)** — Trigger: the user runs
  segment → estimate → optimize. Actors: A1, A2. Steps: (1) the shared ML
  orchestration segments the current frame via `SegmentationController`;
  (2) `ImplantEstimator` produces a starting pose; (3) the pose seeds the run
  via the run controller; (4) refinement proceeds as F1. Outcome: the
  segment/estimate slots and `MlBridge` share one orchestration surface.
  **Covered by:** R12, R13.

---

## Requirements

**[Shared VM layer: home, shape, naming]**
- R1. The shared VM layer extends the existing layers — **no new CMake
  target, no new include prefix**. QObject controllers live in
  `jtml_coordinator` (alongside `OptimizeCoordinator`/`OptimizerManager`),
  plain headless services in `jtml_services`, pure state in `jtml_domain`.
  The dependency graph (view/app → coordinator → services → domain; compute
  standalone) is unchanged; both apps consume the shared layer through their
  existing links.
- R2. Controllers follow the proven split: a plain, Qt/GPU-free state/decision
  core (headless-testable — the pattern plan 005 proved with
  `OptimizerBridge::EvaluateGate`/`applyOptimizedFrame`) wrapped in a thin
  QObject shell owning signals/threads. Signals are the only QObject surface;
  all behavior lives in cores or in the existing seams.
- R3. Naming: shared orchestration classes use `*Controller` in coordinator;
  services keep `*Service`/plain `*Controller`. Collisions with existing
  classes are avoided — the session-state shell is named e.g.
  `SessionStateController`, never `SessionController` (taken by the services
  parse seam).

**[SessionState: the one shared session core]**
- R4. `jta::SessionState` (domain, pure, Qt-free) becomes the single session
  core both views and all controllers read: frame/model counts, current
  frame, selected model rows, primary model (first selected row) — **extended
  with previous-frame and previous-model-row mirrors** (the SaveLastPose
  bookkeeping) and derived helpers (single-selection checks, etc.). All
  headless-testable.
- R5. Both views become write-back adapters over SessionState. The widgets
  app keeps `QItemSelectionModel` as its selection widget (pinned semantics:
  ctrl/shift multi-select, primary = first row) but its handlers write
  derived facts into SessionState; the QML app keeps delegate-based selection
  writing into SessionState. After the sweep, **no view owns private session
  bookkeeping** (MainScreen's `previous_frame_index_`/`previous_model_indices_`
  are deleted).
- R6. A QObject notification shell in coordinator exposes SessionState changes
  (sessionChanged etc.) to both views; the pure core stays signal-free.

**[Extraction surface: MainScreen → shared controllers]**
- R7. `OptimizerRunController` (coordinator, QObject): the complete
  `LaunchOptimizer` drive sequence (intent gate → SaveLastPose mirror → fresh
  `OptimizerManager` + thread + moveToThread → `Initialize` with containers by
  value and selection as plain rows → the 7 binds → start), the run-state
  machine (idle → running → stopping → completed/error), the progress surface
  (stage/calls/min), stop, the Initialize-failure quirk (replicated with a
  comment), and run-locking state. MainScreen's `LaunchOptimizer` +
  `DisableAll`/`EnableAll` and `OptimizerBridge` thin onto it. The deferred
  multi-stage oracle is a future consumer of the same seam.
- R8. `BuildCostFunctionRegistryEntries` moves into `SettingsService` (the
  plan-005 committed follow-up): one registry mapping for both apps; the
  golden-fixture-pinned contract (org/app/groups/keys) unchanged.
- R9. A widget-free render-pipeline builder (services): `matToVTK` + the
  minimal pipeline (renderer, actors, mappers, background image) shared by
  the widgets `Viewer` and `QmlVtkRenderer`. The QQuickVTKItem render-thread
  contract (initializeVTK/destroyingVTK/dispatch_async) remains QML-side; the
  builder is called from each side's safe points.
- R10. Session-state orchestration unifies onto SessionState + a shared
  controller: `SyncSessionState` (counts + labels), the previous-frame
  bookkeeping, and the four save-last-pose copies — as one cut with
  byte-identical relocation first (the definition at mainscreen.cpp:4101 and
  its call sites ~1094/1142/2972, verified at plan time), then the copies
  converge on the shared helpers.
- R11. Study-load orchestration becomes a shared load controller/service over
  `SessionController` parsing + caller-owned containers (frames/models/
  `LocationStorage`/calibration): the widgets load slots (calibration :2207,
  image :2358, model :2502) and `StudyBridge` both thin onto it;
  per-seam behaviors (calibration one-use-per-session, dataset-replace
  semantics) are preserved as the spec.
- R12. Camera A/B slot orchestration (`on_camera_A/B_radio_button_clicked`,
  :2612/:2774) and the segment/estimate slot orchestration
  (`segmentHelperFunction` :1675 + the two `EstimateImplantPose` paths) thin
  onto shared seams — camera state through the session-state core, ML through
  the existing `SegmentationController`/`ImplantEstimator` seams — matching
  the `MlBridge` surface.
- R13. Every cut is **byte-identical relocation first**: `jj diff` shows
  relocation, not rewriting. The widgets' pinned behaviors (selection
  semantics, registry contract, drive sequence + failure quirks, pose-file
  format, camera A/B behavior) are the spec; the QML app keeps working at
  every cut (bridges thin, never re-invent).

**[MainScreen end-state]**
- R14. After the sweep, MainScreen's residue is view-only:
  `ArrangeMainScreenLayout`, VTK actor placement, display-mode radios, key
  handling, dialogs, and thin slot glue calling the shared controllers. The
  parent R10 line-count signal is tracked (5,115 today → ~1,800–2,200 target;
  a signal, not a hard gate), as is the `ui.`-reference count.

**[Headless-consumer readiness]**
- R15. The shared layer is view-agnostic: no shared controller holds a view,
  scene, or renderer pointer, and none assumes a display exists. The full
  controller surface (SessionState writes, run/stop, settings, load, ML) is
  drivable and observable under `QCoreApplication` alone — the
  `OptimizeCoordinator` lifecycle tests are the precedent. A grep-gate
  (no `QWidget`/`QQuick`/`vtkRenderWindow` includes in coordinator controller
  sources) enforces it.
- R16. The pose-sync chain generalizes so no scene lives in the shared layer:
  the run controller writes the optimized/arranged pose into the dataset
  (`LocationStorage` + SessionState) and emits `poseUpdated`; each view maps
  that signal onto its own scene/readout (widgets `Viewer`, QML viewport —
  the existing QML chain is the proof). Shared code never touches a scene
  object.

---

## Acceptance Examples

- AE1. **Covers R7, R13.** Given the pre-cut widgets app, when the
  run-controller cut lands, `jj diff` shows `LaunchOptimizer`'s body
  relocated into `OptimizerRunController`; the widgets run flow (gate order,
  SaveLastPose mirror, binds, Initialize-failure quirk) behaves identically;
  oracle parity unchanged; the QML app is green via a thinned
  `OptimizerBridge`.
- AE2. **Covers R4, R5, R10.** Given the session-state cut, MainScreen's
  `previous_frame_index_`/`previous_model_indices_` are deleted; selection
  behaviors (primary = first row, multi-select, save-last-pose on
  frame/model change) are byte-identical per the scheduled manual-visual
  check; QML app behavior unchanged.
- AE3. **Covers R8.** The registry mapping exists once in `SettingsService`;
  both apps register identical entries; the golden fixture still passes.
- AE4. **Covers R9.** `Viewer` and `QmlVtkRenderer` both build their pipeline
  through the shared builder; `jtml.render_smoke` + the QML render smoke are
  green.
- AE5. **Covers R14.** End-state: MainScreen's residue matches the view-only
  inventory; line count reported; both apps + oracle suite green; QML parity
  (silhouette IoU ≥ 0.85) re-verified.

---

## Success Criteria

- **Human outcome:** the owner's "good groove" — a new experiment surface
  lands once (a shared controller + a tiny per-view adapter), MainScreen is
  minimal glue, and the QML app remains the preferred experiment surface.
- **Handoff quality:** a plan can enumerate units with per-cut gates straight
  from the surface inventory above; no unit invents backend behavior (every
  seam already exists and is tested); each cut's R13 argument is structural
  (`jj diff`), not aspirational.
- **Structural property:** after the sweep, the shared layer is headless-
  drivable — a config-file batch driver would need views only for *output*,
  never for *control* (R15's grep-gate is the checkable proxy).

---

## Scope Boundaries

- NOT the optimizer-backend modularity (parameters + cost + swappable
  backend) — a later session per the handoff.
- NOT the multi-stage oracle (panoptes synthesis item 3) — the run controller
  is shaped as its future seam only.
- NOT QML islands on the widgets mainscreen, and no revisit of the full-QML
  front-end decision.
- NOT touching `OptimizerManager`/`SessionController`/`CostFunctionManager`/
  compute internals; oracle fixtures and tolerances untouched.
- NOT changing pinned behaviors: selection semantics, registry contract,
  drive-sequence quirks, pose-file format, camera A/B behavior.
- NOT unifying the pose-file formats or the settings registry contract.
- NOT deleting the QML bridges — they remain the QML-facing adapter shells
  (the Q_PROPERTY surface), thinning onto the shared controllers.
- NOT building a headless driver/CLI (config-file batch mode) this phase —
  only the view-agnostic controller shape that makes it possible later.

---

## Key Decisions

- **Full VM sweep** (user decision): everything VM-shaped leaves MainScreen —
  residue items, run/session state, load path, camera A/B, segment/estimate
  orchestration — in one arc, not a piecemeal series.
- **Extend coordinator + services; no new layer** (user decision): zero new
  CMake targets or include prefixes; the dependency graph is untouched.
- **SessionState is the one shared session core** (user decision): extended
  with previous-selection mirrors; both views become write-back adapters over
  it; controllers read only it — maximal reuse, minimum per-framework code.
- **Pure cores + thin QObject shells**: the plan-005 testability split
  (public Qt/GPU-free state/decision cores) is the standing pattern for every
  shared controller.
- **Bridges thin onto controllers, never re-invent**: the QML bridge shells
  stay as the QML-facing adapters; their logic moves into the shared layer.
- **View-agnostic controllers (headless-readiness)**: the shared layer is
  drivable under `QCoreApplication` alone — SessionState as the pure core,
  signals for output, no scene/renderer pointers. A future config-file batch
  driver is a consumer of the same surface; nothing headless is built now.
- **Relocation-first ordering principle**: smallest independent cuts first
  (registry mapping → run controller → pipeline builder → session core →
  load/ML/camera), each green; exact unit decomposition is plan-time.

---

## Dependencies / Assumptions

- All seams the controllers will call exist and are tested (verified):
  `OptimizerManager`, `SessionController`, `CostFunctionManager`,
  `SettingsService`, `SegmentationController`, `ImplantEstimator`,
  `pose_copy`, `pose_file_io`, `SessionState`, `OptimizeIntentController`.
- The bridges are the behavioral spec for the shared surface (they encode the
  run-state machine, settings dirty tracking, the pose-sync chain).
- Handoff inventory verified against `src/view/mainscreen.cpp`:
  `LaunchOptimizer` :4135, `DisableAll` :4051 / `EnableAll` :4072,
  `SaveLastPose` :4101, `SyncSessionState` :96,
  `BuildCostFunctionRegistryEntries` :4895, camera radios :2612/:2774, load
  slots :2207/:2358/:2502, `segmentHelperFunction` :1675,
  `ArrangeMainScreenLayout` :375.
- R13 relocation proof = `jj diff`; per-cut gates per parent R9/R14 (headless
  for logic/controller cuts; compile + scheduled manual-visual for
  presentation cuts).
- xcb-only rendering; `QQuickWindow::grabWindow` for QML smokes (established
  conventions).

---

## Outstanding Questions

### Resolve Before Planning

None.

### Deferred to Planning

- [Affects R7][Technical] Exact unit decomposition and cut order for plan 006
  (the ordering principle above narrows, not fixes, it).
- [Affects R7][Technical] Whether run-locking surfaces as controller-owned
  state each view maps onto its controls, or as a per-view helper (the QML
  side already maps via `runState`).
- [Affects R12][Needs research] The camera A/B slot bodies must be read at
  plan time to confirm exactly which state they write (calibration vs
  display) before extraction.
- [Affects R4][Technical] SessionState extension shape (previous-selection
  fields vs a small struct) and the widgets handlers' exact write points.
- [Affects R9][Technical] `matToVTK` + builder API shape and the class/file
  name in services.
- [Affects R10][Needs research] The four save-last-pose call sites' exact
  semantics (~:1094/:1142/:2972 + definition :4101), verified before
  unification.

---

## Next Steps

-> /ce-plan for structured implementation planning
