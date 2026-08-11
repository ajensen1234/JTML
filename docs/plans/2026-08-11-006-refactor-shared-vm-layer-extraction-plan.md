---
title: Shared VM-Layer Extraction — both front-ends, one controller layer
type: refactor
status: active
date: 2026-08-11
origin: docs/brainstorms/2026-08-11-shared-vm-layer-extraction-requirements.md
---

# Shared VM-Layer Extraction (both front-ends, one controller layer)

## Overview

Extract MainScreen's view-model-shaped surface (`src/view/mainscreen.cpp`,
5,115 lines) into a **shared controller layer** that BOTH the widgets app and
the QML experimental app (`src/app/experimental/` bridges) call, so MainScreen
shrinks to minimal view glue. The shared layer extends the existing layers —
QObject controllers in `jtml_coordinator`, plain services in `jtml_services`,
pure state in `jtml_domain` — with zero new CMake targets. Every cut is
byte-identical relocation first (R13): `jj diff` proves relocation, the
widgets' pinned behaviors are the spec, and the QML app keeps working at every
cut. Controllers are view-agnostic (headless-drivable under
`QCoreApplication`, R15/R16): no view/scene/renderer pointers, signals for
output.

---

## Problem Frame

The repo has two front-ends doing the same jobs twice: MainScreen fuses
View+ViewModel in its slots, and the QML bridges (StudyBridge/SettingsBridge/
OptimizerBridge/MlBridge/PoseBridge) are a working view-model layer that was
R13-forced to *replicate* widgets logic instead of sharing it (the
`LaunchOptimizer` drive sequence, `BuildCostFunctionRegistryEntries`, the VTK
pipeline). The owner wants the "good groove": new experiment surfaces land
once (a shared controller + a tiny per-view adapter), MainScreen becomes
minimal glue (~1,800–2,200 lines), and the shared layer is shaped so a future
headless config-file batch driver can drive it without any display. See
`docs/brainstorms/2026-08-11-shared-vm-layer-extraction-requirements.md`
(origin) and `docs/handoff-2026-08-11-vm-layer-extraction.md`.

---

## Requirements Trace

- R1. Shared layer extends existing layers — no new CMake target/prefix.
- R2. Pure Qt/GPU-free cores + thin QObject shells (plan-005 pattern).
- R3. Naming: `*Controller` in coordinator; no collision with existing classes.
- R4. SessionState becomes the one shared session core, extended with
  previous-selection mirrors + helpers.
- R5. Both views become write-back adapters over SessionState; no view owns
  private session bookkeeping.
- R6. QObject notification shell exposes SessionState changes; core stays
  signal-free.
- R7. `OptimizerRunController`: full LaunchOptimizer drive sequence
  (SaveLastPose mirror → intent gate → fresh manager + thread → Initialize →
  binds → start) + run-state machine + progress + stop + Initialize-failure
  quirk + run-locking.
- R8. `BuildCostFunctionRegistryEntries` shared (plan-005 committed follow-up);
  golden-fixture contract unchanged. (Deviation from origin R8/AE3 wording:
  the mapping lands in a sibling services module, not inside `SettingsService`
  — `SettingsService` stays QtCore-only/compute-free; same layer, different
  class.)
- R9. Widget-free render-pipeline builder shared by `Viewer` and
  `QmlVtkRenderer`; render-thread contract stays QML-side.
- R10. Session-state orchestration unifies (SyncSessionState + previous-frame
  bookkeeping + save-last-pose copies) onto SessionState + shared controller.
- R11. Study-load orchestration shared over SessionController parsing +
  caller-owned containers; per-seam behaviors preserved.
- R12. Camera A/B + segment/estimate orchestration thin onto shared seams.
- R13. Every cut is byte-identical relocation first; pinned widgets behaviors
  are the spec; QML app green at every cut. Per cut type: pure moves (U1, U7)
  are literal relocations; parameterized extractions (U3, U5) and QML
  thinnings are **verbatim-behavior extraction with signature adaptation** —
  their gate is the characterization test + behavior diff, not a literal
  `jj diff`.
- R14. MainScreen end-state residue is view-only; line count tracked
  (5,115 → ~1,800–2,200 target, a signal not a gate).
- R15. Shared layer view-agnostic: no view/scene/renderer pointers; drivable
  under QCoreApplication; grep-gate enforces.
- R16. Pose-sync chain generalizes: controller writes dataset + emits
  `poseUpdated`; views map onto their own scenes.

**Origin actors:** A1 (owner/developer — minimal-glue MainScreen, QML as
preferred experiment surface), A2 (coding agent — gate discipline, both apps +
oracle green per cut)
**Origin flows:** F1 (optimizer-run, both apps), F2 (study-load, both apps),
F3 (ML-initiated run, both apps)
**Origin acceptance examples:** AE1 (run-controller relocation + identical
widgets flow + parity), AE2 (session-state cut: private bookkeeping deleted,
selection byte-identical), AE3 (registry once + golden fixture passes),
AE4 (shared pipeline builder + both render smokes green), AE5 (end-state
view-only residue + parity re-verified)

---

## Scope Boundaries

- NOT the optimizer-backend modularity (parameters/cost/swappable backend) —
  later session (handoff).
- NOT the multi-stage oracle — the run controller is shaped as its future seam
  only.
- NOT QML islands on the widgets mainscreen; no full-QML front-end revisit.
- NOT touching `OptimizerManager`/`SessionController`/`CostFunctionManager`/
  compute internals; oracle fixtures + tolerances untouched.
- NOT changing pinned widgets behaviors: selection semantics, registry
  contract, drive-sequence quirks (Initialize-failure, onOptimizerError lock,
  camera conversion asymmetry), pose-file format, camera A/B behavior.
- NOT unifying pose-file formats or the settings registry contract.
- NOT deleting the QML bridges — they remain the QML-facing adapter shells.
- NOT building a headless driver/CLI this phase — only the view-agnostic shape.

### Deferred to Follow-Up Work

- Widgets menu-action guards (Optimize Backward / ML actions / Copy-Prev-Next /
  Load Pose/Kinematics reachable mid-run): preserve in the relocation cut,
  land as a separate gated behavior-delta cut exposing `runInFlight` (flow
  finding M7).
- Unification of the divergent save-last-pose conversion semantics (camera-B
  raw vs converted): after all four call sites sit on the shared core, one
  gated cut (flow finding H4).
- Widgets-side Initialize-failure quirk normalization: the shared
  controller's finished-before-Initialize bind + threadActive Start gate
  ship in U5 (the ghost is tracked; re-run is safe there); the widgets'
  quirk itself (thread started before the error box, UI state, message) is
  preserved verbatim per R13. Normalizing the widgets-side behavior is a
  gated cut (M6), as is the onOptimizerError unlock parity fix (M8).
- Multi-stage oracle consuming `OptimizerRunController`: the oracle workstream.
- The QML app adopting widgets' full run-directive surface (All/Each/From/
  Backward): v1 stays Single.

### Follow-On Optimization Phase (after this plan)

The "smarter ways" workstream — explicitly out of scope here per the handoff
("the optimizer-backend modularity is a LATER session — do not scope it
here") — is the algorithm-side roadmap from
`.panoptes/jtml-research-horizons/synthesis.org` (recommendation items 1–8).
This plan's job is to finish the boundary so that phase is cheap and
measurable:

- **Seam handoffs this plan provides:** the multi-stage oracle (synthesis
  item 3, the highest-leverage direction) consumes the `OptimizerRunDriver`
  seam from U5; the fake-driver tripwire makes optimizer changes
  headless-testable; the run-state/stage/progress surface is the oracle's
  observation channel; U10's adapter-thickness report closes the
  architecture arc.
- **The posture flip:** after U10, each change moves from "jj diff proves
  relocation" to "characterization test + before/after metric proves
  improvement". No algorithm deltas before the measurement apparatus exists
  — the current silhouette-IoU oracle cannot see the z-weak-axis problem
  (DIRECT convergence is noisy), and the cost path carries 7 live bugs with
  zero metric-math tests (synthesis item 1).
- **Recommended sequence:** (1) this plan's deferred quirk cuts (ghost
  thread, unlock parity, save-last-pose conversion unification, menu
  guards) — small and now measurable; (2) synthesis item 1 (fix the 7
  cost-path bugs, pin-first); (3) items 3 → 2 (production-shaped multi-stage
  oracle, then the metric-ablation harness) — the precondition for judging
  any smarter change; (4) item 5 (DIRECT variant switch — refinement
  control, not a new algorithm, is the highest-evidence optimizer upgrade:
  "whether the 20k trunk budget is being squandered on over-refinement");
  (5) item 8 (compute perf: per-eval sync removal, not kernel rewrites).
- **Data-structure note:** SessionState's extension in this plan is the last
  data-structure change of the relocation arc; deeper redesign
  (`LocationStorage`, pose representation, the 4-way save-last-pose
  divergence) belongs to the algorithm workstream, gated on the harness.

---

## Context & Research

### Relevant Code and Patterns

- `src/view/mainscreen.cpp` — verified inventory: `SyncSessionState` :96,
  `ArrangeMainScreenLayout` :375, `SaveLastPose` :4101 (call sites :1094,
  :1142, :2972, :3181, :4137), `DisableAll` :4051 / `EnableAll` :4072 (16
  controls each), `LaunchOptimizer` :4135–4325 (gate :4147–4160, thread+
  manager :4186–4213, Initialize-failure quirk :4216–4228, 7 binds
  :4230–4288), camera slots :2612/:2774 (inline save-last-pose :2641–2662 /
  :2801–2818), load slots :2207/:2358/:2502, `segmentHelperFunction` :1675,
  `BuildCostFunctionRegistryEntries` :4895–5070, dead `matToVTK` :72–91.
  (Note: plan-004's cited line numbers are stale — trust these.)
- `src/app/experimental/` — the bridges ARE the behavioral spec:
  `OptimizerBridge.{h,cpp}` (gate + state core public Qt/GPU-free,
  `saveScenePosesForCurrentSelection` call site :114 / definition :470, gate
  Input `previous==current` :338–340, run-state machine, seed lifecycle),
  `SettingsBridge.cpp` `buildCostFunctionRegistryEntries` definition :466
  (call :54 — verbatim replication),
  `QmlVtkRenderer.{h,cpp}` (1:1 pipeline mirror, render-thread contract),
  `StudyBridge::syncSessionState` :357–365, `ExperimentalSession` (embeds
  `jta::SessionState`).
- `include/domain/session_state.h` — pure holder: counts, current frame,
  selected rows, primary = first row. "Deliberately NOT an observable
  ViewModel."
- `include/coordinator/optimizer_manager.h` — `Initialize` wires
  started→Optimize / finished→quit/deleteLater **before** validation
  (:47–54); failure path still emits `finished` (:885–911); terminal
  OptimizedFrame :1178–1193; cooperative stop :1393. **`Initialize` is
  non-virtual** — a test subclass cannot intercept it (drives the
  `OptimizerRunDriver` seam design, U5).
- `include/services/session_controller.h` — `CameraRadioEvent` :60 +
  `DecideCameraRadios` (headless-pinned in `test/unit/session_controller_test.cpp:406–439`).
- `test/CMakeLists.txt` — `jtml.coordinator` QtTest lifecycle pattern (real
  QThread + stub cost + QSignalSpy, :373–388); direct-compile pattern;
  `jtml.experimental_*` bridge tests; `jtml.qml_parity_check` (IoU 0.993627);
  render smokes. QtTest for QObject/QThread seams; Catch2 for pure logic.

### Institutional Learnings

- `docs/solutions/conventions/jtml-layered-lib-split-2026-08-08.md` — the 003
  relocation ritual: exact-string scripted include rewrites (ast-grep cannot
  target `#include`), `jj diff` every changed line, zero-prefix greps; header
  GLOB `CONFIGURE_DEPENDS` + **explicit `.cpp` list per layer**; AUTOMOC
  undefined-symbol trap (Q_OBJECT header must be in target sources).
- `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md`
  — MVVM strangle rules (extract decision/state; keep real production binding;
  reproduce quirks + pin before normalizing); per-cut gates (headless for
  logic, compile + manual-visual for presentation).
- `docs/solutions/conventions/jtml-qml-experimental-frontend-2026-08-11.md` —
  bridge thinness rule; render-thread contract (dispatch_async = render
  thread, capture by value); pose-sync via direct by-value signal emit
  (queued functor invokeMethod silently fails — see ui-bugs doc); the
  "view-model layer" section names this extraction explicitly.
- `docs/solutions/conventions/jtml-mainscreen-decomposition-patterns-2026-08-10.md`
  — silent-wiring-death class (by-name auto-connects die silently; `setModel()`
  before selection-model connects); byte-identical interleave via view-owned
  loop + controller per-frame ops; extract per-site verbatim; torch includes
  after Qt-object headers (`#undef slots`).
- `docs/solutions/ui-bugs/jtml-qml-model-pose-sync-queued-functor-never-delivered-2026-08-11.md`
  — direct by-value signal emit, receiver-affinity delivery; instrument the
  receiver; rebuild-and-re-run discipline.
- `docs/solutions/build-errors/jtml-moc-signals-section-placement-duplicate-definition-2026-08-11.md`
  — `signals:` sections last in Q_OBJECT classes (mid-class placement turns
  accessors into signals → duplicate definitions).
- `docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md`
  — appearance oracle conventions (IoU vs label, never raw pose); hegel PBT
  next to deterministic Catch2; `QSurfaceFormat::setDefaultFormat` before
  QApplication.
- `docs/solutions/tooling-decisions/jtml-rendering-runtime-xcb-qvtk-2026-08-10.md`
  — xcb-only; `vtkWindowToImageFilter` segfaults; grab-based smokes.

### External References

None — the repo's five prior plans + `docs/solutions/` corpus are the
authoritative patterns; no external contract surfaces are touched.

---

## Key Technical Decisions

- **No new layer** (R1): QObject controllers join `jtml_coordinator`; plain
  services join `jtml_services`; SessionState stays in `jtml_domain`.
  `SettingsService` itself stays QtCore-only — the registry *mapping* becomes
  a sibling services module (services already links compute PRIVATE).
- **Previous-mirror semantics pinned** (H2): `previous_frame` /
  `previous_model_rows` mean "last-selected, == current in steady state" (the
  widgets' `previous_frame_index_` semantics). They feed **save-last-pose
  only** — the run controller builds gate input with `previous == current`
  exactly as `OptimizerBridge` does today. This prevents every QML run being
  rejected after a frame change.
- **Epoch-tagged runs + threadActive Start gate** (H1/M6): every run gets a
  generation counter; relays from stale epochs are dropped; `Start()` rejects
  while a previous thread is alive (covers the Initialize-failure ghost);
  `finished` is bound **before** `Initialize` so ghost termination is
  observed. Fixes the cross-run race where run 1's terminal frame corrupts
  run 2 (and `onFinished` waits on run 2's thread).
- **Injectable drive seam** (M12/Q8, Q8 = plan-004 flow-question ID): the
  controller depends on a narrow `OptimizerRunDriver` interface
  (`Initialize(...)`, `Start()`, `Stop()`, `Wait()` + the manager QObject for
  signal binding) with a production adapter wrapping `OptimizerManager`
  **untouched** — `OptimizerManager::Initialize` is non-virtual, so a test
  subclass cannot intercept it; the seam is the driver, not the manager.
  Tests implement the interface and emit the 7+1 signals, recording
  Initialize args + bind order; this preserves the "NOT touching
  OptimizerManager internals" boundary and is the multi-stage oracle's future
  entry point. Alternative considered: virtualizing
  `OptimizerManager::Initialize` (rejected — touches a shared coordinator
  class and would need a scope-boundary amendment). R12 note: the test double
  is a compute-side driver behind an explicit seam, not a Qt-mocking wrapper
  of view/dialog classes; production binding unchanged.
- **Destructor contract** (H3): stop (cooperative — DIRECT poll is prompt) →
  `quit()` + `wait()` with timeout → delete; ghosts waited via the
  pre-Initialize `finished` bind. Fixes the app-close-mid-run thread crash
  both apps share today. **Timeout-expiry semantics fixed**: never delete a
  running thread (that reproduces the crash); on expiry, log a warning and
  keep waiting — bounded DIRECT iterations make expiry a diagnostic, not a
  branch.
- **Save-last-pose parameterized core** (H4/Q6, Q6 = plan-004 flow-question
  ID): `(pose-source functor, selection set, target frame, convert rule)` —
  the four call-site behaviors are individually correct given their source's
  coordinate frame; a call-site table test (4 rows) pins each before any
  unification. Only the camera-B *comment* is fixed (it claims a conversion
  it doesn't do — the raw save is correct).
- **Severity-carrying messages** (L14): `messageRequested(title, message,
  severity {Info, Warning, Critical})`; QML ignores severity (single Dialog);
  widgets preserves its box-type distinctions. The OptimizedFrame
  out-of-bounds status travels on the relay so the widgets view can box.
- **Diff-based grouped session signals** (M9): emit `datasetChanged` /
  `selectionChanged` only on actual value change, **after** previous mirrors
  are consistent (current code updates mirrors after SyncSessionState — an
  emit inside would notify with stale previous).
- **Run-locking** (M8): controller exposes `state` + `threadActive`; widgets
  keeps its terminal-frame unlock point — EnableAll is driven by the
  controller's terminal-frame relay (the pinned unlock-after-error behavior:
  widgets unlocks at the terminal OptimizedFrame even with the error bit
  set), not purely by state; QML keeps binding `running()`. The 16-control
  enumeration stays view glue.
- **SingleModelOnly stays a QML policy**: the v1 single-model rule is a
  bridge-side pre-check on top of the shared gate (widgets supports multi-run
  directives; v1 QML does not).
- **Seed restored on Initialize failure** (M10a): snapshot before
  `applySeedPose`, restore on the failure path — the estimate-wins-over-drift
  guarantee survives a failed run.
- **ClearDataset resets mirrors + seed** (H5/M10b): prevents cross-dataset
  bogus save-last-pose writes once real mirrors exist.
- **View-side directive reset** (M11): widgets' All/Each index-0 selection
  reset happens **before** `Start()` (two-phase contract) so the selection
  handler's guards behave identically.
- **Display conversions stay view-side** (L15): A↔B display conversions read
  camera radio + VTK state; the controller relays raw A-coord poses; each
  view converts.
- **Ordering deviation from the origin exemplar** (registry → run controller
  → pipeline builder → session core): the render builder lands before the run
  controller (U4 vs U5) to de-risk the phase's main technical unknown (the
  QQuickVTKItem render-thread contract) early; U4 and U5 are
  dependency-independent, so the flagship run controller is not blocked by
  the builder.

---

## Open Questions

### Resolved During Planning

- Run-locking surface: controller-owned `state`/`threadActive`; views map
  (widgets terminal-frame parity; QML `running()`).
- Camera A/B scope: VM slice = active-camera state (via session core) + the
  two inline save-last-pose copies; radio decisions + display-only blocks
  stay view-side (L16).
- SessionState extension shape: `previous_frame_` (int, -1) +
  `previous_model_rows_` (vector<int>) + setters/helpers — the file's
  existing idiom.
- Builder API shape: value-parameterized recipe (free functions / stateless
  builder taking renderers/import/actors by pointer) — never a QObject owning
  VTK state (render-thread constraint).
- Save-last-pose call sites: 5 (SaveLastPose) + 2 inline camera + 1 QML
  mirror = 4 divergent behaviors, table-pinned.

### Deferred to Implementation

- Exact class/method names for the shared modules (e.g. the registry mapping
  and builder file names) — the plan's file names are proposals.
- The builder's pure configuration helper signature (mat → import params).
- The exact timeout *duration* constant (semantics fixed in Key Technical
  Decisions: warn + keep waiting on expiry; never delete a running thread).
- Exact message text for the new re-run-rejected dialog.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

### Run-state machine (shared by both apps)

```
                    ┌────────── Start() gate ──────────┐
                    │  Idle/Completed/Error            │
                    │  AND !threadActive (epoch OK)    │
                    ▼                                  │
 [*] ──► Idle ──► Running ──► Stopping ──► Completed
              ▲      │   ▲        │             │
              │      │   └────────┘  terminal OptimizedFrame
              │      │              (stale-epoch relays dropped)
              │      └──► Error (OptimizerError | Initialize failure)
              └────────── re-run only when threadActive == false
```

**Error is not a dead-end for the widgets mapper:** the manager always emits
a terminal OptimizedFrame (error bit set) after an OptimizerError; the
controller relays it, and the widgets view unlocks on that relay exactly as
today (EnableAll at the terminal frame regardless of the error bit — pinned
unlock-after-error behavior). QML keeps its "stays Error" runState semantics
(applyOptimizedFrame keeps Error when an OptimizerError already moved the
run there).

### Controller shape (one example: OptimizerRunController)

- **Plain core** (Qt/GPU-free, headless-tested): gate Input assembly
  (`previous == current`), 5-state machine + `threadActive`, epoch counter,
  progress/stage mapping (calls vs cumulative budget), seed lifecycle
  (apply/restore/clear).
- **QObject shell** (coordinator): injectable `OptimizerRunDriver` seam
  (production adapter wraps `OptimizerManager` untouched); thread
  ownership + destructor contract; `finished` bound before `Initialize`;
  8 binds (7 + `finished`) relaying through by-value signals re-emitted on
  the controller thread; `messageRequested(title, message, severity)`;
  `poseUpdated(frameIndex, x..za)` (R16 — views map to their scenes).
- **Views**: widgets `LaunchOptimizer` → `controller->start(directive)` +
  view-side DisableAll/EnableAll mapper (EnableAll on the terminal-frame
  relay); QML `OptimizerBridge` → thin shell (Q_PROPERTYs + SingleModelOnly
  pre-check + Dialog mapping).

---

## Implementation Units

- [x] U1. **[Registry mapping extraction]**

**Goal:** One shared `BuildCostFunctionRegistryEntries` in services; both apps
call it; the golden fixture re-pins onto the shared function (its own
regeneration note mandates this).

**Requirements:** R8; AE3.

**Dependencies:** None.

**Files:**
- Create: `include/services/cost_function_registry.h`,
  `src/services/cost_function_registry.cpp`
- Modify: `src/view/mainscreen.cpp` (mapping body :4895–5070 → call),
  `src/app/experimental/SettingsBridge.cpp` (replication :466 → call),
  `test/unit/experimental_settings_test.cpp` (re-pin golden fixture),
  `src/services/CMakeLists.txt` (explicit `.cpp` list)
- Test: `test/unit/cost_function_registry_test.cpp`

**Approach:**
- Relocate the mapping verbatim into a services module taking the three
  `CostFunctionManager`s (services already links compute PRIVATE — the
  "QtCore-only" constraint applies to `SettingsService` itself, not the
  module). Key formats `STAGE@ACTIVE_CF` / `STAGE@CFname@ParamName@TYPE`,
  ACTIVE_CF-first entry order preserved.
- Widgets `onSaveSettings` (:4865) and `SettingsBridge::save()` call the
  shared function. The golden 51-entry fixture (17/stage, `%.17g` exact
  IEEE-754 round-trip) is re-pinned to call the shared function directly and
  must produce the identical table.

**Execution note:** Characterization-first — re-pin the fixture against the
relocated function before any cleanup; the fixture IS the spec.

**Patterns to follow:** 003 relocation ritual (`jj diff` every line);
`test/unit/experimental_settings_test.cpp` golden-table pattern.

**Test scenarios:**
- Happy path: shared function with the widgets first-run manager configuration
  (fresh managers + branch/leaf Dilation=4/1) produces the exact golden
  51-entry table (17 per stage, `%.17g`).
- Happy path: `SettingsBridge::save()` → registry → `LoadSettings` round-trip
  unchanged (existing test (d) re-run against the shared call).
- Edge case: empty/zero-manager configuration → empty entries, no crash.
- Integration: widgets `onSaveSettings` writes the same registry content as
  before (golden compare).

**Verification:** `jtml.experimental_settings` + `jtml.cost_function_registry`
green; `jj diff` shows relocation + fixture re-pin only; widgets + QML apps
compile.

---

- [ ] U2. **[SessionState previous-selection mirrors]**

**Goal:** Extend `jta::SessionState` with previous-selection mirrors + helpers
— the pure foundation for save-last-pose and the session-state controller.

**Requirements:** R4; supports R10.

**Dependencies:** None.

**Files:**
- Modify: `include/domain/session_state.h`, `src/domain/session_state.cpp`
- Test: `test/unit/test_session_state.cpp`,
  `test/unit/test_session_state_properties.cpp` (hegel PBT)

**Approach:**
- Add `previous_frame_` (int, init -1), `previous_model_rows_`
  (std::vector<int>, sorted like `selected_models_`), setters/getters +
  `HasPreviousSelection()`. Semantics pinned in the header comment:
  **"last-selected frame/model set — equals the current selection in steady
  state; consumed by save-last-pose only, never fed to the optimizer gate"**
  (flow finding H2).
- Purely additive — no consumers this unit.

**Patterns to follow:** existing `selected_models_` idiom in
`include/domain/session_state.h`; hegel PBT next to Catch2
(`test_session_state_properties.cpp`).

**Test scenarios:**
- Happy path: set/get round-trip for both mirrors; defaults are -1/empty.
- Edge case: `previous_model_rows_` remains sorted after arbitrary writes
  (invariant shared with `selected_models_`).
- Edge case: `HasPreviousSelection()` false with no previous frame or empty
  rows; true only when both set.
- Property: PBT round-trip + sortedness over generated row sets.

**Verification:** `jtml.session_state` + `_props` green; domain purity grep
(`QObject|vtk|cuda|torch|opencv` in `src/domain`) still clean.

---

- [ ] U3. **[SaveLastPose shared core]**

**Goal:** One parameterized save-last-pose core in services; the four
divergent call-site behaviors relocate onto it **preserved verbatim**, pinned
by a call-site table test.

**Requirements:** R10 (part), R13; supports R7, R12; AE2 (part).

**Dependencies:** U2.

**Files:**
- Create: `include/services/save_last_pose.h`, `src/services/save_last_pose.cpp`
- Modify: `src/view/mainscreen.cpp` (definition :4101–4129 → call; call sites
  unchanged), `src/app/experimental/OptimizerBridge.cpp`
  (`saveScenePosesForCurrentSelection` call :114 / definition :470 → call),
  `src/services/CMakeLists.txt`
- Test: `test/unit/save_last_pose_test.cpp`

**Approach:**
- Parameterized core: `SaveLastPoseToStorage(previous_frame, model_rows,
  pose_source /* std::function<Point6D(int row)> → position+orientation */,
  camera_is_a, convert_rule, calibration, storage)`.
- Call-site table (the four behaviors are each correct for their source's
  coordinate frame — pin, don't unify):
  1. widgets `SaveLastPose`: previous selection, previous frame, `vw` source,
     convert iff camera B checked.
  2. camera-A slot inline: **current** selection, previous frame,
     `model_actor_list` source, **always** convert B→A.
  3. camera-B slot inline: current selection, previous frame, `vw` source,
     **never** convert (fix only its misleading comment).
  4. QML mirror: current selection, current frame, scene source, never
     convert.
- MainScreen's `SaveLastPose` definition relocates onto the core with
  signature adaptation (`vw` reads → pose-source functor,
  `ui.camera_A_radio_button` → `camera_is_a`), behavior pinned by the
  call-site table test (call sites :1094/:1142/:2972/:3181/:4137 unchanged —
  R13 per-cut-type rule). The camera inline copies converge in U9;
  unification of the conversion divergence is Deferred to Follow-Up Work.

**Execution note:** Table test first (characterization), then relocation —
the widgets drive sequence has no other headless tripwire (flow finding L18).

**Patterns to follow:** `OptimizerBridge.cpp` scene-pose read shape (call
:114 / definition :470);
`docs/solutions/conventions/jtml-mainscreen-decomposition-patterns-2026-08-10.md`
(extract per-site verbatim).

**Test scenarios:**
- Happy path (table): each of the 4 rows — given (source, selection, frame,
  convert rule), the core writes the expected `LocationStorage::SavePose`
  arguments.
- Edge case: no previous selection (empty rows / -1 frame) → no-op, no write.
- Edge case: convert-rule matrix — A checked (no convert), B checked (convert
  B→A) for the canonical row.
- Error path: pose-source functor returns out-of-range/zero for a row →
  skipped without corrupting other rows.
- Integration: widgets call sites produce identical storage contents as the
  pre-cut binary (manual-visual + storage dump compare).

**Verification:** `jtml.save_last_pose` green; `jj diff` shows relocation;
widgets manual-visual selection/save flow unchanged.

---

- [ ] U4. **[Render pipeline builder]**

**Goal:** A widget-free, value-parameterized VTK pipeline recipe in services
that `Viewer` and `QmlVtkRenderer` both call — removing the 1:1 mirror and the
dead `matToVTK`.

**Requirements:** R9; AE4.

**Dependencies:** None.

**Files:**
- Create: `include/services/render_pipeline_builder.h`,
  `src/services/render_pipeline_builder.cpp`
- Modify: `src/app/experimental/QmlVtkRenderer.cpp` (mirror chains → builder),
  `src/view/viewer.cpp` (chains :15–60, :89–118, :160–203 → builder),
  `src/view/mainscreen.cpp` (delete dead `matToVTK` :72–91),
  `include/view/mainscreen.h` (remove the `matToVTK` declaration),
  `src/services/CMakeLists.txt`
- Test: `test/unit/render_pipeline_builder_test.cpp` (pure config part)

**Approach:**
- A stateless recipe / free functions taking VTK objects by pointer (renderer,
  image import, actors): ① background chain construction + zero-copy import
  configuration (spacing/origin/extent/scalar-type/channels/
  `SetImportVoidPointer`/Update), ② model chain (reader → mapper → actor),
  ③ camera setup (parallel bg placement + scale; perspective scene view-angle
  + clipping), ④ background refresh, ⑤ actor pose apply. The pure
  mat→import-parameter derivation is a standalone function with unit tests;
  the VTK-touching calls are exercised by the smokes.
- QML side calls it inside `initializeVTK`/`dispatch_async` with
  render-thread-owned objects (the contract stays QML-side — see
  `docs/solutions/conventions/jtml-qml-experimental-frontend-2026-08-11.md`);
  widgets `Viewer` calls from the GUI thread. Interactor styles stay view-side
  (they differ: picking vs implicit-pick workaround).
- The three documented camera divergences between the views are builder
  **parameters, not unified behavior**: (a) background Z placement (widgets
  `-fy*pixel_pitch` vs QML `-focalLengthPx`), (b) scene-camera focal pivot
  (widgets near-origin vs QML primary-model z), (c) window-center/aspect
  calibration plumbing. (a)+(b) become parameters with per-view values;
  (c) stays view-side — the QML comment's "lands with U4" expectation is
  marked stale (it is a QML-side camera-calibration concern, not
  shared-builder scope).
- `matToVTK` is dead in the widgets app (zero call sites); the builder's
  import configuration absorbs its semantics, then it is deleted.

**Execution note:** Land the QML refactor first (biggest duplication win,
render smoke gated), then `Viewer`.

**Patterns to follow:** `QmlVtkRenderer.cpp` mirror table (`QmlVtkRenderer.h:21–37`);
`Viewer` chain construction; grab-based smoke recipe
(`test/oracle/qml_render_smoke.cpp`).

**Test scenarios:**
- Happy path (pure): mat → import params matrix — gray vs color, channel
  count, extent, spacing/origin zeros — exact expected values.
- Edge case (pure): empty/zero-size mat → valid degenerate params, no crash.
- Integration: `jtml.render_smoke` + `jtml.qml_render_smoke` green after both
  refactors (background + model + camera identical to pre-cut captures).
- Integration: QML viewport interaction (model drag, camera orbit) unchanged
  per manual-visual.

**Verification:** both render smokes green; `jj diff` shows the mirror removed
and `matToVTK` deleted; QML render parity unchanged.

---

- [ ] U5. **[OptimizerRunController]**

**Goal:** The shared run controller — gate, drive sequence, run-state
machine, progress, stop, seed, epoch/thread lifecycle, destructor contract.
MainScreen's `LaunchOptimizer` + locking and `OptimizerBridge` thin onto it.

**Requirements:** R7, R13, R15, R16; F1; supports R10; AE1.

**Dependencies:** U2, U3 (reuses the existing `jta::OptimizeIntentController`
gate seam).

**Files:**
- Create: `include/coordinator/optimizer_run_driver.h` (drive interface),
  `src/coordinator/optimizer_run_driver.cpp` (production adapter over
  `OptimizerManager`, untouched), `include/coordinator/optimizer_run_controller.h`,
  `src/coordinator/optimizer_run_controller.cpp`
- Modify: `src/view/mainscreen.cpp` (LaunchOptimizer :4135–4325 → controller;
  DisableAll/EnableAll :4051–4098 → view-side mapper over controller state;
  the onUpdateDisplay/onOptimizedFrame cores relocate), `src/view/mainscreen.h`,
  `src/app/experimental/OptimizerBridge.{h,cpp}` (thin shell: Q_PROPERTYs +
  SingleModelOnly pre-check + Dialog mapping), `src/coordinator/CMakeLists.txt`
- Test: `test/lifecycle/optimizer_run_controller_test.cpp` (QtTest),
  `test/unit/optimizer_run_controller_core_test.cpp` (ports
  `test/unit/experimental_optimizer_gate_test.cpp`)

**Approach:**
- **Plain core** (Qt/GPU-free): the drive sequence — SaveLastPose mirror via
  U3's core pinned **before** the gate (per mainscreen.cpp:4137 /
  OptimizerBridge.cpp:114), then gate evaluation via the existing
  `jta::OptimizeIntentController::Evaluate` (Input assembly in the core with
  **`previous == current`** — H2; the SingleModelOnly pre-check stays a QML
  bridge policy), 5-state machine + `threadActive` (M8), epoch counter,
  progress/stage mapping (calls vs cumulative budget — the oracle seam, M12),
  seed lifecycle (apply after gate; **restore on Initialize failure** — M10a;
  stale guards).
- **QObject shell**: injectable drive seam (narrow `OptimizerRunDriver`
  interface; production adapter wraps `OptimizerManager` untouched — M12/Q8);
  thread ownership; **`finished` bound before `Initialize`** (M6); **Start
  gate**: Idle/Completed/Error-with-no-thread AND !threadActive, epoch-tagged
  relay drops (H1); 8 binds (7 + `finished` — L13) relaying by-value signals
  re-emitted on the controller thread (QTBUG-2842 pattern);
  `messageRequested(title, message, severity)` (L14); out-of-bounds status on
  the frame relay; `poseUpdated` (R16); typed directive enum → string mapping
  (widgets' directives preserved; QML v1 = Single).
- **Destructor contract** (H3): request stop → `quit()` + `wait()` (bounded)
  → delete; ghost threads waited via the pre-Initialize `finished` bind; on
  wait timeout, never delete a running thread — warn and keep waiting
  (bounded DIRECT iterations make expiry a diagnostic).
- **Widgets wiring**: `LaunchOptimizer` becomes
  `controller->start(directive)`; the 16-control DisableAll/EnableAll
  enumeration becomes a view-side mapper, with **EnableAll driven by the
  controller's terminal-frame relay** (the pinned unlock-after-error
  behavior — widgets unlocks at the terminal OptimizedFrame even with the
  error bit set), not purely by state. **Acknowledged delta**: a widgets
  re-run click in the terminal-frame→thread-death window (EnableAll fires
  while the old thread is still finishing) is rejected by the Start gate with
  a message; millisecond-scale in the normal path — covered by a U5 test,
  not silent. The Initialize-failure and onOptimizerError quirks reproduce
  verbatim.
- **QML wiring**: `OptimizerBridge` keeps its Q_PROPERTY surface, the
  SingleModelOnly pre-check (after the shared gate's first guard), and the
  Dialog mapping; the run-state machine, thread lifecycle, progress math, and
  save-last-pose mirror move into the controller. QML's re-run-while-ghost
  hole closes via the Start gate (deliberate, tested behavior improvement on
  the QML side — H1). **Second enumerated QML delta (M10a)**: after a failed
  Initialize, the seed-restore reverts storage/scene to the pre-seed pose —
  the estimate is not silently kept; small and deliberate, listed here so it
  is not surprising.

**Execution note:** Gate-first — write the fake-driver drive-sequence test
(bind set/order, gate-input rule, SaveLastPose-before-gate, seed ordering,
stop, ghost, epoch-drop, destructor mid-run) **before** the relocation. This
is the missing tripwire for the widgets drive sequence (L18).

**Patterns to follow:** `OptimizeCoordinator` lifecycle test pattern
(`test/lifecycle/coordinator_test.cpp`, real QThread + QSignalSpy);
`OptimizerBridge.h` gate/state-core split; the 003 relocation ritual.

**Test scenarios:**
- Happy path: fake driver (implements the drive interface, emits the 7+1
  signals) — full drive sequence: SaveLastPose mirror (before the gate) →
  gate pass → seed applied → Initialize args by value (containers + plain
  rows) → 8 binds connected in order → start; terminal OptimizedFrame →
  Completed.
- Happy path: progress mapping — calls/min/stage derivation (incl. cumulative
  budget math) matches the widgets onUpdateDisplay level logic.
- Edge case (H2): gate Input built with `previous == current` regardless of
  the session mirrors; a frame jump between sync and run still passes the
  gate when selection is valid.
- Edge case (H1): error → immediate re-run attempt rejected while
  `threadActive`; stale-epoch relays (UpdateDisplay/OptimizedFrame from run
  1) arriving during run 2 are dropped.
- Edge case (M6): Initialize failure — ghost thread termination observed via
  the pre-Initialize `finished` bind; `threadActive` clears; re-run opens.
- Error path (M10a): Initialize failure after `applySeedPose` → seed
  restored; a subsequent run starts from the estimate.
- Error path: gate rejection — both statuses (SelectFrameAndModel,
  PoseMatrixDimensionMismatch) surface with the widgets' distinct message
  texts via severity-carrying messages; state unchanged; seed NOT consumed.
- Error path (H3): controller destroyed mid-run (fake driver) → cooperative
  stop + bounded wait + no crash/hang.
- Integration: ported gate tests (all `experimental_optimizer_gate_test`
  cases) green against the shared core.
- Integration: OptimizerError → terminal OptimizedFrame (error bit) →
  widgets unlock on the relay (pinned unlock-after-error); QML runState
  stays Error.
- Edge case: widgets terminal-frame unlock + immediate re-run in the
  thread-death window → Start gate rejects with a message (acknowledged
  delta).
- Integration: widgets manual-visual run (Single + All/Each/From/Backward) +
  `jtml.oracle` + `jtml.qml_parity_check` green; QML app run via thinned
  bridge.

**Verification:** QtTest lifecycle + ported gate tests green; both apps
compile + startup; widgets run flow behavior-identical per manual-visual
(R13 per-cut-type: characterization test + behavior diff); oracle + parity
green.

---

- [ ] U6. **[SessionStateController + MainScreen bookkeeping removal]**

**Goal:** The QObject notification shell over SessionState; MainScreen's
private session bookkeeping deleted; both views write through the one core.

**Requirements:** R5, R6, R10; AE2.

**Dependencies:** U2, U5 (locking state surface).

**Files:**
- Create: `include/coordinator/session_state_controller.h`,
  `src/coordinator/session_state_controller.cpp`
- Modify: `src/view/mainscreen.cpp` (`SyncSessionState` :96–118 → controller;
  selection handlers :2960/:3177 write mirrors via controller;
  `previous_frame_index_`/`previous_model_indices_` deleted;
  `curr_frame()` unchanged behaviorally), `include/view/mainscreen.h`,
  `src/app/experimental/StudyBridge.cpp` (`syncSessionState` :357–365 →
  controller), `src/coordinator/CMakeLists.txt`
- Test: `test/lifecycle/session_state_controller_test.cpp` (QtTest),
  `test/unit/session_state_controller_test.cpp` (sync logic)

**Approach:**
- Controller owns SessionState; **diff-based grouped emission**
  (`datasetChanged`, `selectionChanged`) only when a mirrored value actually
  changes, emitted **after** previous mirrors are consistent (M9/H2 ordering
  constraint).
- `SyncSessionState` logic relocates byte-identically (counts/labels);
  MainScreen's selection handlers write mirrors through the controller
  instead of private members (R5 — the QModelIndexList → rows conversion is
  the loop SyncSessionState already does); `previous_frame_index_` /
  `previous_model_indices_` deleted from `mainscreen.h`.
- **ClearDataset path resets previous mirrors + seed** (H5/M10b) and emits
  `datasetChanged`.
- Exposes `runInFlight` for the follow-up guards (M7) and the study-load
  guard (L17).

**Execution note:** Characterization-first — pin the current sync/selection
behavior with the manual-visual checklist before relocating; the widgets
selection semantics are the pinned spec (R13).

**Patterns to follow:** `StudyBridge::syncSessionState` mirror;
`OptimizeCoordinator` QObject-shell pattern; diff-emission precedent in the
QML bridges' dirty tracking.

**Test scenarios:**
- Happy path: dataset load → `datasetChanged`; selection change → `selectionChanged`
  with correct rows/primary/mirrors.
- Edge case (M9): no actual value change → no signal (diff emission);
  repeated syncs don't spam.
- Edge case (H2): mirrors always consistent at emission time — a consumer
  observing `selectionChanged` sees previous == pre-change current.
- Edge case (H5): ClearDataset → mirrors reset to -1/empty, seed cleared,
  `datasetChanged` emitted.
- Integration: widgets selection behaviors (primary = first row, multi-select,
  save-last-pose on change) byte-identical per scheduled manual-visual.
- Integration: QML app behavior unchanged (StudyBridge thinned).

**Verification:** QtTest + unit green; widgets manual-visual selection flow
identical; QML app unchanged; `jj diff` shows relocation + deletions.

---

- [ ] U7. **[Study-load orchestration]**

**Goal:** One shared load path over `SessionController` parsing +
caller-owned containers; the widgets load slots and `StudyBridge` thin onto
it; per-seam behaviors preserved.

**Requirements:** R11, R13; F2; supports R5.

**Dependencies:** U6.

**Files:**
- Create: `include/services/study_load_controller.h`,
  `src/services/study_load_controller.cpp`
- Modify: `src/view/mainscreen.cpp` (load slots :2207/:2358/:2502 → controller;
  calibration one-use + dataset-replace semantics relocate),
  `src/app/experimental/StudyBridge.cpp`, `src/services/CMakeLists.txt`
- Test: `test/unit/study_load_controller_test.cpp`

**Approach:**
- Relocate parse → populate → dedup → counts → `SyncSessionState` tail
  verbatim into a plain services controller (the scene/background update
  orchestration stays view-side — each view maps the load result onto its own
  renderer). Calibration-one-use-per-session and dataset-replace (calibration
  kept) semantics are the spec, preserved exactly.
- `runInFlight` guard lives in the VIEW slot (services cannot reference
  coordinator — layering); the plain load controller takes the flag as an
  injected `std::function<bool()>` when it must consult it (L17 — QML
  currently lacks the guard; widgets' DisableAll covers its buttons, the
  shared check is defense-in-depth).

**Patterns to follow:** `StudyBridge.cpp` load flow; `SessionController`
parse APIs (`ParseCalibration`/`ParseImages`/`ParseModels`/`PopulateModels`).

**Test scenarios:**
- Happy path: full study load → containers populated, counts set, session
  mirrors updated.
- Edge case: dataset replace (second study) → old frames/models wiped,
  calibration kept; mirrors + seed reset via U6.
- Edge case: calibration one-use — second calibration load rejected (widgets
  parity).
- Error path: parse failure (missing file, bad calibration) → clear error
  status, no partial dataset.
- Integration: both apps' load flows green (manual-visual: load → lists →
  viewport first frame + models).

**Verification:** `jtml.study_load_controller` green; both apps' load flows
manual-visual identical; `jj diff` relocation.

---

- [ ] U8. **[ML orchestration]**

**Goal:** One shared segment/estimate orchestration over
`SegmentationController`/`ImplantEstimator`; MainScreen's segment/estimate
slots and `MlBridge` thin onto it; seed lifecycle via U5.

**Requirements:** R12 (part), R13; F3; AE4 (plan-005 origin —
`docs/brainstorms/2026-08-11-qml-experimental-frontend-requirements.md` —
graceful degradation).

**Dependencies:** U5 (seed API).

**Files:**
- Create: `include/services/ml_orchestrator.h`, `src/services/ml_orchestrator.cpp`
- Modify: `src/view/mainscreen.cpp` (`segmentHelperFunction` :1675 + the two
  estimate paths → orchestrator), `src/app/experimental/MlBridge.cpp`,
  `src/services/CMakeLists.txt`
- Test: `test/unit/ml_orchestrator_test.cpp` (+ existing
  `jtml.experimental_ml_bridge` kept green)

**Approach:**
- Relocate the segment → estimate → SavePose → seed chain verbatim into a
  plain services orchestrator (.pt loading + status surface stay view-side;
  torch include-after-Qt rule applies). Graceful degradation without `.pt`
  models preserved (clear status, plain-optimize path).
- The orchestrator takes segment/estimate **callables** (or a
  `SegmentationController` reference behind a small interface) —
  `SegmentFrame`/`EstimateImplantPose` are not stubbable as-is (raw
  `torch::jit::Module*` + CUDA compute), so the injected seam is what makes
  the happy path and segment-failure scenarios headless-testable; the
  .pt-load and real CUDA estimate paths stay manual-visual.
- Seed handoff: the orchestrator **returns** the seed (or takes a
  `std::function` setter); the VIEW wires it to the run controller — no
  services→coordinator references (layering; stale guards live in the run
  controller, M10a restore applies).

**Execution note:** Characterization-first — the segment/estimate chain is
GPU/torch-heavy; pin with the existing bridge tests + scheduled
manual-visual before relocating.

**Patterns to follow:** `MlBridge.cpp` chain; `SegmentationController` /
`ImplantEstimator` APIs; `docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md`
(oracle conventions for the estimate).

**Test scenarios:**
- Happy path: segment → estimate → seed produced, driven through the
  injected callables with stubs (headless); the real torch/CUDA path is
  manual-visual.
- Edge case: no `.pt` models → clear degradation status, no crash, plain
  run unaffected.
- Edge case: frame/model change between estimate and run → stale seed dropped
  (U5 guard).
- Error path: segment failure → status surfaced, no seed.
- Integration: widgets ML menu flow + QML ML flow manual-visual; parity
  green.

**Verification:** `jtml.ml_orchestrator` green; both apps' ML flows
manual-visual; `jj diff` relocation.

---

- [ ] U9. **[Camera A/B orchestration]**

**Goal:** The VM slice of the camera slots — active-camera state via the
session core and the two inline save-last-pose copies onto the U3 core —
extracted; display-only blocks stay in the view.

**Requirements:** R12 (part), R13; supports R10.

**Dependencies:** U3, U6.

**Files:**
- Modify: `src/view/mainscreen.cpp` (camera slots :2612–2743/:2774–2960:
  inline save-last-pose :2641–2662/:2801–2818 → U3 core; active-camera write
  via session core; radio decisions + display blocks stay),
  `include/view/mainscreen.h`
- Test: `test/unit/save_last_pose_test.cpp` (camera rows already pinned in
  U3), `test/unit/session_controller_test.cpp` (existing DecideCameraRadios
  pin stays green)

**Approach:**
- The two inline copies converge onto `SaveLastPoseToStorage` **preserving
  each verbatim** (A: current selection/prev frame/actor-list source/always
  convert; B: current/prev/`vw` source/never convert — the table test pins
  them). Only the B-slot comment is corrected.
- Active-camera state flows through the session core (the `SetActiveCamera`
  mirror write relocates; the radio remains the source of truth).
- The during-run "Allow For Updates on Screen" display blocks (:2746/:2909)
  and all VTK reads/writes stay view-side (L16).
- The pre-existing A-slot re-click behavior (re-running B→A on already-
  converted actors in biplane) is documented in a comment, not changed (R13).

**Patterns to follow:** U3 core; `DecideCameraRadios` headless pin.

**Test scenarios:**
- Happy path: camera A/B switch → active-camera state in session core,
  storage contents match the pinned table rows (A converts, B raw).
- Edge case: A radio re-click (already checked) → behavior documented,
  unchanged.
- Integration: manual-visual — camera A/B switching in monoplane + biplane,
  pose re-placement, during-run display blocks.

**Verification:** headless pins green; manual-visual camera flows unchanged;
`jj diff` relocation.

---

- [ ] U10. **[End-state verification]**

**Goal:** Prove the sweep: MainScreen residue is view-only, line counts
reported, R15 grep-gate holds, full suite + parity green.

**Requirements:** R14, R15; AE5.

**Dependencies:** U1–U9.

**Files:**
- None (verification) — report + docs: `docs/handoff-2026-08-11-vm-layer-extraction.md`
  (mark phase complete), new `docs/solutions/conventions/` compound for the
  shared VM layer (after the phase, per the repo's ce-compound convention)

**Approach:**
- Inventory check: MainScreen residue = `ArrangeMainScreenLayout`, VTK actor
  placement, display radios, key handling, dialogs, thin slot glue.
- Line-count + `ui.`-reference count report vs the 5,115 baseline (target
  ~1,800–2,200; a signal, not a gate).
- **Adapter-thickness report** (the "lands once" property): lines added per
  thinned bridge/slot after thinning, with a per-adapter budget — evidence
  alongside the line-count signal that a new experiment surface is a small
  adapter, not a second implementation (origin success criterion).
- R15 grep-gate: no `QWidget`/`QQuick`/`vtkRenderWindow` includes in
  `src/coordinator` controller sources; domain purity grep unchanged.
- Full gates: `pixi run test` (headless), `ctest -L oracle`, `ctest -L render`
  (xcb), `jtml.qml_parity_check` (IoU ≥ 0.85 band + baseline compare).

**Test expectation:** none — verification-only unit.

**Verification:**
- All suites green; parity re-verified; line counts + residue inventory
  reported; handoff doc updated; compound entry written.

---

## System-Wide Impact

- **Interaction graph:** The 7 (+finished) optimizer binds, selection-handler
  chains (SyncSessionState → SaveLastPose → mirrors), camera radio handlers,
  load-slot chains, and ML chains all re-home onto controllers; by-name
  auto-connects in MainScreen must be re-verified after slot surgery
  (silent-wiring-death class — see learnings).
- **Error propagation:** `messageRequested(title, message, severity)` is the
  single channel; widgets preserves box-type distinctions (critical/info/
  warning), QML ignores severity. The OptimizedFrame out-of-bounds status
  travels on the relay for the widgets box.
- **State lifecycle risks:** thread ownership moves to the run controller
  (destructor contract, ghost threads via pre-Initialize `finished` bind,
  epoch-tagged relay drops); seeds and previous mirrors must reset on
  ClearDataset; mirrors must be consistent before session signals emit.
- **API surface parity:** both apps converge on the same controller APIs; the
  QML bridges keep their Q_PROPERTY shells — QML-side behavior changes only
  where explicitly gated (re-run-while-ghost rejection H1; seed-restore
  revert M10a). Widgets-side deltas are likewise enumerated (Start-gate
  rejection in the thread-death window).
- **Integration coverage:** `jtml.oracle` + `jtml.qml_parity_check` + both
  render smokes are the cross-layer gates; manual-visual checklists scheduled
  per presentation cut.
- **Unchanged invariants:** registry contract (org/app/groups/keys +
  golden 51 entries), widgets selection semantics, drive-sequence quirks,
  pose-file format, camera A/B behavior, `SettingsService` QtCore-only
  purity, `jtml_domain` Qt-free purity, oracle fixtures + tolerances.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Cross-run signal race corrupts runs after the QML unlock semantics move into the shared controller (H1) | Epoch tagging + threadActive Start gate + finished-before-Initialize; fake-driver test pins the race |
| Drive-seam design must not touch OptimizerManager (non-virtual Initialize) | Narrow `OptimizerRunDriver` interface + production adapter; OptimizerManager untouched; headless tests use the fake driver (M12/Q8) |
| Widgets unlock-after-error regression if the state machine makes Error terminal | EnableAll driven by the controller's terminal-frame relay (pinned); test: OptimizerError → terminal frame → unlock |
| Gate rejects QML runs if previous-mirror semantics are misread (H2) | Header-pinned semantics ("last-selected, == current in steady state") + gate-input rule test (`previous == current`) |
| Save-last-pose unification silently corrupts poses (4 divergent behaviors, H4) | Parameterized core + 4-row call-site table test; unification deferred, never part of a relocation cut |
| App-close mid-run thread crash (H3) | Destructor contract (stop → bounded wait → delete) with mid-run destruction test |
| Golden registry fixture drift on extraction (R8) | Re-pin the fixture to call the shared function directly; `%.17g` exact round-trip asserted |
| AUTOMOC undefined symbols / moc signals-section duplicate definitions (new QObject controllers) | Q_OBJECT headers in target sources (GLOB picks headers; explicit `.cpp` list); `signals:` last; known-error playbook in `docs/solutions/build-errors/` |
| QtTest signal affinity (QTBUG-2842) | Relays re-emitted on the controller thread; QSignalSpy observes controller signals, never worker-thread emissions |
| QML render-thread contract violation in the shared builder (R9) | Builder is value-parameterized (never owns VTK state); called only from initializeVTK/dispatch_async on the QML side |
| QML parity drift across cuts | `jtml.qml_parity_check` (IoU 0.993627 baseline) + render smokes at every cut's verification |
| Stale plan-004 line numbers mislead | This plan's inventory is verified against the current tree (see Context & Research) |

---

## Documentation / Operational Notes

- Update `docs/handoff-2026-08-11-vm-layer-extraction.md` → mark the phase
  complete and route to the optimizer-backend session (U10).
- Write the `docs/solutions/conventions/` compound for the shared VM layer
  after the phase (bridge-thinning pattern, epoch/thread contract, mirror
  semantics, call-site table discipline) — per the repo's ce-compound
  convention.
- The decision record style precedent (`src/view/CMakeLists.txt:6` note):
  add a brief note at `src/coordinator/CMakeLists.txt` documenting the
  controller surface once U5 lands.
- No packaging/CPack changes (experimental app is a dev tool); xcb-only
  render environment unchanged.

---

## Sources & References

- **Origin document:** `docs/brainstorms/2026-08-11-shared-vm-layer-extraction-requirements.md`
- Handoff: `docs/handoff-2026-08-11-vm-layer-extraction.md`
- Plan 005 (the bridges + deferred follow-ups): `docs/plans/2026-08-11-005-feat-qml-experimental-frontend-plan.md`
- Plan 004 (prior mainscreen decomposition): `docs/plans/2026-08-10-004-refactor-mainscreen-decomposition-plan.md`
- Parent requirements (R8–R10): `docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md`
- Conventions: `docs/solutions/conventions/jtml-layered-lib-split-2026-08-08.md`,
  `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md`,
  `docs/solutions/conventions/jtml-qml-experimental-frontend-2026-08-11.md`,
  `docs/solutions/conventions/jtml-mainscreen-decomposition-patterns-2026-08-10.md`
- Bugs: `docs/solutions/ui-bugs/jtml-qml-model-pose-sync-queued-functor-never-delivered-2026-08-11.md`,
  `docs/solutions/build-errors/jtml-moc-signals-section-placement-duplicate-definition-2026-08-11.md`
- Code: `src/view/mainscreen.cpp`, `src/app/experimental/*`,
  `include/coordinator/optimizer_manager.h`, `include/domain/session_state.h`,
  `include/services/session_controller.h`, `src/coordinator/optimize_coordinator.cpp`
