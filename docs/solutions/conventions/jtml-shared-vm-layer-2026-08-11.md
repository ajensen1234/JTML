---
date: 2026-08-11
last_updated: 2026-08-11
module: jtml_coordinator
tags: [view-model, controllers, coordinator, services, mvvm, extraction, plan-006]
problem_type: convention
severity: medium
---

# Shared VM layer (plan 006): the conventions that make the two front-ends one

Plan 006 extracted MainScreen's VM-shaped surface into a shared controller
layer that BOTH the widgets app and the QML experimental app call. MainScreen
(5,115 → 4,987 lines) is now view glue; the QML bridges are thin shells. This
entry records the standing conventions — the next architect touching any of
these seams should read this before changing them.

## The shared layer map (where things live)

| Surface | Location | Nature |
|---|---|---|
| `OptimizerRunController` + `OptimizerRunControllerCore` | `src/coordinator/` | QObject shell + Qt/GPU-free core (gate, 5-state machine, epoch, stage math, seed) |
| `OptimizerRunDriver` (interface) + production adapter | `src/coordinator/` | the swappable-run seam; wraps `OptimizerManager` UNTOUCHED |
| `SessionStateController` | `src/coordinator/` | QObject shell wrapping `jta::SessionState` (diff-emission) |
| `jta::SessionState` (extended: previous mirrors) | `include/domain/` | the one session core both views write through |
| `SaveLastPoseToStorage` + `SavePoseConvertRule` | `src/services/` | parameterized save-last-pose core |
| `StudyLoadController` | `src/services/` | parse→populate→dedup→counts (view updates stay view-side) |
| `MlOrchestrator` | `src/services/` | segment→estimate→SavePose→seed chain, callable-injected |
| `BuildCostFunctionRegistryEntries` | `src/services/cost_function_registry.{h,cpp}` | the one registry mapping (golden-fixture-pinned) |
| `jta::render_pipeline` (free functions) | `src/services/` | the one VTK pipeline recipe (value-parameterized) |

Dependency rule: **services never reference coordinator** (layering). Cross-
layer needs are injected — the run-in-flight probe is a `std::function<bool()>`
wired by composition roots; the ML seed is RETURNED and view-wired.

## The load-bearing conventions

1. **The drive seam, not a fake manager.** `OptimizerManager::Initialize` is
   non-virtual — a test subclass cannot intercept it. Headless tests implement
   the narrow `OptimizerRunDriver` interface and emit the 7+1 signals; the
   production adapter wraps the real manager. This is also the multi-stage
   oracle's future entry point (synthesis item 3).
2. **Epoch-tagged runs + threadActive Start gate.** Every run gets a
   generation counter; relays from stale epochs are dropped; `Start()` rejects
   while a previous thread is alive (covers the Initialize-failure ghost);
   `finished` is bound BEFORE `Initialize`. Without this, run 1's terminal
   frame corrupts run 2 (and `onFinished` waits on run 2's thread).
3. **Destructor contract:** request stop → `quit()` + bounded `wait()` →
   delete; on expiry NEVER delete a running thread — warn + keep waiting
   (bounded DIRECT iterations make expiry a diagnostic).
4. **Previous-mirror semantics (H2):** `previous_frame`/`previous_model_rows`
   mean "last-selected, == current in steady state". They feed save-last-pose
   ONLY — gate input always uses `previous == current`. Do not "fix" this.
5. **Two-phase selection commit (M9):** `UpdateSession` diffs + defers
   `selectionChanged`; `CommitSelection` advances the mirrors and emits with
   the PRE-CHANGE mirrors — preserving the pinned widgets handler order
   (sync → SaveLastPose reads pre-change mirrors → mirrors advance).
6. **Save-last-pose is 4 divergent behaviors (H4).** The widgets canonical
   copy (previous selection/frame, `vw` source, convert iff B), the camera-A
   inline (current selection, actor-list source, ALWAYS convert B→A), the
   camera-B inline (current selection, `vw` source, NEVER convert — its old
   comment was wrong, the raw save is correct), and the QML mirror (current
   frame, scene source). The `SaveLastPoseToStorage` call-site table test
   pins each; **pin, don't unify** — unification is a separately gated cut.
7. **R13 per cut type:** pure moves are literal relocations (`jj diff`); 
   parameterized extractions and QML thinnings are "verbatim-behavior
   extraction with signature adaptation" — gated by the characterization
   test + behavior diff, not a literal `jj diff`.
8. **Error is not a dead-end for the widgets mapper:** widgets unlocks at the
   terminal `OptimizedFrame` relay even with the error bit set; QML keeps
   "stays Error" runState. `messageRequested(title, message, severity)`
   carries box-type distinctions; QML ignores severity.
9. **QML-side policies stay QML-side:** the SingleModelOnly pre-check and the
   Dialog mapping live in `OptimizerBridge`; the shared controller is
   view-agnostic (R15 — headless-drivable under `QCoreApplication`; grep-gate
   enforces no `QWidget`/`QQuick`/`vtkRenderWindow` in coordinator).
10. **Diff-based grouped session emission:** `datasetChanged` /
    `selectionChanged` fire only on actual value change, AFTER mirrors are
    consistent; `ResetForDatasetClear` resets mirrors + seed and emits.

## Known deltas introduced by the phase (deliberate, tested)

- QML re-run-while-ghost rejection (epoch/threadActive gate) — fixes the
  cross-run race.
- QML seed-restore on Initialize failure (M10a) — estimate-wins-over-drift
  survives a failed run.
- Widgets re-run click in the terminal-frame→thread-death window is rejected
  with a message (millisecond window).
- Out-of-bounds terminal frame skips the storage write (old crash path; box +
  relay status preserved).

## Follow-on (do NOT scope into future extractions)

The optimizer-backend workstream (parameters/cost/swappable backend) is a
separate session per the handoff: the multi-stage oracle consumes the
`OptimizerRunDriver` seam; the metric-ablation harness is the precondition
for any "smarter" algorithm change. Measurement-first: no algorithm deltas
before the harness exists (the silhouette-IoU oracle cannot see the
z-weak-axis problem). See plan 006's "Follow-On Optimization Phase" section.

## Related

- Plan: `docs/plans/2026-08-11-006-refactor-shared-vm-layer-extraction-plan.md`
- Requirements: `docs/brainstorms/2026-08-11-shared-vm-layer-extraction-requirements.md`
- Handoff: `docs/handoff-2026-08-11-vm-layer-extraction.md`
- `docs/solutions/conventions/jtml-qml-experimental-frontend-2026-08-11.md`
  (the bridges' original thinness rule + render-thread contract)
- `docs/solutions/conventions/jtml-layered-lib-split-2026-08-08.md` (layer rules)
