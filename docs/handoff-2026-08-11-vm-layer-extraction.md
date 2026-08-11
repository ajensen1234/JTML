# Handoff — next phase: shared VM-layer extraction (brainstorm → plan)

> **PHASE COMPLETE (2026-08-11).** Executed by
> `docs/plans/2026-08-11-006-refactor-shared-vm-layer-extraction-plan.md`
> (U1–U10, all landed): MainScreen 5,115 → 4,987 lines of view-only residue;
> the shared layer (OptimizerRunController + drive seam, SessionStateController,
> SaveLastPoseToStorage, StudyLoadController, MlOrchestrator, registry mapping,
> render-pipeline builder) is live under both front-ends; headless 46/46,
> oracle + qml_parity_check (IoU 0.993627) + render smokes green; `probe_vtk`
> GLX failure is pre-existing/environmental. Conventions:
> `docs/solutions/conventions/jtml-shared-vm-layer-2026-08-11.md`. Manual-visual
> checklists (widgets run directives, selection flows, camera A/B, ML) remain
> for a human GPU session. NEXT: the Follow-On Optimization Phase (plan 006's
> Scope Boundaries) — deferred quirk cuts, then synthesis items 1 → 3 → 2 → 5
> → 8; the multi-stage oracle consumes the `OptimizerRunDriver` seam.

**Read first:** `AGENTS.md` (build/test/jj conventions + layered layout), then
`docs/plans/2026-08-11-005-feat-qml-experimental-frontend-plan.md` (all 9 units
landed), `docs/solutions/conventions/jtml-qml-experimental-frontend-2026-08-11.md`
(the QML compound — especially "The view-model layer (the architecture lesson)"
and "Model pose sync-back"), `docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md`
(R8–R10: the full-MVVM commitment that this phase executes), and the panoptes
synthesis `.panoptes/jtml-research-horizons/synthesis.org` (items 1–3, 8).

## Where things stand (all committed, green)

- Plan 005 built a **second front end** — `jtml_experimental`, a QML app in
  `src/app/experimental/` — linking the same widget-free backend
  (domain/services/coordinator/compute). It drives the REAL `OptimizerManager`,
  reuses the list models, and its **bridges** (`StudyBridge`/`SettingsBridge`/
  `OptimizerBridge`/`MlBridge`/`PoseBridge` behind an `AppBridge` hub) ARE a
  working view-model layer — the "VM" that the widgets app never had.
- The widgets app (`src/view/mainscreen.cpp`, ~5,115 lines) is untouched and
  still fuses View + ViewModel in its slots.
- Gates: 37/37 headless, `jtml.oracle` + `qml_parity_check` (IoU 0.993627,
  digit-identical to baseline) + render smokes green. QML decision record at
  `src/view/CMakeLists.txt:6` (full QML front-end deferred; experimental app
  sanctioned).

## The task: brainstorm → plan the shared VM-layer extraction

The user's framing: *"We've done a lot of work here creating a GUI that can
communicate with a lot of the core engine parts — that gives us room to start
looking at the surface areas we can meaningfully extract from `mainscreen.cpp`
so that we can really get a good groove going and have only the minimal set of
elements needed as basic glue in place. We can worry about the modularization
of the optimization protocol later."*

So: **determine the layer of abstraction for the VM-type things** — the shared
controllers/view-models that BOTH the widgets app and the QML app call, so
MainScreen shrinks to minimal glue. The optimizer-backend modularity is
explicitly deferred.

## Context the brainstorm should start from

- **The bridges are the spec.** The QML bridge APIs (selection contract,
  run-state machine, settings dirty tracking, applyViewerPose, the pose-sync
  chain) are precisely what a front-end-agnostic controller must expose. The
  brainstorm's job is to lift these into a shared layer with the RIGHT
  abstraction (coordinator-layer QObjects? services? a new `controllers/`
  surface?) without the QML-specific shapes leaking into it.
- **The duplication residue (R13-forced, now removable):**
  1. `OptimizerBridge`'s drive sequence vs `MainScreen::LaunchOptimizer`
     (thread + manager + 7 connects) — the biggest mirror; a shared
     `OptimizerRunController` serves both (same seam as the multi-stage oracle,
     synthesis item 3 — though the oracle work itself can stay deferred).
  2. `BuildCostFunctionRegistryEntries` replicated in the QML app (golden-
     fixture pinned) — extract into `SettingsService`.
  3. `matToVTK` + the render-pipeline construction duplicated in
     `QmlVtkRenderer` — extract a widget-free pipeline builder both `Viewer`
     and the QML renderer use.
- **What MainScreen still holds that is VM-shaped** (candidates to extract):
  `LaunchOptimizer` + the 7 signal binds, `DisableAll/EnableAll` (run-state
  locking), `SyncSessionState` + `previous_frame_index_` bookkeeping, the four
  save-last-pose copies (R13-preserved; unification is a separate gated cut),
  the load-path orchestration (now duplicated in `StudyBridge`), the camera
  A/B slot orchestration, the segment/estimate slot orchestration (mirrored in
  `MlBridge`). What is genuinely view-only: `ArrangeMainScreenLayout` (~650
  lines), VTK actor placement, display-mode radios, key handling, dialogs.
- **Constraint reality:** R13 (no behavior delta per cut — `jj diff` shows
  relocation), R9/R14 per-cut gates (headless for logic; compile + scheduled
  manual-visual for presentation), the layer purity gates (`jtml_domain` is
  Qt-free — controllers that need QObject signals live in coordinator or
  services), no Qt-mocking, PBT complements deterministic tests. The widgets
  app's pinned behaviors (selection semantics, registry contract, optimizer
  drive sequence) are the spec — extraction must be byte-identical relocation
  first, then the QML bridges thin onto the shared layer.

## Open questions to explore (deliberately open)

1. **Where does the shared VM layer live?** A new `controllers/` slice of
   coordinator? Existing services? What's the naming convention (the
   `*Controller` vs `*Service` vs `*Bridge` split)?
2. **QObject vs plain:** the run-state machine and the 7 signal binds need
   QObject signals; selection state is pure. What's the right split per
   controller (e.g., a pure `SessionController` core + a thin QObject shell)?
3. **The selection contract:** the QML app deliberately uses delegate-based
   selection (no `QItemSelectionModel`); the widgets app pins
   `QItemSelectionModel` semantics. Can one shared selection/state object serve
   both, or do the two views keep separate selection adapters over a shared
   state core?
4. **Which extraction order:** what's the highest-leverage first cut (the
   run controller? the registry mapping? the pipeline builder? the pose
   sync/state chain)?
5. **MainScreen's end-state:** with the VM layer extracted, what does the
   "minimal glue" residue look like — line-count target and what stays
   forever-view?

## Deliverables of the phase

- `ce-brainstorm` → a right-sized requirements doc (R-IDs) with the agreed VM
  layer shape and the extraction surface inventory.
- `ce-plan` → implementation units with per-cut gates (U-IDs, Files, Approach,
  Test scenarios, Verification), mirroring plans 001–005.
- The optimizer-backend modularity (parameters + cost + swappable backend) is
  a LATER session — do not scope it here.

## Key pointers

- `docs/solutions/conventions/jtml-qml-experimental-frontend-2026-08-11.md` — the VM-layer lesson + bridge inventory.
- `docs/solutions/ui-bugs/jtml-qml-model-pose-sync-queued-functor-never-delivered-2026-08-11.md` — the pose-sync chain (a concrete round-trip pattern to generalize).
- `docs/plans/2026-08-10-004-refactor-mainscreen-decomposition-plan.md` — what plan 004 extracted and what it deliberately left in the view.
- `docs/solutions/conventions/jtml-layered-lib-split-2026-08-08.md` — layer rules for where the controllers land.
- `.panoptes/jtml-research-horizons/synthesis.org` — items 1–3, 8 (bug fixes, ablation harness, multi-stage oracle, compute perf) are the algorithm-side roadmap; the VM extraction is the architecture-side companion.
- `docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md` — R8's full-MVVM commitment (this phase executes it).
