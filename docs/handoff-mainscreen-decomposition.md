# Handoff — next phase: MainScreen decomposition (brainstorm → plan)

**Read first:** `AGENTS.md` (build/test/jj conventions + layered layout), then
`docs/plans/2026-08-07-003-refactor-layered-directory-restructure-plan.md`
(U1–U6 all complete) and `docs/solutions/` (searchable learnings).

## Where things stand (all committed, green)

- Plans 001 (U1–U8), 002 (U9–U11), 003 (U1–U6) are **complete**: layered libs
  `jtml_domain/services/coordinator/view` STATIC + `jtml_compute` SHARED + thin
  `src/app` composition root; headless suite **24/24**; 9 hegel PBT targets;
  render path verified (runs under `xcb`).
- Rendering bug resolved + compounded: the app must run under
  `QT_QPA_PLATFORM=xcb` (Wayland/EGL broken on this box — `pixi run run` now
  defaults it); `ctest -L render` (`test/oracle/render_smoke.cpp`) is the
  regression test; VTK standalone windows don't work in this build (probe
  documents it). See `docs/solutions/tooling-decisions/jtml-rendering-runtime-xcb-qvtk-2026-08-10.md`.
- Head: clean `wip` change; jj workflow = `jj describe -m "<scope>: <msg>"` then `jj new`.

## The task: brainstorm then plan the MainScreen decomposition

`src/view/mainscreen.cpp` is still a ~5.7k-line god object. The MVVM strangle
pulled logic OUT into pure seams, but the **view class itself is not decomposed**.
This phase is a fresh **ce-brainstorm → ce-plan** (fresh session recommended) to
design it properly.

### Context the brainstorm should start from
- The pure seams already extracted (headless-tested): `SessionState` (app state),
  `OptimizeIntentController` (optimize decision+packaging), `ModelListBuilder`
  (name dedup), `pose_file_io` (persistence), `OptimizeCoordinator` +
  `DirectOptimizer` (lifecycle/optimization). These are the *model/view-model/
  service* direction already in place.
- The remaining god-object content in `mainscreen.cpp`: widget slots + VTK render
  binding + a large chunk of orchestration + list/view-model wiring.

### Open questions to explore (deliberately open — filenames/classes so far are "stabs in the dark")
1. **Data shapes**: array-of-structures vs structure-of-arrays for frames/models/
   poses (the app stores `loaded_frames`, `model_locations_` matrix, session state).
   Would SoA open better seams?
2. **Real view-model classes**: should the list widgets get actual Qt view-model
   classes (QAbstractListModel) behind `image_list_widget`/`model_list_widget`,
   or is a plain state holder (SessionState) still the right level? What does the
   UI actually need (selection, primary, current frame, display-mode)?
3. **Slot-by-slot decomposition targets**: which mainscreen slots cluster into
   coherent services/controllers (e.g. the segment/estimate/DRR block ~lines
   1700–2450 explicitly deferred in plan 002; the load-path slots; the pose
   save/load/edit slots; the optimize entry already controlled)?
4. **What stays view-only**: the irreducible widget+VTK binding vs what moves.
5. **Boundaries/constraints**: R15 (no behavior change per cut — use `jj diff`
   against the pre-refactor baseline to prove it, as the render bug taught us),
   per-cut headless gates, PBT complements deterministic tests, no Qt-mocking
   wrappers / no god-object characterization (R12).

### Deliverables of the phase
- `ce-brainstorm` → a right-sized requirements doc (R-IDs) with the agreed
  decomposition shape.
- `ce-plan` → implementation units with per-cut gates (mirror plans 001–003:
  U-IDs, Files, Approach, Test scenarios, Verification, Execution note).
- Remember: the render/xcb environment caveat and the VTK standalone limitation
  apply to any visual verification step in the plan.

## Key pointers
- `docs/solutions/conventions/jtml-layered-lib-split-2026-08-08.md` — lib/layer conventions.
- `docs/solutions/logic-errors/cost-function-parameter-double-truncation-2026-08-08.md` — PBT-found-bug pattern.
- `docs/solutions/tooling-decisions/jtml-rendering-runtime-xcb-qvtk-2026-08-10.md` — rendering runtime.
- `test/HEGEL-PBT-GUIDE.md` — PBT authoring (9 targets already exist).
