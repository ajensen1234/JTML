---
date: 2026-08-10
topic: mainscreen-decomposition
---

# JTML MainScreen Decomposition (View-Model Extraction)

## Problem Frame

`src/view/mainscreen.cpp` is a 5,693-line god object. The MVVM strangle (plans 001–003,
R8 of the origin requirements) pulled the *pure* seams out — `SessionState`,
`OptimizeIntentController`, `ModelListBuilder`, `pose_file_io`, `OptimizeCoordinator` +
`DirectOptimizer` — but the view class itself is undivided. It half-approximates a
view-model without being one, and its role has never been decided. Every remaining
behavior — list interaction, load path, pose editing, edge processing, settings
persistence, camera switching, optimizer binding, and the segment/estimate/DRR block —
is embedded in one class, so touching any single behavior means reading (or risking)
the whole file.

The decomposition is **shape-driven**: the components are dictated by what is actually
in the code, not by a prescribed end-state size. If something does not need to be in
`MainScreen`, it should not be. (Note: the file is large partly because the original
`.ui` file never modularized into sub-widgets; that UI structure is accepted as-is for
this phase — see Scope Boundaries.)

This document is the requirements for the *decomposition phase* of the program-level
refactor. It executes origin R8–R10 and carries origin constraints R9, R12, R15 forward
(origin: `docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md`).

---

## Actors

Carried from the origin document:

- A1. **Developer/researcher (user):** owns the app, runs the GUI for manual-visual gates.
- A2. **Coding agent (pi):** performs the extractions and tests under the user's direction.

---

## Key Flows

- F1. **Decomposition-with-gate** — Trigger: a cluster is identified for extraction.
  Actors: A2, A1. Steps: (1) identify a natural component and its slot boundaries; (2)
  extract it into its target layer (domain/services/coordinator/view-model); (3) gate:
  headless unit tests for logic/service/coordinator extractions, compile + scheduled
  manual-visual check for presentation-only cuts, GPU oracle label for torch/CUDA logic;
  (4) verify the cut against the pre-refactor baseline with `jj diff` — relocation only,
  no behavior delta; (5) proceed to the next cut. Outcome: every extraction is
  individually shippable and gated; no cut changes behavior. **Covered by:** R1, R13, R14.

- F2. **View-model binding** — Trigger: the list-widget swap lands. Actors: A2.
  Steps: (1) `FrameListModel` / `ModelListModel` own item names and selection state; (2)
  `QListView` renders them passively; (3) selection changes propagate model → view →
  VTK re-render through thin MainScreen wiring; (4) `update_image_list_widget` and the
  selection-sync bookkeeping disappear from MainScreen. Outcome: list behavior is
  headless-testable and observable (selection, primary model, current frame). **Covered
  by:** R4, R5.

- F3. **GPU block extraction** — Trigger: the segment/estimate/DRR cluster is extracted.
  Actors: A2, A1. Steps: (1) relocate segmentation (torchscript) and femoral/tibial
  estimation logic byte-identical into `SegmentationController` + `ImplantEstimator`;
  (2) gate with compile + the existing `oracle`/`gpu` ctest label on a GPU machine; (3)
  verify poses within recorded tolerance against the golden baseline; (4) MainScreen
  keeps only the thin slot wiring. Outcome: the largest cluster moves without a headless
  test gap, gated by the GPU tier. **Covered by:** R12, R14.

---

## Requirements

**[Decomposition shape]**
- R1. Decompose `MainScreen` into the natural components present in the code, not toward
  a prescribed size: the eight agreed components — list view-models, settings persistence,
  pose save/load/edit, edge processing, dataset load path, camera switching, optimizer
  binding thinning, and segmentation/estimation — each land as a separate unit with its
  own gate (F1).
- R2. After decomposition, `MainScreen`'s role is decided and stated: **View + composition
  root** — widget wiring, VTK render binding, layout/resize, and the irreducible
  view-only slots (display-mode radios, interaction modes, reset view, key handling).
  Anything that does not need to be there moves out.
- R3. Data shapes stay hybrid: `Frame` AoS (images + edge params + derived images),
  `LocationStorage` pose matrix, `SessionState` scalars. No restructure unless a specific
  seam demands it; the multiple count trackers may remain as-is.

**[View-models]**
- R4. The image and model list widgets become passive `QListView` views over real
  `QAbstractListModel` classes that own item names and selection state; MainScreen's list
  bookkeeping (`update_image_list_widget`, background-highlight removal, selection sync)
  moves into the models or disappears. The models are headless-testable (no widgets).
- R5. `ModelListBuilder` name-dedup logic is reused by the model list model; item names,
  multi-selection, the primary-model rule (first selected row), and current-frame
  semantics match today's behavior exactly.

**[Services/controllers]**
- R6. A `DatasetController` owns the load path (calibration/image/model file parsing,
  frame/model/pose-matrix initialization); `QFileDialog` remains in the view. Headless
  gateable: paths in, populated dataset out.
- R7. An `EdgeProcessor` owns aperture/low-threshold/high-threshold application,
  apply-all, and reset-edge over frames; headless gateable on `cv::Mat` (params + frame
  in, edge/dilated images out).
- R8. The pose save/load/copy/kinematics slots thin to view calls over the existing
  `pose_file_io` seam; copy-previous/next and ambiguous-pose logic move to domain; headless
  gateable (boundary behavior: first/last frame, single selection).
- R9. A `SettingsService` owns QSettings persistence (`LoadSettingsBetweenSessions` /
  `onSaveSettings`); widget-free and headless gateable (save → load round-trip).
- R10. Camera A/B switching state and active-dataset selection move out of the slot body;
  the VTK re-render side stays view.
- R11. Optimizer binding thins to the existing `OptimizeIntentController` +
  `OptimizeCoordinator`; the directive→intent mapping is complete in the controller, and
  `DisableAll`/`EnableAll` remain view-only (widget-level concern).
- R12. A `SegmentationController` + `ImplantEstimator` extract the segment/estimate/DRR
  block (~880 lines: `on_actionSegment_FemHR/TibHR_triggered`, `segmentHelperFunction`,
  `on_actionEstimate_Femoral/Tibial_Implant_s_triggered`, reset-segmentation, DRR
  settings entry). Logic is relocated byte-identical; gated by compile + the GPU oracle
  label (torch/CUDA cannot be headless).

**[Gates and constraints (carried from origin)]**
- R13. No behavior change per cut: each cut is verified by `jj diff` against the
  pre-refactor baseline showing pure relocation, not logic delta (origin R15 — the lesson
  of the render/xcb bug).
- R14. Per-cut gates: logic/service/coordinator extractions get headless unit gates green
  before the next cut; presentation-only cuts get compile + a scheduled manual-visual
  check; GPU-dependent logic gates on the `oracle`/`gpu` ctest label (origin R9).
- R15. No Qt-mocking wrappers and no "instantiate the real `MainScreen`" characterization
  tests (origin R12). Where an extracted component has pure logic with invariants
  (edge-parameter application, pose copy boundaries, settings round-trip, name dedup),
  hegel PBT complements — never replaces — the deterministic unit tests (repo convention,
  `test/HEGEL-PBT-GUIDE.md`).
- R16. Visual verification respects the runtime environment: the app runs under
  `QT_QPA_PLATFORM=xcb` (Wayland/EGL broken), and VTK standalone windows do not work in
  this build — any manual-visual step uses the app under xcb and the `ctest -L render`
  smoke path (`test/oracle/render_smoke.cpp`) where applicable.

---

## Acceptance Examples

- AE1. **Covers R1, R2, R14.** Given the decomposition is complete, `src/view/mainscreen.cpp`
  contains only view wiring, VTK binding, layout/resize, and the listed view-only slots;
  each of the eight components exists in its target layer with its gate green.
- AE2. **Covers R4, R5.** Given the list swap has landed, the two `QAbstractListModel`
  classes are exercised headlessly (names, selection, primary-model rule, current frame)
  with zero widgets; the `QListView` swap compiles and passes a scheduled manual-visual
  check under xcb.
- AE3. **Covers R12, R14.** Given the GPU block extraction, the code compiles and the
  `oracle`/`gpu` ctest label passes on a GPU machine with poses within recorded tolerance
  of the golden baseline.
- AE4. **Covers R13.** Given any single cut, `jj diff` against the pre-refactor baseline
  shows relocation only — no changed arithmetic, control flow, or widget behavior.

---

## Success Criteria

- **Human outcome:** fearless editing — a change to any single behavior (a pose action,
  an edge parameter, a list interaction, a segmentation step) is verifiable by a fast
  headless test or the GPU gate, without reading 5,693 lines or clicking through the GUI;
  `MainScreen` has a decided role rather than an accidental one.
- **Handoff quality:** the plan enumerates units per component with per-cut gates, file
  lists, and test scenarios (including boundary/error/integration cases); no unit requires
  the implementer to invent behavior or scope.

---

## Scope Boundaries

- No data-shape restructure: `Frame` AoS, `LocationStorage` matrix, and `SessionState`
  scalars stay as they are (R3); services are extracted *around* them.
- No modularization of `mainscreen.ui` into sub-widgets — the flat `.ui` structure is
  accepted as-is for this phase; the decomposition targets the C++ class, not the UI
  hierarchy.
- No renaming or re-parenting of `MainScreen`; it remains the `QMainWindow` View +
  composition root (R2).
- No extraction of the display-mode radios, VTK interaction/reset/normal-up binding,
  key handling, or layout/resize code — these stay view-only.
- No behavior change, no new features, and no dialog overhaul (carried from origin).
- No Qt-mocking wrappers and no god-object characterization tests (R15).
- No golden-oracle fixture extension (carried from origin): Kneel_1 remains the regression
  scope unless a cut touches changed-case behavior.
- Not a rewrite: the validated numerical core (DIRECT, cost functions, CUDA) is preserved
  and gated, not silently re-derived (carried from origin).

---

## Key Decisions

- **Shape-driven, not size-driven:** components fall out of what is actually in the code;
  no prescribed end-state line count. `MainScreen` becomes the View + composition root by
  subtraction, and its role is stated rather than approximated.
- **Hybrid data shapes kept:** `Frame` AoS + `LocationStorage` pose matrix + `SessionState`
  scalars remain; extraction happens around them, and a shape change requires a concrete
  seam to demand it.
- **Real view-models:** `QListView` + `QAbstractListModel` replace the `QListWidget`s —
  the lists become passive views over headless-testable models; selection, primary, and
  current-frame become observable. (A plain state holder was the right level for the
  strangle; the view-layer decomposition is where the real models belong.)
- **All eight components in one phase:** one plan, with the GPU-gated segment/estimate
  block ordered last, after the headless-gateable components prove the pattern.
- **Relocation-first extraction:** logic moves byte-identical; the gate proves the move;
  behavior change is prohibited per cut (R13) — the lesson of the render/xcb bug.

---

## Dependencies / Assumptions

- The prior seams exist and are green: `SessionState`, `OptimizeIntentController`,
  `ModelListBuilder`, `pose_file_io`, `OptimizeCoordinator` + `DirectOptimizer` (headless
  suite 24/24, 9 hegel PBT targets).
- `Frame`, `Model`, and `LocationStorage` public APIs are the stable interface the new
  services operate on (R3).
- The torch/torchscript segmentation path is unchanged by the extraction; a GPU machine
  is available for the `oracle`/`gpu` label runs (R12).
- Visual checks run under xcb (`QT_QPA_PLATFORM=xcb`); the VTK-standalone limitation and
  the `ctest -L render` smoke path apply (R16).
- `jj` is the VCS; R13 verification uses `jj diff` against the pre-refactor baseline.

---

## Outstanding Questions

### Resolve Before Planning

None.

### Deferred to Planning

- [Affects R4][Technical] Exact model class names and library placement (jtml_view vs
  jtml_services), and whether the `QListView` swap changes any current-row/scroll/selection
  semantics the current slots depend on.
- [Affects R10][Technical] Whether camera A/B switching shares the `DatasetController` or
  warrants its own controller.
- [Affects R11][Technical] How widget-wide `DisableAll`/`EnableAll` is preserved while
  slot bodies move to services.
- [Affects R9][Technical] Whether `SettingsService` preserves the existing registry key
  names and default-value behavior exactly.
- [Affects R1][Needs research] Unit ordering and phase grouping across the eight
  components (ce-plan decides).

---

## Next Steps

-> /ce-plan for structured implementation planning
