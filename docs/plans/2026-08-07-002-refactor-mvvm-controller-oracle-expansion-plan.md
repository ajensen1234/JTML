---
title: refactor: Complete U7 MVVM — optimize-intent controller, MainScreen shrink, oracle expansion
type: refactor
status: complete
date: 2026-08-07
origin: docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md
deepened: 2026-08-07
supersedes_unit: U7 (phase-1 done; this plan carries the re-scoped follow-on work)
---

# Complete U7 MVVM — Optimize-Intent Controller, MainScreen Shrink, Oracle Expansion

## Overview

The parent plan (`2026-08-07-001-refactor-testability-mvvm-plan.md`) marked U1–U8 done and
re-scoped the *full* MVVM decomposition of `MainScreen` as follow-on work. This sub-plan
lands that follow-on in three gated slices, continuing the strangle that already shipped
`pose_file_io` and `SessionState`:

1. **U9 — Optimize-intent controller**: extract the widget-free "can I optimize / what do I
   need" decision and the `Initialize` argument packaging out of `MainScreen::LaunchOptimizer`
   into a headless-tested controller (origin AE4).
2. **U10 — MainScreen shrink (R10)**: move list-population name logic and the pure
   load-time checks (model name-dedup, frame size-consistency) out of MainScreen, so its
   line count and `ui.`-reference count move measurably down and the view keeps only
   widget/render binding.
3. **U11 — Tier-2 oracle expansion**: extend the GPU oracle beyond frame 0 / single stage /
   budget 3000 so it exercises the production `Optimize()` orchestration (cumulative
   trunk→branch→leaf budget via `SetCallOffset`, multi-frame) end-to-end, closing the
   documented coverage gap.

Each slice is individually shippable and gated (R9): logic/service/controller extractions
get headless unit gates; presentation-only cuts get compile + scheduled manual-visual check.
No behavior change to the optimizer path (R15).

---

## Problem Frame

`MainScreen` is a 5688-line god object holding app-state, command orchestration, services,
and widget/render binding in one file. The refactor makes each seam independently verifiable
headless so a change to an algorithm or a button path is caught by a fast pass/fail rather
than by launching the app and clicking.

The two remaining *extractable* seams of substance are (a) the optimize-intent decision +
`OptimizerManager` argument packaging that lives inline in `LaunchOptimizer`, and (b) the
load-path logic (name dedup, size checks) plus list-population name building. Both are pure
enough to unit-test headless, and both directly reduce MainScreen's `ui.`-reference and line
counts (R10/AE4). The oracle's frame-0-only limitation is the last documented coverage gap.

The production optimizer must **not** be rewired to the stub `OptimizeCoordinator`
(R15 — behavior preservation). The controller owns decision + packaging; `MainScreen` and
`OptimizerManager` keep the real GPU binding. The `OptimizeCoordinator` (test seam) is
intentionally unchanged here.

---

## Requirements Trace

- R8. Decompose MainScreen: presentation (widgets/slots/VTK binding) separated from
  app-state/command orchestration and from services.
- R9. Each extraction lands with its test gate green; logic/service/coordinator → headless
  unit gate; presentation-only → compile + scheduled manual-visual.
- R10. MainScreen's line count and `ui.`-reference count move measurably down.
- R12. No Qt-mocking wrappers; no god-object characterization tests.
- R15. Preserve the cumulative budget and optimizer behavior; no re-derivation.

**Origin actors:** A1 (engineer / fearless editor), A2 (implementing agent).
**Origin flows:** F3 (extraction-with-gate).
**Origin acceptance examples:** AE4 (covers R8, R9, R10 — coordinator/services out of
MainScreen; headless gates; coupling metrics moved in the tracked direction).

---

## Scope Boundaries

- The production optimizer loop is **not** rewired to `OptimizeCoordinator`; that stub
  remains a test-only seam (R15).
- No behavior change: recovered poses, budget semantics (cumulative 20k/25k/30k), and the
  render/color/opacity behavior are unchanged.
- The oracle stays appearance-based (silhouette/IoU), gated only under the `oracle` label.
- No new JSON/config format; no persistence schema changes.
- Biplane oracle is written narrowly: exercise the cumulative multi-stage + multi-frame
  path; a full biplane (camera-B) gate is deferred (see below).

### Deferred to Follow-Up Work

- **Biplane GPU oracle with appearance gate**: `Kneel_1` is monoplane and `Subject0` (the
  real biplane fixture) has no `Labels/` directory, so a symmetric IoU gate is not directly
  available. Deferred to a dedicated effort that resolves a ground-truth for camera B.
- **Migrating `OptimizerManager`'s cross-thread `Qt::DirectConnection` stop**: the coordinator
  already owns cooperative-stop; relocating production threading is out of scope here and
  gated by R15.
- **Segment / estimate / DRR slot decomposition** (lines ~1700–2450): further strangle after
  U10, not part of this plan.

---

## Context & Research

### Relevant Code and Patterns

- `src/gui/mainscreen.cpp:4491` — `LaunchOptimizer` (inline validation + arg packaging).
- `include/core/session_state.h` / `src/core/session_state.cpp` — pure app-state holder; the
  pattern the controller follows (widget-free, headless-tested).
- `include/core/pose_file_io.h/.cpp` — pure persistence service; the strangle pattern.
- `src/core/optimizer_manager.cpp:883` (`Optimize`), `:1225` (`RunDirectStage`) — the
  production multi-stage + biplane drive that U11's oracle must exercise.
- `test/oracle/oracle_test.cpp` — current frame-0 / single-stage / budget-3000 oracle.
- `test/unit/test_session_state.cpp` — the headless-unit registration pattern to mirror.
- `test/CMakeLists.txt` — how pure unit targets compile CUDA/Qt-free from selected `.cpp`.

### Institutional Learnings

- `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md` —
  per-layer gate (R9), AUTOMOC gotcha, `file(GLOB)` explicit-list trap, cumulative budget
  load-bearing, single persistent worker thread, no god-object characterization.
- `docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md` —
  oracle label vertical flip (`cv::flip(label,0)`), empirical frame↔label correspondence
  (not filename-aligned), oracle runs from repo root (`WORKING_DIRECTORY`), conda
  `BUILD_RPATH` override via `-Wl,-rpath`, hegel is a swappable FetchContent layer.

### External References

- None required; the codebase + compounded learnings are sufficient for this refactor
  (strong local patterns exist — skip external research).

---

## Key Technical Decisions

- **Controller owns decision + packaging, not threading.** `OptimizeIntentController` returns
  a typed `result` (can-run / failure reason) and, when runnable, the prepared
  `Initialize`-compatible payload. `MainScreen::LaunchOptimizer` stays the thin caller that
  binds the real `OptimizerManager` thread/GPU (R15).
- **Controller is pure (no Qt, no widgets).** It takes plain values (model count, frame
  count, selection rows, current frame, `model_locations_` dims, directive) rather than
  widgets, mirroring `SessionState`. This makes it trivially headless-testable.
- **Reads route through `session_state_`.** Where `LaunchOptimizer` still reads
  `selectionModel()->selectedRows()`, `ui.image_list_widget->currentIndex().row()`, or
  `ui.model_list_widget->currentIndex().row()` directly, it now reads `session_state_`
  (`GetSelectedModels`, `GetCurrentFrame`, `GetModelCount`). This closes the missed seam the
  previous slice left open.
- **List-population is split: pure name logic in, `addItem` + VTK stays.** Model-name
  dedup and biplane `"A:…\nB:…"` label composition become pure builders; the view keeps only
  `addItem(...)` and render binding (compile + manual-visual gate, R9).
- **Oracle drives a shared stage-loop routine, not a hand-rolled reproduction.** The
  trunk→branch→leaf loop with cumulative budget (`SetCallOffset` + `budget_ +=`) is extracted
  into a small routine that **both** `OptimizerManager::Optimize()` and the oracle call. This is
  the only design satisfying R2 (the gate asserts against shared production code, never its own
  reproduction) and it is *not* R12 premature abstraction — the routine has two real consumers.
  Production behavior is unchanged (R15): `Optimize()` calls the same helper it previously ran
  inline. Keep the appearance/IoU gate and a documented reduced budget for
  runtime.
- **Anti-circularity (R2):** the controller/builder unit tests assert against independent
  truth (validation truth tables, expected name/dedup strings) — never by re-deriving
  production logic or instantiating real `MainScreen` (R12).

---

## Open Questions

### Resolved During Planning

- Does AE4 mean rewiring production to `OptimizeCoordinator`? **No** — R15 preserves
  behavior; the coordinator stays a test seam. The controller extracts decision + packaging
  only.
- **U11 fork resolved:** the oracle drives a *shared* stage-loop routine that production
  `OptimizerManager::Optimize()` also calls (not a hand-rolled reproduction, not the stub
  `OptimizeCoordinator`). This satisfies R2 non-circularity (the gate asserts against shared
  production code) and is not R12 premature abstraction (two real consumers: production + oracle).
- Oracle multi-stage under cumulative budget is runtime-expensive. **Mitigated** by keeping a
  documented reduced budget (e.g. trunk/branch/leaf chained, reduced) — the gate's purpose here
  is exercising the production orchestration path, not re-proving the cumulative cap
  (Tier-1 covers the cap).
- `OptimizerManager` leaks `optimizer_thread`/`optimizer_manager` on init failure. **Accepted**
  as a pre-existing latent issue surfaced for a note; not fixed in this behavior-preserving
  refactor (would be a separate change).

### Deferred to Implementation

- Exact `OptimizeIntentController` method names / returned struct shape — resolved against
  real `LaunchOptimizer` code during implementation.
- **U11 shared stage-loop routing + budget reconciliation:** whether the shared stage-loop
  helper lives inside `OptimizerManager` (a private helper both `Optimize()` and the oracle
  call) or as a sibling free function, and the authoritative cumulative caps — reconcile the
  20k/25k/30k (plan) vs `[10000,20000,30000]` (`baseline.json`) vs `trunk 20000 / 2×branch
  5000 / leaf 5000` (`settings_constants.h`) discrepancy first, recording the resolved series
  in `golden_oracle.org`.
- **Cross-frame carryover choice:** whether the oracle reproduces production's frame N-1→N
  optimum chaining or treats frames independently; decide during implementation and align the
  U11 "end-to-end" claim accordingly.
- Whether to fold the controller into `jtml_core` or a new `include/core/` unit — follow the
  `session_state` precedent (new header + `.cpp` in `src/core/CMakeLists.txt` explicit list).
- **U10 dedup quirk:** the real dedup loop resets the scan index and for three identical names
  yields `["A(2)","A(3)","A"]` (third unsuffixed). Before extraction, characterize these
  multi-duplicate / cross-set-collision outputs and either (a) reproduce them exactly in the
  builder with tests pinning each quirk, or (b) get an explicit decision that it's a fixable
  display bug with a documented behavior change — do not normalize silently (R9/R12/R15).

---

## Implementation Units

- [x] U9. **Optimize-intent controller (headless AE4 gate)**  — landed (`tsrvnltv`); headless `jtml.optimize_intent` green

**Goal:** Extract the widget-free "can I optimize / what do I need" decision and the
`OptimizerManager::Initialize` argument packaging out of `MainScreen::LaunchOptimizer` into a
headless-tested controller.

**Requirements:** R8, R9, R10, AE4.

**Dependencies:** SessionState (landed), U10 gates build on this.

**Files:**
- Create: `include/core/optimize_intent_controller.h`
- Create: `src/core/optimize_intent_controller.cpp`
- Modify: `src/core/CMakeLists.txt` (add the new `.cpp` to the explicit list — AUTOMOC trap)
- Modify: `src/gui/mainscreen.cpp`, `include/gui/mainscreen.h` (route LoadOptimizer reads
  through `session_state_`; delegate the predicate + packaging to the controller)
- Test: `test/unit/test_optimize_intent_controller.cpp`, registered in `test/CMakeLists.txt`
  as a `headless` Catch2 target compiled CUDA/Qt-free from the new `.cpp` + `session_state.cpp`

**Approach:**
- The controller takes plain inputs: model count, frame count, selected model rows, current
  frame, `previous_frame_index_` (the last-viewed frame), `model_locations_` frame/model dims,
  and the directive. It returns a typed result: runnable
  (`{ok=true, directive, primary_model_index, current_frame}`) or a failure reason
  (no-frame/model-selected, frame/model out of range, current-frame-differs-from-last-viewed,
  pose-matrix dimension mismatch).
- **The full predicate truth must live in ONE place.** The current predicate also gates on
  `current frame != previous_frame_index_` (the "view changed since last sync" guard). Capture
  `previous_frame_index_` as a controller input so the decision isn't silently split between
  the controller and `MainScreen`. On the common path `previous_frame_index_ ==
  session_state_.GetCurrentFrame()` (set by the selection-changed slot), so behavior is
  preserved (R15).
- `LaunchOptimizer` builds these inputs from `session_state_` (+ `loaded_*` sizes +
  `model_locations_`), calls the controller, pops the mapped error box on failure, and on
  success assembles `OptimizerManager::Initialize(...)` from the returned intent — **keeping
  the identical thread/GPU/manager wiring**.
- Do not relocate threading or change the `Initialize` payload types (R15).

**Patterns to follow:** `session_state` (pure, widget-free, headless-tested); `pose_file_io`
(strangle pattern); the gold unit-test registration in `test/CMakeLists.txt`.

**Test scenarios:**
- Happy path: all preconditions satisfied for each directive ("Single"/"All"/"Each"/"From"/
  "Backward"/"Sym_Trap") → `{ok=true}` with the correct primary model index and current frame.
  (`Covers AE4` intent surface.)
- Happy path: current frame equals `previous_frame_index_` → `{ok=true}` (the common
  selection-change path, mirroring production).
- Edge case: current frame differs from `previous_frame_index_` → typed
  "current-frame-changed" failure reason (preserves the existing guard).
- Edge case: empty selection → typed "select frame and model first" reason (not a crash).
- Edge case: no current frame (`current_frame < 0`) and frame==model-count=0 → failure reason.
- Edge case: `current_frame >= frame_count` and `primary >= model_count` → failure reasons.
- Error path: `model_locations_` frame/model dims mismatch `loaded_*` sizes → "pose dimension
  matrix differs" reason.
- Integration: after the controller returns `ok`, `LaunchOptimizer` assembles an
  `Initialize` call whose primary-model index equals `session_state_.GetPrimaryModelIndex()`.

**Verification:** `test/unit/test_optimize_intent_controller.cpp` passes headless and is
registered in the default suite; MainScreen compiles with the controller path; `pixi run
build` + `pixi run test` green; `ctest -L oracle` green; GUI smoke (offscreen + real) runs the
optimize entry to a valid precondition.

---

- [x] U10. **MainScreen shrink — list population + load-logic extraction (R10)**  — landed (`ynnkwmwl`); `jtml.model_list_builder`(+`_props`) green; mainscreen.cpp 5693 lines

**Goal:** Move the *real, non-trivial* pure load logic (model name-dedup, frame same-size
consistency) and the list-population display-name construction out of MainScreen, so
`src/gui/mainscreen.cpp`'s line count moves down and the load slots keep only widget + render
binding. The `ui.`-reference count is expected to move only marginally (single digits) —
this cut is primarily a **line-count** reduction engineered for maintainability, not a `ui.`
reduction (the two load slots' `ui.` refs are mostly widget bindings that stay in the view).

**Requirements:** R8, R9, R10, AE4.

**Dependencies:** None (independent of U9; but land after it per dependency ordering so each
is a clean shippable cut).

**Files:**
- Create: `include/core/model_list_builder.h` and `src/core/model_list_builder.cpp` (model
  name-dedup + frame display-label composition)
- Modify: `src/core/CMakeLists.txt` (add the new `.cpp` to the explicit list)
- Modify: `src/gui/mainscreen.cpp` (replace inline `addItem` name-building + inline dedup and
  size-check logic with calls to the builders; keep `addItem` and VTK binding in the view)
- Test: `test/unit/test_model_list_builder.cpp`, registered `headless` Catch2, CUDA/Qt-free

**Approach:**
- Extract the pure model-name uniquification (dedup within new set via `(N)` suffix, then vs
  existing names) into a `ModelListBuilder` function that returns the ordered display names
  as `std::vector<std::string>` (genuinely Qt-free, mirroring `session_state`/`pose_file_io`).
- Extract frame display-label composition (monoplane = baseName; biplane = `"A: …\nB: …"`)
  into the same service as a `std::string`-returning function.
- Extract the frame same-size consistency predicate (frame vs `loaded_frames` edges) into a
  pure helper. Keep the QErrorMessage-driven *caller* logic in the view.
- `MainScreen::on_load_model_button_clicked` / `on_load_image_button_clicked` call the
  builders, then do `ui.*_list_widget->addItem(...)` and the existing VTK/color binding
  (which stays in the view). This removes the inline loops and name-building from the view.
- **Scope note (R12):** only the real logic moves — name-dedup and the same-size predicate
  are headless-tested; the one-line label compositions are folded into the builder purely to
  keep naming in one place, not as premature abstraction.

**Patterns to follow:** `pose_file_io` / `session_state` service shape; `test/CMakeLists.txt`
pure-unit registration.

**Test scenarios:**
- Happy path: model-name dedup returns expected ordered names for a fresh set with no
  collisions.
- Edge case: a duplicate name within the new set gets a `(2)`/`(3)` suffix.
- Edge case: a name colliding with an existing loaded model gets a `(2)` suffix.
- Happy path: biplane label composition yields `"A: <base>\nB: <base>"`; monoplane yields the
  baseName.
- Happy path: same-size predicate accepts equal-size frames and rejects a mismatched one.
- Integration: the load slot calls the builder, then `addItem`s exactly the returned names in
  order (`ui.`-reference count drops; rendering unchanged).

**Verification:** builder tests pass headless and are in the default suite; MainScreen
compiles; the **line count** of `src/gui/mainscreen.cpp` is lower than the pre-cut baseline
and the two load slots no longer contain the inline dedup / size-check / label-name blocks
(`ui.`-reference count may stay within single digits of baseline — this is a line-count cut);
`pixi run test` 8+ green; `ctest -L oracle` green; scheduled manual-visual check of the load
buttons (list population names unchanged).

---

- [x] U11. **Tier-2 oracle expansion — multi-frame + cumulative multi-stage (covers the
  documented gap)**

**Goal:** Extend the GPU oracle so it exercises the production `Optimize()` orchestration —
multi-frame across Kneel_1's 3 frames and the cumulative trunk→branch→leaf budget via
`SetCallOffset` — rather than a single hand-rolled `DirectOptimizer` on frame 0 / budget 3000.

**Requirements:** R2 (independent ground truth), R15 (cumulative budget path exercised),
AE2 (golden gate).

**Dependencies:** None (independent of U9/U10 — the oracle does not consume the controller or
builder; its target and registration already exist; ordering is benign).

**Files:**
- Modify: `test/oracle/oracle_test.cpp` (multi-frame + multi-stage drive; appearance gate per
  frame)
- Modify: `test/golden/baseline.json` / `golden_oracle.org` (record per-frame budget and gate
  values if changed)
- Modify: `test/CMakeLists.txt` (extend the `jtml.oracle` target / add a second oracle case
  if a separate target is cleaner)
- Test: the oracle itself is the gate (`ctest -L oracle`); assert per-frame IoU > 0.85
- Related (read-only reference): `src/core/optimizer_manager.cpp` (`Optimize`, `RunDirectStage`)
  as the stage-sequence source of truth

**Approach:**
- **Commit to a shared stage-loop routine (resolves the R2 circularity fork).** The oracle and
  production `OptimizerManager::Optimize()` both call the *same* stage-loop routine (trunk→
  branch→leaf with cumulative budget via `SetCallOffset` and `starting_point_ = current optimum`).
  This is the single, unconditional decision — no "cheap hand-rolled reproduction" fork remains.
  Because the routine is shared, the oracle asserts against production code, so budget-semantics
  drift or a changed stage count fails the gate (anti-circularity R2); a dedicated shared routine
  (with two consumers) is not R12 premature abstraction. See also the first
  `Resolved During Planning` entry below.
- **Reconcile the budget numbers first (blocking prerequisite for U11).** The cumulative caps
  appear as 20k/25k/30k in the plan, `[10000,20000,30000]` in `baseline.json`
  (`budget_effective_cumulative`), and `trunk 20000 / 2×branch 5000 / leaf 5000 = 35000` in
  `settings_constants.h`. Before wiring the gate, pin the single source of truth from
  `settings_constants.h` and record the resolved series in `golden_oracle.org` — U11's replay
  caps and the R15 preservation claim must reference the same resolved numbers (see U11
  `Deferred to Implementation`).
- **De-risk frame 2/3 first:** before building the full multi-frame gate, run a low-cost
  recoverability spike on frames 1 and 2 (`1024/2807-2808.tif`) using their `fem.jts`
  per-frame start poses, confirming each can recover an IoU > 0.85 silhouette. Only proceed
  to wire the permanent gate once frames 2/3 are confirmed recoverable (avoids shipping a
  flaky gate the team stops trusting).
- Loop over Kneel_1 frames `1024/2806-2808.tif`. **Correspondence must be pinned per frame**
  using each frame's **own** `fem.jts` start pose from `baseline.json`
  (`expected_pose_per_frame`), rendered to pick the max-IoU label — not the frame-0 pose
  reused across all frames (frames 2/3 have materially different poses in `baseline.json`).
- **Cross-frame start handoff:** state explicitly whether the shared stage-loop reproduces
  production's cross-frame carryover (frame N>0 starts from frame N-1 optimum), and pin it in
  the stage-loop helper so the oracle and production chain frames identically. If the shared
  routine owns the per-frame `starting_point_` assignment, this is covered structurally; if
  the oracle treats frames independently, re-scope the claim to "per-frame recoverability"
  (see `Deferred to Implementation`).
- Drive the production trunk→branch→leaf sequence with cumulative budget semantics: `budget_`
  accumulates (trunk+)+branch+leaf, `SetCallOffset(cost_function_calls_)` carries the count
  across stages, `starting_point_ = current optimum` at stage boundaries, with branch/leaf
  `DIRECT_DILATION` managers (dilation 6/3/1, ranges 20/3, leaf z opened) mirroring
  `OptimizerManager::Optimize()` / `RunDirectStage`.
- Keep the appearance (rendered-silhouette IoU > 0.85) gate per frame; keep `cv::flip(label,
  0)` and `WORKING_DIRECTORY=repo root`.
- Use a documented reduced budget (e.g. trunk ~ as today, then chained branch/leaf) so runtime
  stays bounded; record the effective caps in `baseline.json`/`golden_oracle.org`. Estimate and
  record the expected end-to-end runtime, and raise the `jtml.oracle` ctest `TIMEOUT` (current
  3600s) to a safe margin above it, so the expanded multi-frame run cannot hard-timeout on the
  RTX 3090.
- Because the oracle drives the *shared* stage-loop routine (not a reproduction), the
  appearance IoU gate and the budget-accumulation/`SetCallOffset`/`starting_point_` invariants
together guard production from drift (R2/R15) — the gate asserts against the same code
production runs.
- Preferably drive a shared stage-loop helper (or `OptimizerManager`) so the oracle cannot
  drift from production (anti-circularity R2). Decide during implementation; the oracle must
  at minimum reproduce the production stage sequence exactly.

**Patterns to follow:** existing `oracle_test.cpp` pipeline setup; `golden_oracle.org`
spec; the tooling recipe in `docs/solutions/tooling-decisions/...` (flip, empirical
correspondence, WORKING_DIRECTORY, rpath).

**Test scenarios:**
- Happy path: each of the 3 Kneel_1 frames recovers a pose whose rendered silhouette IoU >
  0.85 vs its empirically-pinned *per-frame* label (correspondence resolved from that frame's
  own `fem.jts` start pose) (`Covers AE2`).
- **De-risk spike (pre-gate):** frames 1 and 2 alone reach IoU > 0.85 from their per-frame
  start poses — proves recoverability before the permanent multi-frame gate is wired.
- Integration: the multi-stage run reaches a branch and a leaf stage with `SetCallOffset`
  carrying `cost_function_calls_` forward — proving the cumulative-budget chain executes
  (assert budget accumulation and call count monotonicity across stages).
- Edge case: frame↔label correspondence heuristic fails (best start-pose IoU ≤ 0.50) → the
  test fails loudly with a clear message rather than silently reusing frame 0.
- Error path: a corrupted per-frame gate value fails the run — proving the expanded oracle
  still catches drift.

**Verification:** `ctest -L oracle` passes end-to-end (multi-frame, multi-stage) with per-frame
appearance gates green on the RTX 3090; baseline/golden updated; headless suite unaffected
(still 8+ green); documented budget caps recorded.

---

## System-Wide Impact

- **Interaction graph:** `LaunchOptimizer`, its 6 caller slots, and the 5 `on*` result slots
  are touched by U9 (via the controller). The two load slots are touched by U10. The oracle
  loop/target is touched by U11 only — production app code is untouched by U11.
- **Error propagation:** controller returns typed failure reasons; `LaunchOptimizer` maps them
  to the existing `QMessageBox::critical` messages (no user-visible change).
- **State lifecycle risks:** no new persistent state. `OptimizerManager` thread/manager leak
  on init failure is pre-existing and unchanged (noted, not fixed).
- **API surface parity:** `OptimizerManager::Initialize` signature and payload types are
  unchanged. `SessionState` is read (not mutated) by the controller.
- **Integration coverage:** U9's `ok`-then-Initialize path proves controller→manager wiring;
  the manual GUI smoke covers the button path. U11 is GPU-labeled and never runs headless.
- **Unchanged invariants:** cumulative budget (20k/25k/30k), DIRECT behavior, pose/render
  semantics, and the `OptimizeCoordinator` stub are all unchanged.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| R15 violation (rewiring production to the stub coordinator) | Controller owns decision/packaging only; MainScreen keeps the real `OptimizerManager` binding. Explicit non-goal. |
| AUTOMOC undefined-symbol link breakage from the new core `.cpp` | Add each new `.cpp` to the explicit `src/core/CMakeLists.txt` list (compounded gotcha). |
| Setting the cumulative budget "right" breaks R15 semantics | Keep `SetCallOffset` + `budget_ +=` chaining exactly as `Optimize()` does; do not fix to per-stage. |
| Oracle runtime blowup under full cumulative budget | Use a documented reduced chained budget; record expected runtime and raise `jtml.oracle` ctest `TIMEOUT` above it (current 3600s). |
| Oracle drift from production stage sequence | The oracle drives the *shared* stage-loop routine that production `Optimize()` also calls, so a production stage-sequence change fails the gate (R2 non-circularity); no hand-rolled reproduction, so no silent parallel-drift. |
| Frames 2/3 not recoverable → flaky permanent gate | Run a low-cost recoverability spike on frames 1/2 (per-frame start poses) before wiring the permanent gate. Commit a **hard floor** up-front: no frame may drop below an explicit IoU value (e.g. below a documented gate, the frame is an explicit scope-cut decision by the owner, never an ad-hoc loosening of 0.85). |
| `ui.`-count/lines not actually decreasing after cut | Measure pre/post per cut; if a cut increases coupling, adjust scope before committing. |
| Manual-visual gate not scheduled for presentation cuts | Explicitly schedule the load-button list-population visual check with U10. |

---

## Documentation / Operational Notes

- Record the expanded oracle budget caps and per-frame gates in `baseline.json` /
  `golden_oracle.org` at U11.
- Add a `docs/solutions/` note if the oracle-expansion reveals a new convention (e.g. calling
  `OptimizerManager` from the oracle, or a shared stage-loop helper).
- No docs for the controller/builder beyond the header comments (following the `session_state`
  precedent).

---

## Sources & References

- **Origin document:** [`docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md`](docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md)
- **Parent plan:** [`docs/plans/2026-08-07-001-refactor-testability-mvvm-plan.md`](docs/plans/2026-08-07-001-refactor-testability-mvvm-plan.md) (U7 phase-1 done; U9–U11 carry the re-scoped follow-on)
- **Handoff:** `docs/handoff-2026-08-07-testability-mvvm.md`
- Related code: `src/gui/mainscreen.cpp` (`LaunchOptimizer:4491`, load slots
  `on_load_image_button_clicked:2670`/`on_load_model_button_clicked:2852`,
  `update_image_list_widget:1581` [render-update only, NOT listname logic]),
  `src/core/optimizer_manager.cpp` (`Optimize:883`, `RunDirectStage:1225`),
  `test/oracle/oracle_test.cpp`
- **Compounded learnings:** `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md`, `docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md`
