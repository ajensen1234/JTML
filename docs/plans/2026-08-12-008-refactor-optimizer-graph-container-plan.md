---
title: "Optimizer Graph Container: stages-as-data over the existing engine (jtml-production PoC)"
type: refactor
status: active
date: 2026-08-12
origin: docs/brainstorms/2026-08-12-optimizer-path-requirements.md
---

# Optimizer Graph Container: stages-as-data over the existing engine

## Overview

JTML's optimizer run shape is hard-coded in `OptimizerManager::Optimize()`
(trunk 20k → 2×branch 5k → leaf 5k, cumulative 20/25/30/35k, dilation 6/4/1).
This plan makes that shape **data**: a StageScript (a vector of StageSpec)
built by pure functions, executed by the existing `Optimize()` loop through
the existing seams — driver, `RunDirectStage`, GPU cost path, IoU gates —
with **bit-identical behavior** as the feasibility PoC (requirement R6–R7).

The plan executes Phases 0–2 of the confirmed research-run path
(origin R8): the foundation that makes the baseline trustworthy (bug pins,
behavior-neutral fixes, the live distance-map index fix, exactly one
re-baseline, the CUDA-free finite-check), the two measurement instruments
that pin the container (z-profile probe, multi-stage oracle), and the
container itself (Cuts A/B/C/F → the `jtml-production` graph). The algorithm
battery, the CUDA/perf workstream (incl. Cut 0 instrumentation + Cut 4 meter
fix, origin R10), and the polish horizon are **separate follow-up plans** —
per the session decision to keep the graph work independent of CUDA work.

Normative grounding: `.panoptes/optimizer-deep-dive/synthesis.org` (angle 04
= backend architecture; findings 1–14) — every gate, pin, and line reference
below comes from it or from the verification pass this plan's research ran
against the current tree.

---

## Problem Frame

The run shape is code, not data: `Optimize()`'s stage loop
(`src/coordinator/optimizer_manager.cpp`) interleaves session/GPU state and
signal emission with the only algorithm policy — the trunk/branch/leaf
sequence. Changing any per-stage constant (budget, range, dilation, variant,
ordering) is an edit to a 1,700-line QObject manager, and the search is
currently measured against a baseline contaminated by one live kernel bug
(`distance_map_metric.cu:27` grid-index formula, verified). The owner's goal:
express runs as **graphs** (configurations, not code edits), prove the
container works bit-identically over the existing engine, and only then
improve the optimization itself.

The current shape, expressed as the v1 graph (verified against the code —
engine runtime values; `baseline.json`'s `dilation_px {6,3,1}` is the
known-stale docs-claim, reconciled by the probe in U5):

```
jtml-production:  [Trunk: classic DIRECT, budget 20000, range (35)^6, dil 6]
               -> [Branch x2: classic DIRECT, budget 5000 each, range (15,15,25,25,25,25), dil 4]
               -> [Leaf: classic DIRECT, budget 5000, range (3,3,15,3,3,3), dil 1]
cumulative caps: 20000 / 25000 / 30000 / 35000
```

---

## Requirements Trace

- R1 (graph = ordered stages, implicit seed-forwarding edges): U7, U9
- R2 (C++-typed named registry; manifest references by name): U7
- R3 (per-stage optimizer-variant slot, bit-identical classic default): U8
- R4 (execute through existing seams; no new layer; Cut E deferred): U7, U9
- R5 (schema must not preclude biplane / tiered-dilation / polish): U7
- R6 (v1 registry ships `jtml-production`; stubs allowed): U7, U9
- R7 (behavior-preserving PoC: caps, bookkeeping, poses, oracle green, qml
  parity; four lineage invariants asserted): U4, U6, U9
- R8 (sequencing confirmed: this plan executes Phases 0–2): U1–U10
- R9 (no algorithm delta before the apparatus): honored — no variant work in
  this plan; the battery is a follow-up plan
- R10 (Cut 0 + Cut 4 early): **deferred to the follow-up perf plan** (see
  Scope Boundaries — session decision to keep CUDA work separate; this
  plan's gates are oracle-IoU + bit-identity, not measured evals/s)

**Origin actors:** A1 (owner — registers graphs), A2 (measurement harness —
scores), A3 (production app — consumes the default graph unchanged)
**Origin flows:** F1 (graph-driven optimizer run), F2 (harness scoring)
**Origin acceptance examples:** AE1 (multi-stage oracle drives
`jtml-production` through the driver seam; costCalls on 20/25/30/35k; four
lineage invariants hold; per-frame IoU ≥ 0.85 vs the re-baselined values)

---

## Scope Boundaries

- No algorithm work: no variant switch, no battery, no `1-DTC-GL-gb` (R9).
  The per-stage variant slot is built (U8) and stays at bit-identical
  defaults.
- No CUDA/perf work: no Cut 0 instrumentation, no Cut 4 meter fix, no async
  copies, no N-way batching, no batch seam (origin R10–R14 → the follow-up
  perf plan, which also owns the meter's 1000× unit-error fix).
- No polish work: no four-arm study, no simplex arm, no G1–G4 gate (origin
  R16 → horizon plan).
- No measurement expansion beyond the pinning instruments: the ablation
  runner + `ablation.json` + perturbation suite are the follow-up
  measurement plan (origin R9's scoring surface is only *prepared* here via
  the probe + multi-stage oracle).
- No re-litigation of the run's resolved decisions (Mahfouz normalization
  kernel-as-spec; dilation three-way as ablation axis; gb numbers; GLh
  formula; NaN-impossibility of DIRECT_DILATION).
- No caching/memoization (origin R14), no biplane, no ML initializer beyond
  a schema note (origin R5/R15/R16).

### Deferred to Follow-Up Work

- **Perf plan (CUDA workstream)**: Cut 0 instrumentation, Cut 4 meter fix,
  Cut 1 async copies, streamed N-way + batch seam + N-bank eval state, all
  parked perf items (fragment-fill removal, metric fusion, resolution arm,
  pose-array design) — gated on the re-baseline this plan produces.
- **Measurement plan**: ablation runner + `ablation.json` + perturbation
  suite, bench-marks block, ci/nightly profiles.
- **Algorithm plan**: analytic battery, staged basin-capture metric, variant
  switch (needs this plan's harness).
- **Hygiene pass (run's Phase-3 items)**: Mahfouz value guards
  (`implant_mahfouz_metric.cu:324/:446`), curvature de-scope
  (`DIRECT_DILATION.cpp:42-43`), dilation doc reconciliation to measured
  values, ON-vs-OFF silhouette IoU protocol. Kept out of this plan to keep
  the container work focused; the re-baseline stays single-variable either
  way.
- **Polish horizon plan**: four-arm study, Yamazaki z-arm, biplane, ML
  initializer.

---

## Context & Research

### Relevant Code and Patterns

- `src/coordinator/optimizer_manager.cpp` — the `Optimize()` stage loop
  (landmarks: trunk reset :995–996, branch group single init/dilate/emit
  :1052–1076 before the repeat loop :1080, `budget_ +=` :1096/:1151,
  `CalculateSymTrap` :1135, leaf error-gated destruct :1158–1163, epilogue
  trunk-restore :1170–1184), `RunDirectStage` + injected-cost lambda
  (:1234–1252), `EvaluateCostFunctionAtPoint` (:1372–1387), the manager's
  per-manager parameter scan (dilation/dark-silhouette, :237–345).
- `include/coordinator/optimizer_run_driver.h` + `src/coordinator/optimizer_run_driver.cpp` —
  the by-value `OptimizerRunLaunch` seam; adapter forwards verbatim
  (:58–78); sole `new OptimizerManager` at :31.
- `include/domain/direct_optimizer.h` + `src/domain/direct_optimizer.cpp` —
  the extracted DIRECT: injected `std::function<double(const Point6D&)>`
  cost, cumulative guard `(calls + offset) < budget_`, `SetCallOffset`,
  callbacks, largest-denormalized-side one-side trisection. Note the stale
  illustrative comment at direct_optimizer.h:58–59 ("trunk 10k → branch 20k
  → leaf 30k") — fix opportunistically in U8.
- `include/coordinator/optimizer_run_controller_core.h` — typed `Directive`
  enum incl. `SymTrap=5` (:64–69); the counter-derived observation channel
  (stageText/costCalls/progress) the oracle's stage-bookkeeping gate uses.
- The named-registry pattern to mirror: `CostFunctionManager`'s
  `listCostFunctions()` (CostFunctionManager.cpp:343+) — C++-typed named
  data — and its services-layer sibling `BuildCostFunctionRegistryEntries`
  (`src/services/cost_function_registry.cpp`, golden-pinned).
- `test/oracle/oracle_test.cpp` — the GPU Tier-2 oracle; the hand-rolled
  cost twin at :286–291 (monoplane-only; `kBudget=3000` flat shape; Canny
  3/0/150 pinned; IoU/L1 gate vs the Study2Grid label).
- `test/golden/baseline.json` — canonical caps [20000, 25000, 30000, 35000];
  the re-baseline event's target.
- CMake conventions — `src/coordinator/CMakeLists.txt` uses an explicit
  .cpp source list + header GLOB: new `optimizer_stage_script.cpp` needs one
  added line; test targets follow the direct-compile pattern (see
  Institutional Learnings).

### Institutional Learnings

- `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md` —
  the master playbook: cumulative budget 20/25/30/35k is **load-bearing** (do
  NOT "fix" to per-stage caps); direct-compile test pattern; AUTOMOC gotcha
  (Q_OBJECT headers in the test target's source list); oracle/baseline.json
  contract; Tier-1 golden + hegel PBT twin house rule.
- `docs/solutions/conventions/jtml-layered-lib-split-2026-08-08.md` —
  layered-lib conventions; new .cpp files must be added to the explicit
  source list; attesting zero behavior change for pure relocation.
- `docs/solutions/conventions/jtml-shared-vm-layer-2026-08-11.md` — the
  shared-layer map: `OptimizerRunController`/core (epoch gate, stage math),
  driver seam, relay channel; the multi-stage oracle is the seam's next
  consumer.
- `docs/solutions/logic-errors/cost-function-parameter-double-truncation-2026-08-08.md` —
  typed cost parameters: hegel PBT invariants catch type-class bugs in the
  parameter registry (relevant to U1's metric-semantics PBT twins).
- `docs/solutions/logic-errors/jtml-heatmap-guard-allocator-preconditions-2026-08-12.md` —
  guard preconditions + uninitialized-membership pitfalls (relevant to U3's
  finite-check and the Mahfouz value-guard design).
- `docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md` —
  hegel + oracle tooling recipes (rpath/dl for props targets; GPU-labeled
  oracle construction).
- `docs/solutions/test-failures/jtml-test-isolation-qsettings-cache-2026-08-12.md` —
  per-process QSettings path caching (relevant if oracle tests touch config).

### External References

- None needed — the panoptes deep-dive (`.panoptes/optimizer-deep-dive/synthesis.org`
  + angles 01–06, Zotero-corpus-grounded) is the freshly completed external
  research; this plan's research pass verified its claims against the code.

---

## Key Technical Decisions

- **StageScript as pure functions in a dedicated TU**
  (`include/coordinator/optimizer_stage_script.h` + `src/coordinator/optimizer_stage_script.cpp`):
  `StageSpec {StageKind kind, Point6D range, unsigned int budget, unsigned
  int repeat, unsigned int cfm_index}` + `BuildStageScript` +
  `DeriveStageCostParams` — one added line in `jtml_coordinator`'s explicit
  source list. Mirrors the run's angle-04 sketch; verified 1:1 against the
  current loop blocks (trunk `{Trunk, (35)^6, 20000, 1, 0}`, branch group
  `{Branch, branch_range, 5000, number_branches, 1}`, leaf `{Leaf, (3,3,15,3,3,3), 5000, 1, 2}`,
  Sym_Trap `{Leaf, ..., 0, 2}` for repeat=0).
- **`DeriveStageCostParams` reproduces the manager's parameter scan**
  (optimizer_manager.cpp:237–345) — NOT the CFM internals: dilation stays
  cost-owned, and the wizard "DO NOT EDIT" regions in `CostFunctionManager.*`
  are never touched by the derivation (the stage-guard bug fix at
  CostFunctionManager.cpp:46–48 is the one in-place exception, one line,
  flagged with a comment).
- **`BuildGpuCostAdapter` carries `Calibration` by value** (monoplane
  default) — closes the oracle-twin's monoplane-only gap and serves the
  future biplane consumer without a signature change.
- **`DirectOptimizer::Options` via guarded divergence**: every new path
  guards on "different from default" so defaults reproduce today's search
  bit-identically (the run's R13 pin-first doctrine); the launch plumbing
  (OptimizerRunLaunch → adapter →
  Initialize → RunDirectStage) is additive only.
- **The stage-guard bug is fixed in place** (tautology `||` → `&&`): the
  CFM's `stage_` is not the stage source of truth going forward — the
  StageScript `cfm_index` is — so no accessor relocation into wizard-owned
  regions.
- **One re-baseline, single-variable**: distance-map index fix only; Canny
  pinned 3/0/150; recovered-pose delta recorded explicitly; no other change
  bundled (origin Finding 12).
- **Convergence-tradeoff design note lives in the stage-script TU**: cover is
  a stage-loop property, intentionally dropped across stages (Flood L447–458);
  `Options` carries NO cover knob.
- **Cut D disposition (review-resolved):** the run's Cut D (the adapter
  factory, "stage-shape-agnostic") is **merged into U9**:
  `BuildGpuCostAdapter` IS that factory, absorbing only the oracle twin's
  body. No separate cut.
- **Dilation pin rule (review-resolved):** the engine's runtime value is
  6/4/1 (the code path the oracle executes); `baseline.json`'s `dilation_px
  {6,3,1}` is the known-stale docs-claim. U1/U6 pins assert the ENGINE
  value; reconciliation to the measured value happens after U5's probe data,
  in the hygiene pass.
- **Known-visible defect during this plan's duration:** the ms/call meter's
  1000× unit error (Linux IPS/ETA display) remains visible until the perf
  plan lands. Owner = perf plan; the wall-clock half is eligible for the
  hygiene pass. Recorded here so it is tracked, not silently parked.

---

## Open Questions

### Resolved During Planning

- StageSpec field set: `{kind, range, budget, repeat, cfm_index}` — verified
  against the loop blocks (angle 04 R2-2 + research pass).
- Registry TU placement: dedicated `optimizer_stage_script` TU in
  jtml_coordinator (headers auto-globbed, one explicit .cpp line).
- Stage-guard fix vs the wizard banner: fixed in place with a comment;
  `cfm_index` is the future source of truth.

### Deferred to Implementation

- Exact `BuildStageScript(OptimizerSettings, directive)` mapping for
  directives beyond Single/Sym_Trap (All/Each/From/Backward): transcribe
  the current branch of `Optimize()` verbatim during U9 — the manifest
  records which directive produced which script.
- Whether the finite-check's GLh surrogate hook is stubbed or omitted in
  U3: the run allows either (the hook is reserved); decide when the battery
  plan lands.
- N-bank count, nsys availability, GPU-active verdict: owned by the
  follow-up perf plan, not this one.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
Today's Optimize() (verified landmarks)          After Cut B (script-driven)
------------------------------------------------  -------------------------------------------------
trunk block (:995-1044)                          stage_script_ = BuildStageScript(settings, directive)
  init CFM trunk (cfm 0)                            for each StageSpec s in stage_script_:
  dilate + emit UpdateDilationBackground              init CFM s.cfm_index + DeriveStageCostParams
branch group (:1052-1085)                            dilate + emit UpdateDilationBackground
  init CFM branch (cfm 1)                             for i in 0..s.repeat:
  dilate + emit                                        re-seed from current optimum (per-repeat)
  for b in 0..number_branches:                        RunDirectStage(BuildGpuCostAdapter(...), s)
    re-seed from current optimum                      budget accumulates (SetCallOffset)
    RunDirectStage(...)                             epilogue trunk-restore dilate + emit
leaf block (:1111-1163)                           (Sym_Trap: repeat=0 -> init+dilate+emit,
  init CFM leaf (cfm 2)                               CalculateSymTrap, no search)
  dilate + emit
  CalculateSymTrap()  (directive SymTrap)
  RunDirectStage(...)  -- gated on !sym_trap_call (the leaf SEARCH is
                        skipped under SymTrap; init + CalculateSymTrap run)
epilogue trunk-restore (:1170-1184)

jtml-production graph (registered data, v1):
  [{Trunk,(35)^6,20000,1,0}, {Branch,(15,15,25,25,25,25),5000,2,1}, {Leaf,(3,3,15,3,3,3),5000,1,2}]
```

Four lineage invariants (asserted in the multi-stage oracle, U6):
group-once dilation across repeats · per-repeat re-seed (assert the
recovered-pose **sequence**) · asymmetric z-leaf · frame-to-frame seed
chaining.

---

## Implementation Units

- [ ] U1. **Tier-0 metric-semantics pins + CPU references**

**Goal:** Characterize the cost-path behaviors at the pure surface
before any fix — the run's R13 characterization pass that creates the surfaces the
fixes plug into.

**Requirements:** R8 (Phase 0 pins); supports R7, R9

**Dependencies:** None

**Files:**
- Create: `test/unit/test_metric_semantics.cpp`, `test/unit/test_metric_semantics_properties.cpp`
- Modify: `test/CMakeLists.txt` (two new targets, `headless` label, TIMEOUT)

**Approach:**
- Per-metric deterministic cases + hegel PBT invariants, per the run's
  Tier-0 inventory: chamfer stage functions; distance-map
  CropIndexToGlobal full-coverage pin (**corrected formula passes, buggy
  formula fails** — RED today, this is the spec for U4's fix); Mahfouz
  float-vs-truncated ratio pair; IoU/L1 with the `IOU(∅,∅)` reference
  decision (1.0 as spec-with-flagged-deviation); dilation registry/constants
  pins extended to the lineage values (settings_constants.h IS a Flood tibia
  transcription — pin it as such); the stage-guard accessor pin (the run's
  pin list assumes an accessor that does not exist today — see the second
  wizard-region exception below); the
  sym_trap tibia-transform pin; DD PolePenalty extraction + init-0/Y-axis
  pins.
- **Second documented wizard-region exception:** a minimal `getStage()`
  accessor in `include/compute/CostFunctionManager.h` (3 lines, commented) —
  the only way the stage-guard pin is observable. `stage_` is dead state
  today (never read); the pin is pure hygiene, cheap to land.
- Direct-compile pattern: pure sources compiled into the test targets (no
  Qt/CUDA surface); props target follows the hegel rpath/dl recipe.

**Execution note:** characterization-first — the index pin is RED against
today's kernel by design; do not "fix" the test to match the bug.

**Patterns to follow:** `test/unit/test_direct_optimizer.cpp` +
`_properties.cpp` target wiring (direct-compile, Catch2 + hegel);
`docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md`.

**Test scenarios:**
- Happy path: each metric's CPU reference returns the documented value for a
  hand-computed input (e.g., a 2×2 crop with known EDGE/DILATED pattern).
- Edge case: `IOU(∅,∅)` returns the spec'd 1.0 (flagged deviation from the
  kernel's 0/0 NaN).
- Edge case: distance-map CropIndexToGlobal covers every pixel exactly once
  for the oracle's actual 64×64-crop / 16×16-block geometry (PBT invariant:
  bijective mapping).
- Edge case: Mahfouz truncated-vs-float ratio pair differs measurably (the
  characterization that justifies kernel-as-spec on 2.55/×255/−2.67/−1).
- Error path: the buggy index formula demonstrably fails the full-coverage
  invariant (the pin is RED pre-fix — run and record, then U4 flips it).
- Integration: registry/constants pins match `baseline.json`'s recorded
  budget shape; dilation values pin the ENGINE runtime (6/4/1) —
  `baseline.json`'s `dilation_px {6,3,1}` is the known-stale docs-claim,
  reconciled after U5's probe data.

**Verification:** `pixi run test` headless green with the index pin
recorded RED (expected-fail surfaced, not silently removed); the PBT twins
pass their invariants.

---

- [ ] U2. **Behavior-neutral cost-path fixes**

**Goal:** Land the five behavior-neutral fixes — zero oracle impact — as
separate jj changes, closing the latent bug classes before the live one.

**Requirements:** R8 (Phase 0)

**Dependencies:** U1

**Files:**
- Modify: `src/compute/CostFunctionManager.cpp` (stage guard :46–48),
  `src/compute/DD_NEW_POLE_CONSTRAINT.cpp` (:134 init, :117–120 Y-axis),
  `src/compute/sym_trap_function.cpp` (:106 tibia x→z), `src/compute/DIRECT_DILATION.cpp`
  (dead duplicate :101–104)
- Test: `test/unit/test_metric_semantics.cpp` (pins flip from RED/spec to green)

**Approach:**
- Stage guard: tautology `||` → `&&` (one line, comment noting the wizard
  banner; `cfm_index` is the future source of truth).
- DD: initialize `min_dist` (removes the UB read; also cover the
  all-flags-false return path); fix `Y_dist` to a perpendicular axis.
- sym_trap: tibia x→z argument correction at :106 (Bug 4 — the trap
  analysis prerequisite).
- Delete the dead duplicate DistanceMapMetric call
  (`DIRECT_DILATION.cpp:101–104`).
- One jj change per fix; each is behavior-neutral for the oracle's cost
  (DIRECT_DILATION) by construction — the pins from U1 prove it.

**Execution note:** pin-first — each fix lands against its U1 pin, one
change at a time.

**Patterns to follow:** the run's Phase-1 recipes (angle 02); pin-first
doctrine.

**Test scenarios:**
- Happy path: after the stage-guard fix, the accessor pin reports the
  stage set by the caller (previously always Trunk).
- Edge case: DD min_dist with all pole flags false returns the documented
  fallback (no garbage).
- Edge case: DD Y_dist differs from X_dist for an asymmetric pole
  placement.
- Integration: oracle unaffected — full oracle run (U4's precondition) shows
  IoU/L1 unchanged vs the pre-fix baseline within noise.

**Verification:** headless suite green; oracle IoU/L1 identical to the
pre-fix recorded values (evidence the fixes were behavior-neutral); the U1
pins that were RED/spec-documenting now assert the fixed behavior.

---

- [ ] U3. **CUDA-free finite-check at EvaluateCostFunction**

**Goal:** Close the dead-weight NaN class at the one shared chokepoint —
`DirectOptimizer::EvaluateCostFunction` — regardless of GLh.

**Requirements:** R8 (Phase 0); supports R9

**Dependencies:** U1

**Files:**
- Modify: `include/domain/direct_optimizer.h`, `src/domain/direct_optimizer.cpp`
  (finite-check at the eval chokepoint: `non_finite_count_`,
  `GetNonFiniteCount()`, iteration-callback event; GLh surrogate hook
  reserved)
- Test: `test/unit/test_direct_optimizer.cpp` (headless Tier-0 probe)

**Approach:**
- Wrap the result of the injected cost: `!std::isfinite(result)` →
  increment the counter, treat as infeasible (never store, never wins the
  optimum), and surface an iteration-callback event. Behavior-neutral for
  all finite evals (DIRECT_DILATION provably finite — no oracle impact).
- CUDA-free by construction (domain layer purity: `include/domain/direct_optimizer.h`
  stays Qt/GPU-free).

**Execution note:** test-first — the headless probe drives a fake cost
lambda returning NaN/±Inf/sNaN.

**Patterns to follow:** the run's finite-check design (angle 02, Round 2);
the guard-precondition lesson from
`docs/solutions/logic-errors/jtml-heatmap-guard-allocator-preconditions-2026-08-12.md`.

**Test scenarios:**
- Happy path: finite evals behave identically (counter stays 0, optimum
  updates normally).
- Edge case: NaN / +Inf / −Inf / sNaN costs — no crash, counter increments,
  the optimum never reflects a non-finite eval, surviving storage columns
  stay finite.
- Edge case: a NaN sequence followed by finite evals — the search resumes
  normally.
- Integration: the iteration-callback event fires once per non-finite eval
  batch (ordering preserved).

**Verification:** headless suite green; `GetNonFiniteCount()` observable in
the probe; no change to the finite-path trace.

---

- [ ] U4. **Distance-map index fix + adversarial GPU probe + one re-baseline**

**Goal:** Fix the one live kernel bug and re-baseline exactly once,
single-variable, recording the recovered-pose delta — the trustworthy
baseline every later step (this plan's PoC and the follow-up plans) is
judged against.

**Requirements:** R7, R8 (Phase 0); the critical path of the whole program

**Dependencies:** U2, U3

**Files:**
- Modify: `src/compute/distance_map_metric.cu` (:27 — `blockIdx.y + gridDim.x`
  → `blockIdx.y * gridDim.x`), `test/golden/baseline.json` (re-baseline
  event), `test/oracle/oracle_test.cpp` (run_config pin note; adversarial
  probe rides along)
- Test: `test/unit/test_metric_semantics.cpp` (the U1 index pin flips
  RED→green), `test/oracle/oracle_test.cpp` (adversarial finiteness probe)

**Approach:**
- Apply the one-line index fix; the U1 full-coverage pin is the spec.
- Run the Tier-2 oracle: sanity sweep + adversarial poses (±200 mm off-axis,
  behind camera, 90°/180° rotations — assert finite AND
  `cudaGetLastError() == cudaSuccess` after each eval; DIRECT_MAHFOUZ at an
  empty-silhouette pose is expected RED today — characterization only).
- Re-baseline `baseline.json` in the same change, recording the
  recovered-pose delta explicitly (direction prediction: z moves toward
  `fem.jts`, the 6.31 mm frame-1 gap shrinks; magnitude unknown until the
  GPU run). Single-variable: nothing else bundled (Canny pinned 3/0/150).

**Execution note:** characterization-first; the re-baseline is an event
with recorded numbers, not a prose edit.

**Patterns to follow:** the run's Finding 1/12 re-baseline rules; the
oracle's existing gate structure (`oracle_test.cpp`).

**Test scenarios:**
- Happy path: `Covers AE1.` — post-fix oracle per-frame IoU ≥ 0.85 on all
  three Kneel_1 frames (gate unchanged).
- Edge case: the adversarial sweep (off-axis/behind-camera/rotated poses)
  yields finite costs and clean `cudaGetLastError()` — the sticky-error
  trap stays closed.
- Edge case: DIRECT_MAHFOUZ at an empty-silhouette pose records the
  expected NaN characterization (no assertion change; documented).
- Integration: baseline.json's recorded pose/IoU/L1/z-gap delta is the new
  reference — the plan's subsequent gates cite it.

**Verification:** one re-baseline event (jj change) containing the index
fix + oracle re-run + baseline.json delta; headless green; the delta is
recorded, not asserted (direction claim confirmed or refuted by the data).

---

- [ ] U5. **z-profile probe**

**Goal:** The cheapest instrument that sees the z-weak axis — executable
per the run's spec, consuming the same injected-cost lambda as production.

**Requirements:** R8 (Phase 1); supplies the polish precondition's data
later

**Dependencies:** U4

**Files:**
- Create: `test/oracle/z_profile_test.cpp` (new `jtml.z_profile` target,
  `oracle`/`gpu` label, TIMEOUT 3600 — nightly-grade; measure after run 1
  and adjust)
- Modify: `test/CMakeLists.txt`

**Approach:**
- 31-render Δz sweep at fixed truth per (variant, dilation, frame) via the
  `EvaluateCostFunctionAtPoint` entry point (`optimizer_manager.cpp:1372–1387`)
  and the oracle twin; report valley_depth / argmin_offset /
  plateau_halfwidth / noise_floor (5 repeats of cost(0), expect 0 variance
  on the int-atomic path).
- Term-decomposition mode: full/chamfer/dilated costs from the SAME render
  at each sweep point (zero extra renders).
- Sweep set: dilations {6,3,1,4} (the probe arbitrates 6/3/1-vs-6/4/1 by
  data) + the lineage set S1 {6,4,1}, S2 {6,3,1}, S3 {10,6,1}, S4 {6,2,1} +
  the bilateral-vs-render-only semantics axis; sweep direction = the
  focus→COG ray (record the off-axis angle vs camera-z).
- yamazaki-polynomial mode (order-4 fit on the same 31 renders:
  poly_argmin + fit residual) and coupling mode (± the minimal-detectable-
  regression in-plane Δ to measure dz*/d(in-plane)).
- Assertions p1–p4: min-at-zero within noise; valley_depth < 0.05 ⇒
  z-blind (recorded); ranking asserted pin-first (run 1 records, later runs
  enforce); z_gap anti-correlates with valley_depth.

**Execution note:** pin-first — run 1 records the ranking, later runs
enforce it.

**Patterns to follow:** the run's probe spec (angle 03 R2-2/R3-1/R3-2;
angle 06 R2-1/R3-4); the oracle's cost-lambda entry points.

**Test scenarios:**
- Happy path: sweep over the golden pose ±15 mm z yields a V-shaped cost
  with min-at-zero within the noise floor (recorded, not enforced, on the
  first nightly run).
- Edge case: an injected z-flat variant (or dilation where valley_depth <
  0.05) is classified z-blind and recorded as informational.
- Edge case: noise_floor = 0 on the int-atomic path (5 repeats identical).
- Integration: term-decomposition values satisfy full = chamfer + dilated
  composition at each sweep point (same-render consistency).
- Integration: `Covers AE1.`-adjacent — probe + oracle agree on the
  recovered z at the golden pose within the recorded band.

**Verification:** `jtml.z_profile` runs on the GPU machine; the JSON record
(per-variant valley metrics) is produced; p1 holds.

---

- [ ] U6. **Multi-stage oracle on the driver seam**

**Goal:** The production-shaped oracle (20k→25k→30k→35k, per-stage
dilation, tibia-after-femur, sym_trap) through the `OptimizerRunDriver`
seam — the pinning instrument for Cut B and the home of the four lineage
invariants.

**Requirements:** R7, R8 (Phase 1); pins U9

**Dependencies:** U4 (U5 recommended — the probe's ranking records inform
the oracle's z expectations)

**Files:**
- Create: `test/oracle/multistage_oracle_test.cpp` (new
  `jtml.oracle_multistage` target, `oracle`/`gpu` label, TIMEOUT 7200 —
  nightly-grade (the existing flat oracle already uses TIMEOUT 3600;
  MEASURED: ~15 s total at ~7k evals/s — U6 refutes the derived 20–60
  evals/s; the generous TIMEOUT is kept per spec)
- Modify: `test/CMakeLists.txt`, `test/golden/baseline.json` (z-profile +
  oracle records)

**Approach:**
- Drive `OptimizerRunController` directly (SingleModelOnly is a QML-bridge
  policy only — the oracle bypasses the bridge; the QML bridge's own run
  path is covered separately by `test/oracle/qml_parity_check.cpp`) with
  the production shape — the same
  shape the `jtml-production` graph will express in U9.
- Three sym-trap pins, not a seam addition: launch.directive ==
  "Sym_Trap"; `orientationSymTrapUpdated` relay count ≥ 60 (the happy path
  emits 61: 60 sweep poses + 1 restore emit; a count of 0 catches
  CalculateSymTrap's zero-pose early return, ~:1304–1308); costCalls()
  lands at **0** — U6-measured: the `if (!sym_trap_call)` guard at
  optimizer_manager.cpp:927 wraps trunk AND branches, so under SymTrap only
  the leaf-CFM init + CalculateSymTrap run (60 uncounted analysis evals;
  stageText stays "Idle"; the early return at :1201 skips the final
  UpdateDisplay). (Both the synthesis's "20000" and the review's "30000"
  readings were wrong — the outer guard was missed.)
- Side effects to plan for: CalculateSymTrap writes `Results.csv` /
  `Results.xyz` / `Results2D.xy` into the process CWD and sleeps ~5 s
  (60 × 5000/60 ms) + 60 extra leaf-cost evals. Run the oracle from a
  scratch CWD (or accept + clean the three files after the tibia pass).
- Stage-bookkeeping gate: costCalls() on the cumulative caps
  20/25/30/35k; gates per frame: IoU ≥ 0.85 (hard), per-px L1 banded-record,
  z-gap banded-informational (15 mm band, never a gate).
- Tibia-after-femur: Run 1 femur (Single, fem.jts seed) → read
  optimizedFrameRelayed → Run 2 tibia (SymTrap, femur row = Run-1
  recovery); pin the tibia label correspondence with the start-pose-IoU
  procedure before the tibia gate means anything.
- Four lineage invariants (assertion homes): group-once dilation across
  branch repeats; per-repeat re-seed — assert the recovered-pose SEQUENCE
  (branch i's pose feeds branch i+1's seed); asymmetric z-leaf; frame-to-
  frame seed chaining (frame N's trunk seed == frame N−1's recovery).
- z_leaf_budget ∈ {5000, 20000, 50000} × z-range {15, 20} recorded as the
  first-class ablation axis (manifest fields; the JTML default 5000/15
  measured first).

**Execution note:** characterization-first — run 1 records the ranking
expectations; assertions enforce from run 2.

**Patterns to follow:** the fake-driver test pattern
(`test/lifecycle/optimizer_run_controller_test.cpp`); the run's oracle spec
(angle 03 R2-3/R3-1/R3-3; angle 04 R3-3).

**Test scenarios:**
- Happy path: `Covers AE1.` — femur run costCalls lands on 20/25/30/35k;
  per-frame IoU ≥ 0.85 vs the re-baselined values; stageText reports
  Trunk → Branch 1/2 → Extra Z-Translation (the leaf; the channel never
  says "Leaf") → Finished.
- Edge case: sym_trap directive — costCalls lands at 0 (U6-measured: the
  outer `!sym_trap_call` guard skips trunk AND branches; only leaf init +
  CalculateSymTrap run); relay count ≥ 60 (61 on the happy path); stageText
  stays "Idle".
- Edge case: `number_branches` = 0 or leaf disabled — the script's enabled
  flags map 1:1 (no phantom stages).
- Error path: a run whose stage bookkeeping misses a cap fails the gate
  (the oracle can SEE what the flat-3000 oracle is blind to).
- Integration: tibia-after-femur seed chaining — femur recovery feeds the
  tibia trunk seed; the recovered-pose sequence is asserted, not just the
  final pose.
- Integration: frame-to-frame chaining across the 3 Kneel_1 frames.

**Verification:** `jtml.oracle_multistage` green on the GPU machine;
costCalls/stageText/IoU records appended to baseline.json; the four lineage
invariants asserted.

---

- [ ] U7. **Cut A — stage-script TU + pure builders + graph registry**

**Goal:** The container's pure surface: StageScript types, the named
builders, the `jtml-production` graph as registered data — zero production
behavior change.

**Requirements:** R1, R2, R4, R5, R6

**Dependencies:** U1 (pins); parallel with U5/U6

**Files:**
- Create: `include/coordinator/optimizer_stage_script.h`,
  `src/coordinator/optimizer_stage_script.cpp`
- Create: `test/unit/test_stage_script.cpp` (+ `_properties.cpp` twin)
- Modify: `src/coordinator/CMakeLists.txt` (one explicit .cpp line),
  `test/CMakeLists.txt`

**Approach:**
- `StageKind {Trunk, Branch, Leaf}` + `StageSpec {kind, range, budget,
  repeat, cfm_index}`; repeat=0 expresses the Sym_Trap no-search leaf;
  the SymTrap script is `[{Leaf, repeat=0}]` — U6-measured: today's engine
  skips trunk AND branches under SymTrap (the `if (!sym_trap_call)` guard at
  optimizer_manager.cpp:927 wraps both; costCalls == 0, only
  CalculateSymTrap's 60 uncounted analysis evals run). The script-driven
  loop must reproduce this bit-identically;
  repeat=N expresses the branch group (dilate+emit once, re-seed per
  repeat).
- `BuildStageScript(settings, directive)` reproduces the current loop's
  shape verbatim for every directive (Single/SymTrap fully; All/Each/From/
  Backward transcribed from the current branch); `DeriveStageCostParams`
  reproduces the manager's per-manager parameter scan (dilation /
  dark-silhouette) — never the CFM internals (wizard regions untouched).
- The registry: named graphs as C++ data (mirroring `listCostFunctions`),
  v1 = `jtml-production`; stubs recorded for future kinds (polish stage,
  initializer prefix, flood-direct-jta shape — data-only, not exercised).
- Convergence-tradeoff design note in the TU (cover is a stage-loop
  property, dropped across stages; Options carries no cover knob).
- Unit pins: BuildStageScript maps settings → exact StageSpec sequence;
  DeriveStageCostParams maps CFM params → expected dilation/dark values;
  hegel PBT: length preservation, budget monotonicity, repeat semantics.

**Execution note:** test-first for the pure builders (they are the spec).

**Patterns to follow:** `CostFunctionManager::listCostFunctions` (named
registry), `BuildCostFunctionRegistryEntries`
(`src/services/cost_function_registry.cpp`, golden-pinned mapping).

**Test scenarios:**
- Happy path: default settings → the exact three-spec sequence
  [{Trunk,20000,1,0},{Branch,5000,2,1},{Leaf,5000,1,2}] with the canonical
  ranges.
- Edge case: `number_branches`=0 / leaf disabled → stage absent, caps
  recompute correctly.
- Edge case: repeat=0 (Sym_Trap) → leaf spec with search suppressed.
- Edge case: PBT — budget sums to the cumulative caps for randomized
  settings; stage order is preserved; no negative/zero budgets.
- Integration: DeriveStageCostParams matches the manager's runtime
  parameter scan for the production settings (golden-pinned values).
- Error path: an invalid directive or a negative budget fails fast with a
  clear error.

**Verification:** headless suite green; `jtml-production` graph
registration test passes; no production binary behavior change (existing
suite green untouched).

---

- [ ] U8. **Cut C — DirectOptimizer::Options with bit-identical defaults**

**Goal:** The per-stage optimizer-variant slot (origin R3) as a typed
Options struct whose defaults reproduce today's search bit-identically.

**Requirements:** R3, R9

**Dependencies:** U1 (Tier-1 golden pins); parallel with U5–U7

**Files:**
- Modify: `include/domain/direct_optimizer.h`, `src/domain/direct_optimizer.cpp`
- Modify: `include/coordinator/optimizer_run_driver.h` (OptimizerRunLaunch
  gains the additive Options field, defaulted)
- Modify: `src/coordinator/optimizer_run_driver.cpp` (adapter forwarding
  verbatim), `src/coordinator/optimizer_manager.cpp` (Initialize →
  RunDirectStage plumbing)
- Test: `test/unit/test_direct_optimizer.cpp`, `test/oracle/oracle_test.cpp`
  (parity)

**Approach:**
- Options fields per the run's mapping: selection=Original, epsilon=0.0,
  delta_limit=off, size_measure=L2, split_rule=OneSide, ties=All,
  hidden_constraints=off, globally_biased=off — each mapped line-by-line
  onto today's code.
- Guarded divergence: every path guards on "different from default" before
  diverging (the run's R13 bit-identical-defaults proof strategy).
- **Scope boundary (review-resolved):** non-default Options fields are
  FAIL-FAST STUBS in this plan — selecting one produces a clear error.
  No variant semantics ship here; the divergence branches land with the
  algorithm plan. The slot's STRUCTURE and the default-path bit-identity
  are what this unit proves.
- Launch plumbing end-to-end: OptimizerRunLaunch (additive field) → adapter
  → Initialize → RunDirectStage ctor; run the Initialize-caller sweep first
  (only the adapter constructs the manager per research — verify before
  landing).
- Fix the stale illustrative comment at direct_optimizer.h:58–59
  opportunistically (canonical caps are 20/25/30/35k).

**Execution note:** characterization-first — Tier-1 golden + PBT twin +
oracle parity are the numeric spec.

**Patterns to follow:** the run's Options cut recipe (angle 04 R2-6/R3-2).

**Test scenarios:**
- Happy path: default Options produce an identical Tier-1 golden trace
  (convergence + budget accounting) to the pre-Options code.
- Edge case: each non-default field hits the fail-fast guard with a clear
  error (stub semantics — no variant behavior ships in this plan; the
  divergence branches land with the algorithm plan).
- Integration: oracle parity — baseline config via the new plumbing yields
  IoU/L1 within the recorded noise of the re-baselined values.
- Integration: fake-driver suite records the launch by value — compile-
  unchanged (additive field).

**Verification:** headless + oracle green; parity diff empty for the
default config; call-site sweep recorded (4-arg sites today → Options
defaulted sites).

---

- [ ] U9. **Cut B — script-driven Optimize loop + BuildGpuCostAdapter (the PoC)**

**Goal:** The feasibility PoC: `Optimize()` consumes the StageScript and
runs `jtml-production` through the adapter — bit-identical against the
re-baselined oracle, with the four lineage invariants asserted.

**Requirements:** R1, R4, R6, R7; AE1

**Dependencies:** U6 (multi-stage oracle pins the current shape), U7
(script + registry), U8 (Options plumbing)

**Files:**
- Modify: `src/coordinator/optimizer_manager.cpp` (script-driven loop +
  `BuildGpuCostAdapter`), `include/coordinator/optimizer_manager.h`
- Modify: `test/oracle/oracle_test.cpp` (the :286–291 twin body moves into
  the adapter; golden assertions stay verbatim), `test/oracle/qml_parity_check.cpp`
  (the bridge's run path must stay green through the script-driven loop —
  the instrument drives `OptimizerBridge::run()` with the degenerate
  3000/0/0 shape)
- Test: `test/oracle/multistage_oracle_test.cpp` (invariant assertions),
  `test/unit/test_stage_script.cpp` (builder pins unchanged)

**Approach:**
- The loop becomes: init CFM per StageSpec (cfm_index) + DeriveStageCostParams
  → dilate + emit once per group → per-repeat re-seed → RunDirectStage with
  BuildGpuCostAdapter → epilogue trunk-restore, transcribed verbatim.
- `BuildGpuCostAdapter(principal_model, calibration, stage_manager)` =
  the RunDirectStage lambda body + the oracle twin's body, carrying
  Calibration by value (monoplane default) — three consumers converge
  (production runner, oracle, future z-profile probe).
- Preserve the leaf-destruct error-gating asymmetry verbatim (transcribe,
  don't fix — flagged to the hygiene pass); preserve the epilogue emit
  order.
- The golden's end-to-end assertions stay verbatim; only the twin body
  moves. The four lineage invariants (U6) are the preservation contract.
- Stage bookkeeping unchanged: `budget_`/`cost_function_calls_`/stage flags
  emit through the existing observation channel (counter-derived; the
  oracle's caps gate is the proof).

**Execution note:** characterization-first — prove bit-identity (pose→score
diff + oracle parity) before any cleanup.

**Patterns to follow:** the run's Cut B pin strategy (angle 04 R2-3/R3-3);
the driver seam's by-value discipline.

**Test scenarios:**
- Happy path: `Covers AE1.` — the multistage oracle drives
  `jtml-production` through the script-driven loop; costCalls lands on
  20/25/30/35k; per-frame IoU ≥ 0.85 vs the re-baselined values; stageText
  sequence Trunk → Branch 1/2 → Extra Z-Translation (the leaf) → Finished.
- Integration: recovered-pose SEQUENCE across branch repeats (invariant 2)
  matches the pre-Cut-B oracle record.
- Integration: frame-to-frame seed chaining (invariant 4) across the 3
  frames.
- Edge case: Sym_Trap directive → repeat=0 leaf → no search, costCalls at
  0, relay ≥ 60 (pins from U6 stay green through the relocation).
- Error path: a stage with a bad cfm_index fails fast with the manager's
  existing error path (no silent skip).
- Parity: recorded pose→score diff empty between the pre-Cut-B and
  post-Cut-B runs on the same frames (bit-identity evidence).

**Verification:** headless + oracle green; golden assertions verbatim;
bit-identity diff empty; `jtml-production` is the running default with zero
production-visible change — qml parity verified through
`test/oracle/qml_parity_check.cpp` (the bridge DOES execute runs via
`OptimizerBridge::run()`; the parity instrument is the proof).

---

- [ ] U10. **Cut F — torch include/link hygiene**

**Goal:** Remove the layer-map smell: the unused ATen include and the
PUBLIC torch link — one gated change, no behavior change.

**Requirements:** R4 (seam discipline)

**Dependencies:** None (any time after U1)

**Files:**
- Modify: `include/compute/CostFunctionManager.h` (delete the unused
  `<ATen/ops/div_native.h>` include at :28), `src/compute/CMakeLists.txt`
  (torch link PUBLIC → PRIVATE)

**Approach:**
- Both moves in one jj change: headless build green + oracle green +
  link-only diff. The design NO-GO on torch in the cost path stands; this
  is hygiene only (the legit consumer `machine_learning_tools.cpp` stays
  inside jtml_compute).

**Execution note:** none (pure hygiene).

**Patterns to follow:** the run's Cut F recipe (angle 04 R2-5).

**Test scenarios:**
- Integration: full build (headless + oracle targets) green after the link
  change; no TU that includes CostFunctionManager.h compiles ATen anymore
  (grep-guard).
- Edge case: `segment_image` still links (the legit torch consumer).

**Verification:** headless + oracle green; link-only diff confirmed (jj
diff shows only the include + link line).

---

## System-Wide Impact

- **Interaction graph:** `Optimize()` is the only algorithm policy among
  three interleaved work kinds (session/GPU state + signal emission stay in
  the manager); the script-driven loop preserves the emit order the UI and
  the counter-derived observation channel (stageText/costCalls/progress)
  depend on. Consumers of the injected-cost lambda converge on
  `BuildGpuCostAdapter` (production runner, oracle twin, z-profile probe).
- **Error propagation:** per-stage errors flow through the manager's
  existing error path; the leaf-destruct error-gating asymmetry is
  transcribed verbatim (flagged to the hygiene pass); a bad cfm_index fails
  fast rather than silently skipping.
- **State lifecycle risks:** cumulative-budget accounting is load-bearing —
  the script must NOT reset `cost_function_calls_` per stage (only trunk
  resets, branch/leaf accumulate, per the verified current behavior);
  per-repeat re-seed must read the CURRENT optimum, not a captured one.
- **API surface parity:** `OptimizerRunLaunch` gains one additive (defaulted)
  Options field; `DirectOptimizer` ctor stays 4-arg with Options defaulted;
  the driver interface, Initialize signature, by-value forwarding, and the 8
  binds never change; qml parity verified through
  `test/oracle/qml_parity_check.cpp` (the QML bridge executes runs via
  `OptimizerBridge::run()` — the plan must not claim otherwise).
- **Integration coverage:** the multi-stage oracle (U6) is the cross-layer
  proof that the flat-3000 oracle cannot give (caps, per-stage dilation,
  sym_trap, seed chaining); the bit-identity diff in U9 is the relocation
  proof.
- **Unchanged invariants:** the driver seam, the GPU cost path (no kernel
  edits except U4's one-line index fix), the IoU ≥ 0.85 appearance gate,
  the z-gap banded-informational doctrine, the golden assertions, and the
  wizard-owned regions of `CostFunctionManager.*` (two documented
  exceptions: the one-line stage-guard fix in U2 + the minimal `getStage()`
  accessor in U1, both flagged).

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Re-baseline magnitude unknown (GPU-dependent) | Direction pre-registered (z toward fem.jts, gap shrinks); magnitude recorded, never asserted; single-variable rule keeps the delta interpretable |
| Wizard "DO NOT EDIT" regions in `CostFunctionManager.*` | Two documented exceptions (the one-line stage-guard fix + the minimal `getStage()` accessor), both flagged; `DeriveStageCostParams` mirrors the manager's scan instead; `cfm_index` becomes the stage source of truth |
| Stage-guard fix blast radius | U1 accessor pin + full headless/oracle suite; the CFM's `stage_` is largely ignored downstream (research-verified) |
| Cut B drift during relocation | Four lineage invariants asserted in the multi-stage oracle (U6); golden assertions verbatim; bit-identity pose→score diff |
| Cumulative budget "fix temptation" | Learning-pinned: 20/25/30/35k is authoritative; per-stage reset would break the caps gate |
| Oracle runs need the GPU (3090) | Oracle-labeled targets run explicitly on the GPU machine (existing convention); no oracle assertion in headless CI |
| Line-number drift vs the synthesis | Research pass re-verified all landmarks at current lines; re-grep before editing |
| Bit-identity claim flakiness | Diff the recorded pose→score sequence, not prose; int-atomic path is deterministic (verified) |
| Scope creep into perf/variant work | Scope Boundaries + Deferred-to-Follow-Up; the variant slot ships empty-by-default (R9) |

---

## Documentation / Operational Notes

- `test/golden/baseline.json` gains the re-baseline event (U4) and the
  probe/oracle records (U5/U6) — each is a versioned data event, not a
  prose edit.
- The convergence-tradeoff design note lives in the stage-script TU (U7):
  cover is a stage-loop property, dropped across stages by design; `Options`
  carries no cover knob; variant docs must not claim global-cover guarantees
  for the stage sequence.
- The run's dilation doc reconciliation (6/3/1-vs-6/4/1-vs-lineage) is
  deliberately NOT resolved in this plan — the probe (U5) produces the data
  that later reconciles `golden_oracle.org`.
- Per-session convention: one jj change per logical unit/fix; no interactive
  pagers.

---

## Sources & References

- **Origin document:** [docs/brainstorms/2026-08-12-optimizer-path-requirements.md](docs/brainstorms/2026-08-12-optimizer-path-requirements.md)
- **Input document:** [docs/brainstorms/2026-08-12-optimizer-path-brainstorm-input.md](docs/brainstorms/2026-08-12-optimizer-path-brainstorm-input.md)
- **Normative research base:** `.panoptes/optimizer-deep-dive/synthesis.org` (final; angles 01–06)
- Related code: `src/coordinator/optimizer_manager.cpp`, `include/coordinator/optimizer_run_driver.h`, `include/domain/direct_optimizer.h`, `test/oracle/oracle_test.cpp`, `test/golden/baseline.json`
- Related plans: [docs/plans/2026-08-11-006-refactor-shared-vm-layer-extraction-plan.md](docs/plans/2026-08-11-006-refactor-shared-vm-layer-extraction-plan.md) (seam handoffs), [docs/plans/2026-08-07-001-refactor-testability-mvvm-plan.md](docs/plans/2026-08-07-001-refactor-testability-mvvm-plan.md) (U6 oracle conventions)
- Institutional learnings: `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md`, `docs/solutions/conventions/jtml-layered-lib-split-2026-08-08.md`, `docs/solutions/conventions/jtml-shared-vm-layer-2026-08-11.md`, `docs/solutions/logic-errors/cost-function-parameter-double-truncation-2026-08-08.md`, `docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md`
