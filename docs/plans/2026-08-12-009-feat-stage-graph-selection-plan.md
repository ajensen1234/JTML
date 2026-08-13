---
title: "Stage-Graph Selection: the run shape pickable on the UI (model + viewmodel first)"
type: feat
status: active
date: 2026-08-12
origin: docs/optimizer-stage-graphs.md
---

# Stage-Graph Selection: the run shape pickable on the UI

## Overview

Plan 008 made the optimizer run shape **data** (the StageScript / named-graph
registry). This plan makes the run shape **selectable**: named stage graphs
(`jta::ListStageGraphs()`) become a UI choice, mirroring the per-stage
cost-function picker (`SettingsBridge`'s `trunkCostFunctions` /
`trunkCostFunctionIndex` pattern). Per the owner: exposing the **model and
viewmodel** of the selection is the must-have; the **view wiring** (the
actual dropdown) is parked as follow-up work.

The selection is run-level (one graph per run — the graph IS the shape), not
per-stage like the cost-function choice.

## Problem Frame

Today the run shape is chosen indirectly: the settings fields
(`enable_branch_`, `number_branches`, budgets, ranges) are transcribed into a
script by `BuildStageScript`, and the named-graph registry
(`StageGraphByName`) is validated + unit-tested but **not consumed by the
production run**. The owner wants the opposite mental model: *build a bunch
of stages for different scenarios, then pick the one you want* — like the
cost-function list. The cost-function picker precedent (verified):
`SettingsBridge` exposes `QStringList trunkCostFunctions` (CONSTANT — the
registry) + `int trunkCostFunctionIndex` (WRITE, `NOTIFY settingsEdited`),
and the widgets side uses a per-stage `cost_function_listWidget`
(`settings_control.cpp:43,484-576`).

## Requirements Trace

- R2 (origin requirements doc): named registry — graphs are C++-typed named
  data; the manifest/UI references by name.
- The owner's feature intent: the stage graph is selectable on the UI
  (mirroring the cost-function picker); the model/viewmodel surface is the
  must-have deliverable; the view wiring is parked.
- AE-style acceptance: selecting `jtml-production` on the VM surface yields
  the exact same run shape as the default settings-built script (the parity
  pin); an unknown graph name fails fast, never silently falls back.

---

## Scope Boundaries

- **No view wiring in this plan**: the QML ComboBox (settings panel) and the
  widgets picker (settings_control parity) are follow-up work — the VM
  surface is stable first.
- Run-level selection only (one graph per run), not per-stage.
- Graphs are **code-registered data** (the registry) — no UI for editing or
  authoring graphs (that stays a code change, per R2/A1).
- No change to the `Sym_Trap` directive semantics: a named graph wins over
  the directive's shape; the directive still selects frames. (Documented
  precedence: graph > directive for stages.)
- No graph-persistence schema work beyond what `SaveOptimizerSettings`
  already provides.

### Deferred to Follow-Up Work

- **View wiring**: the QML settings-panel ComboBox (bind to
  `stageGraphs`/`stageGraphIndex`) and the widgets settings_control list
  widget (parity with the cost-function list widget) — one follow-up change
  once the VM surface lands.
- Optional: a `stageGraphDescription`/tooltip surface (per-graph prose) if
  the picker needs context beyond the name.

---

## Context & Research

### Relevant Code and Patterns

- `include/coordinator/optimizer_stage_script.h` — `StageGraph {name,
  stages}`, `ListStageGraphs()`, `StageGraphByName(name)` (fail-fast on
  unknown + reserved stub names), `StageScript`/`StageSpec`.
- `src/coordinator/optimizer_manager.cpp:215-227` — the Initialize script
  build (try/catch `std::invalid_argument` → `error_message` + failed
  Initialize); the named-graph branch slots directly into this block.
- `src/app/experimental/SettingsBridge.h:114-123` — the cost-function picker
  pattern to mirror (`QStringList ... CONSTANT` + `int ...Index` WRITE
  NOTIFY `settingsEdited`); `src/app/experimental/SettingsBridge.cpp:382-401`
  (the index getter/setter impl), `:50-90` (save/load/reset — `save()`
  already calls `SaveOptimizerSettings(optimizer_)`, so a new
  `OptimizerSettings` field persists for free).
- `include/services/optimizer_settings.h` — the model; carried by value in
  `OptimizerRunLaunch` (additive field, fake-driver suite compile-unchanged).
- `test/unit/test_stage_script.cpp` — the builder/registry pins; the parity
  pin's home.
- `test/unit/experimental_settings_test.cpp` — the bridge parity tests
  (5 TEST_CASEs); the VM surface tests' home.

### Institutional Learnings

- `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md` —
  cumulative-budget doctrine; pin-first; direct-compile pattern.
- `docs/solutions/conventions/jtml-shared-vm-layer-2026-08-11.md` — the
  bridges are thin pass-throughs; all behavior lives in the seams; the
  SettingsBridge holds one `OptimizerSettings` + three CFMs.
- `docs/solutions/logic-errors/jtml-symtrap-outer-guard-pin-2026-08-12.md` —
  characterize-then-assert; the multistage oracle is the bit-identity gate.

### External References

- None needed — the pattern to mirror is in-repo (the cost-function picker).

---

## Key Technical Decisions

- **Named graph wins over the builder**: when `stage_graph_name` is
  non-empty, `Initialize` sets `stage_script_` from
  `StageGraphByName(name).stages`; otherwise `BuildStageScript` (the
  current path — backward compatible, "" default). The graph IS the shape;
  the settings fields are the "custom" path.
- **Fail fast on unknown names** (no silent fallback): an unknown
  `stage_graph_name` fails Initialize through the existing error path —
  a stale persisted name must surface, not silently run the wrong shape.
- **Index sentinel**: `stageGraphIndex == -1` means "" (the settings-built
  shape); otherwise the index into `stageGraphs`. The QML view maps -1 to a
  "custom/settings" entry when wired.
- **Persistence rides the existing path**: `SaveOptimizerSettings(optimizer_)`
  already persists `OptimizerSettings` — the new field flows through both
  apps' registry path with zero new schema.
- **Parity pin**: `StageGraphByName("jtml-production").stages ==
  BuildStageScript(default settings, "All")` — guards registry-vs-builder
  drift (they must stay equivalent for the default shape).

---

## Implementation Units

- [ ] U1. **Model: `OptimizerSettings.stage_graph_name`**

**Goal:** The selection's model field: a named graph for the run, "" =
settings-built (the default).

**Requirements:** R2; the owner's model must-have

**Dependencies:** None

**Files:**
- Modify: `include/services/optimizer_settings.h` (add `QString
  stage_graph_name;` + `#include <QString>`), `src/services/optimizer_settings.cpp`
  (ctor default: clear)

**Approach:**
- The exact edit shape was drafted during investigation: a `QString
  stage_graph_name` field with the comment "the named stage graph for this
  run: "" = the settings-built script; otherwise the registered graph name
  (fail-fast on unknown via the manager's Initialize)". Ctor sets it empty.
- Additive field: `OptimizerRunLaunch` carries `OptimizerSettings` by value —
  no launch changes; the fake-driver suite compiles unchanged.

**Execution note:** none (pure model addition).

**Patterns to follow:** the existing `OptimizerSettings` field style
(plain members + ctor defaults in `src/services/optimizer_settings.cpp`).

**Test scenarios:**
- Happy path: default-constructed `OptimizerSettings` has
  `stage_graph_name.isEmpty() == true`.
- Edge case: a round-tripped settings object (copy) preserves a set name.

**Verification:** headless suite green (existing settings consumers
unchanged).

---

- [ ] U2. **Manager consumption: named graph wins + parity pin**

**Goal:** `Initialize` runs the named graph when set (fail-fast on unknown),
and the registry-vs-builder parity is pinned.

**Requirements:** R2; AE (jtml-production ≡ settings-built)

**Dependencies:** U1; plan-008 U7 (registry) + U9 (loop)

**Files:**
- Modify: `src/coordinator/optimizer_manager.cpp` (the Initialize script
  build block ~:215-227)
- Test: `test/unit/test_stage_script.cpp` (the parity pin)

**Approach:**
- In the existing try block: if `optimizer_settings_.stage_graph_name`
  is empty → `BuildStageScript` (unchanged); else →
  `stage_script_ = jta::StageGraphByName(name).stages` (the registry throws
  `std::invalid_argument` on unknown/reserved names — caught by the
  existing handler → `error_message` + failed Initialize; no silent
  fallback).
- Precedence documented in the comment: the graph is the shape; the
  directive still selects frames; under `Sym_Trap` a named graph runs its
  own stages (the leaf-only script applies only to the settings-built
  path).
- Parity pin: `StageGraphByName("jtml-production").stages ==
  BuildStageScript(defaults, "All")` — the builder's normal-directive shape
  with default settings must equal the registered production graph (both
  derive from `settings_constants.h`).

**Execution note:** characterization-first — the parity pin is the spec;
if the pin fails, the registry or the builder drifted from
`settings_constants.h` and the drift is the bug.

**Patterns to follow:** the existing try/catch script-build block
(`optimizer_manager.cpp:215-227`); the U7 builder pins.

**Test scenarios:**
- Happy path: `Covers AE.` — the parity pin holds (jtml-production ≡
  default settings-built shape, stage-for-stage: kind/range/budget/repeat/
  cfm_index).
- Edge case: an empty name takes the builder path (bit-identical to today).
- Error path: an unknown name fails Initialize with the existing error
  path (a unit-level probe with a fake cost or the manager's error
  message; the oracle suite verifies no regression).
- Integration: the multistage oracle (production shape, `stage_graph_name`
  empty) passes UNCHANGED — the run is bit-identical to plan 008's record.

**Verification:** headless green (incl. the new parity pin); oracle suite
green with the default (empty) name.

---

- [ ] U3. **ViewModel: SettingsBridge stage-graph surface**

**Goal:** The QML viewmodel surface: the graph name list + the selection
index, mirroring the cost-function picker.

**Requirements:** the owner's viewmodel must-have

**Dependencies:** U1 (the field), U2 (consumption exists), plan-008 U7 (the
registry — the app layer may include coordinator headers)

**Files:**
- Modify: `src/app/experimental/SettingsBridge.h` (the two Q_PROPERTYs +
  accessor declarations), `src/app/experimental/SettingsBridge.cpp` (the
  three accessors; `#include "coordinator/optimizer_stage_script.h"`)
- Test: `test/unit/experimental_settings_test.cpp` (the bridge test target)

**Approach:**
- `Q_PROPERTY(QStringList stageGraphs READ stageGraphs CONSTANT)` —
  `jta::ListStageGraphs()` names (the registry is compile-time data);
- `Q_PROPERTY(int stageGraphIndex READ stageGraphIndex WRITE
  setStageGraphIndex NOTIFY settingsEdited)` — index of
  `optimizer_.stage_graph_name` in the list, **-1 = "" (settings-built)**;
  the setter writes the name (or clears for -1), no-ops on no change,
  marks dirty + emits `settingsEdited` (the bridge's session-edit
  contract);
- `reset()` clears it automatically (fresh `OptimizerSettings`).
- Persistence: free — `save()` already calls
  `SaveOptimizerSettings(optimizer_)`; `load()` restores the name with the
  other settings.

**Execution note:** test-first for the index mapping (the -1 sentinel is
the contract).

**Patterns to follow:** the cost-function picker surface
(`SettingsBridge.h:114-123` + `SettingsBridge.cpp:382-401` — the
`costFunctionNames`/`costFunctionIndex`/`setCostFunctionIndex` helpers and
the dirty-marking setter pattern).

**Test scenarios:**
- Happy path: `stageGraphs()` contains `"jtml-production"`; the default
  index is -1 (empty name — the settings-built default).
- Happy path: `setStageGraphIndex(0)` → `stageGraphIndex() == 0`,
  `optimizer_.stage_graph_name == "jtml-production"`, dirty is set, and
  `settingsEdited` fired.
- Edge case: `setStageGraphIndex(-1)` clears the name (back to
  settings-built).
- Edge case: out-of-range index (≥ list size) is a no-op (name unchanged,
  not dirty).
- Integration: `save()`/`load()` round-trips the name through the settings
  registry (the existing bridge fixture pattern).

**Verification:** headless green (the bridge test target); the existing
bridge/QML parity tests unchanged.

---

## System-Wide Impact

- **Interaction graph:** `Initialize`'s script build (U2) is the only
  consumer of the new field; the SettingsBridge setters follow the
  established dirty/settingsEdited contract; the QML app's run launch
  already carries `OptimizerSettings` by value through the controller.
- **Error propagation:** unknown graph names fail Initialize (existing
  error path) — never a silent fallback to the settings-built shape.
- **State lifecycle:** the name is session state in the bridge (explicit
  save), persisted with the other optimizer settings; a reset clears it.
- **API surface parity:** `OptimizerSettings` gains one additive field
  (by-value copies everywhere — no signature changes); the bridge gains two
  properties (additive); the widgets side is untouched until the follow-up
  view wiring.
- **Unchanged invariants:** the driver seam, the stage loop, the oracle
  gates, and the default (empty name) run shape — bit-identical until a
  name is set.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Registry-vs-builder drift (jtml-production diverges from the default settings shape) | The U2 parity pin asserts equality stage-for-stage; drift fails the pin, not production |
| A persisted name goes stale (graph removed/renamed) | Fail-fast Initialize with the existing error message — the user sees the stale name, never a silent wrong shape |
| The -1 sentinel confuses the future QML ComboBox | Documented in the header + the guide; the view maps -1 to a "custom (settings)" entry |
| Named graph + SymTrap directive interaction | Documented precedence (graph wins for stages; directive selects frames) + a unit test on the manager error/script selection |
| Bridge tests need the registry (coordinator include in the app layer) | The app already links coordinator; the test target follows the existing bridge-test fixture pattern |

---

## Documentation / Operational Notes

- `docs/optimizer-stage-graphs.md` — the "honest wiring note" (the registry
  is not consumed by the run yet) flips after U2: the guide gains a
  "selecting a graph" section (settings field + VM surface + the view
  follow-up pointer).
- The graph-selection recipe belongs in the guide's Recipes section once
  the VM lands.

---

## Sources & References

- **Guide:** `docs/optimizer-stage-graphs.md` (the schema + recipes this
  plan's selection surface exposes)
- Related code: `include/coordinator/optimizer_stage_script.h`,
  `src/coordinator/optimizer_manager.cpp` (Initialize script build),
  `src/app/experimental/SettingsBridge.{h,cpp}` (the cost-function picker
  pattern), `include/services/optimizer_settings.h`
- Related plans: [docs/plans/2026-08-12-008-refactor-optimizer-graph-container-plan.md](docs/plans/2026-08-12-008-refactor-optimizer-graph-container-plan.md)
  (U7 registry, U9 loop — the container this plan's picker selects)
