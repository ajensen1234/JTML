---
title: "CostFunctionManager::updateCostFunctionParameterValues is a silent no-op (writes land on a by-value copy)"
date: 2026-08-12
category: logic-errors
module: jtml_compute
problem_type: logic_error
component: tooling
symptoms:
  - "U5 z-profile term-decomposition check showed full DIRECT_DILATION cost differing from (chamfer term + dilated term) by 4650 at dilation 3"
  - "configuring Dilation through updateCostFunctionParameterValues(int...) left the parameter at its registered default"
  - "the overload returns true on its success path even though nothing was written — no error, no warning"
  - "all three overloads (double/int/bool) share the defect: the get*Parameters() getters return the parameter vectors BY VALUE"
  - "production (SettingsBridge.cpp:586, settings_control.cpp:1056-1084) and the U6/U7 oracles are unaffected — they use setIntParameterValue on the ACTIVE class"
root_cause: wrong_api
resolution_type: code_fix
severity: medium
tags: [cost-function, parameter, silent-noop, by-value-copy, pbt, hegel, wizard-region]
related_components:
  - testing
---

# CostFunctionManager::updateCostFunctionParameterValues is a silent no-op

## Problem

The `updateCostFunctionParameterValues` family in `src/compute/CostFunctionManager.cpp` (double :162–184, int :185–207, bool :208–226) never writes through to the stored parameter. Each overload calls the corresponding `get*Parameters()` getter on the cost-function class, which returns `std::vector<Parameter<T>>` **by value** (`src/compute/CostFunction.cpp:104–112`; declared `include/compute/CostFunction.h:44–46`), then calls `setParameterValue(value)` on the returned element — a temporary copy destroyed at the end of the full expression. The overload returns `true` on its success path as if the write happened.

## Symptoms

- Setting `Dilation` via `updateCostFunctionParameterValues(..., "Dilation", v)` leaves the parameter at the registered default (6) for any `v != 6`; the call returns `true`.
- The U5 z-profile term-decomposition check (full cost == chamfer term + dilated term from the same render) diverged at dilation 3 with diff = 4650 — the probe initially configured dilation through this overload, so the "dilated term" rendered at the default 6 while the composed reference used 3 (`test/oracle/z_profile_test.cpp:682–695`, diagnostic-only check).
- `test/oracle/oracle_test.cpp:239–240` passes `Dilation = 6` — equal to the registered default — so the no-op is invisible there; it "works" only by coincidence (recorded in `test/golden/baseline.json`).
- `test/oracle/multistage_oracle_test.cpp:323–331` explicitly documents the finding and uses the production path instead.

## What Didn't Work

- The first probe run read the dilation-3 divergence as a **cost-engine problem** (candidates: the dilated-term metric, the int-atomic reduction at higher dilations, the term decomposition itself). That track was wrong — the engine was fine; the harness had configured dilation through the no-op overload.
- **Weakening the check** ("assert within a tolerance") to absorb diff = 4650 was rejected — consistent with this family's doctrine (the double-truncation doc): a failing consistency invariant is a genuine defect somewhere; find it, don't bake it in.

## Solution

Root cause identified: **mutating a by-value getter result**. The production path never hits it — `SettingsBridge::setDilation` writes via `setIntParameterValue` on the **active** cost-function class (`src/app/experimental/SettingsBridge.cpp:586`; `src/view/settings_control.cpp:1056–1084`, :1203–1205 — the widgets path), which mutates the owned member; the U6/U7 oracles use the same production path. The probe switched to that path and the composition check now shows diff == 0 (or ~7e-12 float jitter) at every sweep point.

The file carries the wizard DO-NOT-EDIT banner (`CostFunctionManager.cpp:8–10`), so the finding was **recorded for the hygiene pass, not touched**. The hygiene fix has two defensible shapes: (a) make the overloads delegate to `set{Int,Double,Bool}ParameterValue` on the named class (write-through), or (b) delete the family if no consumer needs it (grep before choosing — current consumers: oracle fixtures that "work" by default-coincidence). Production code needs no change.

## Why This Works

The getters return the parameter containers by value (`std::vector<Parameter<int>> getIntParameters()`), so `available_cost_functions_[i].getIntParameters()[j].setParameterValue(v)` mutates a temporary. The active-class setters (`setIntParameterValue(name, value)`) walk the owned vector and mutate the real `Parameter<T>` — the write lands where the cost kernel reads it. The failure mode is a silent landmine: any future consumer of the update family believes parameters were persisted.

## Prevention

- **Add a hegel PBT set-then-get round-trip invariant over the update family** (the exact invariant shape that caught the sibling truncation bug — `jtml.cost_function_props` already covers the typed setters): `updateCostFunctionParameterValues(name, param, v)` then `getIntParameterValue(name, got)` must yield `got == v`. A property test with draws spanning negatives and non-default values would fail immediately against the current no-op.
- **Prefer the active-class setters** (`getActiveCostFunctionClass()->set*ParameterValue`) — the only path the cost kernels observe.
- **Before mutating a "getter" result, check whether the getter returns by value** — `std::vector<Parameter<T>>` by value is a copy; mutation is a no-op by construction.
- **Grep before reuse**: `updateCostFunctionParameterValues` consumers today either pass the default value (invisible) or were switched away (the oracles) — treat any new consumer as suspect until the family is fixed or deleted.

## Related Issues

- `docs/solutions/logic-errors/cost-function-parameter-double-truncation-2026-08-08.md` — the sibling silent-corruption bug in the same registry (`Parameter<double>` stored as `int`); same prevention family (bit-exact typed PBT round-trip, never weaken the invariant). Moderate overlap — same area, different root cause; consolidation candidate for a future refresh.
- `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md` — the PBT-invariant doctrine home.
- `docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md` — hegel PBT recipes (`jtml.cost_function_props` target).
- `docs/solutions/logic-errors/jtml-symtrap-outer-guard-pin-2026-08-12.md` — the other wizard-region episode this session (one in-place exception documented; engine-truth-over-assumption doctrine).
- Plan: `docs/plans/2026-08-12-008-refactor-optimizer-graph-container-plan.md` Execution Notes — the hygiene-pass item carrying this finding.
