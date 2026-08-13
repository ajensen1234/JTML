---
title: "SymTrap costCalls pin wrong twice: the outer guard wraps trunk AND branches (engine truth: 0)"
date: 2026-08-12
category: logic-errors
module: jtml_coordinator
problem_type: logic_error
component: testing_framework
symptoms:
  - "panoptes research synthesis pinned SymTrap costCalls() at 20000 (leaf skipped), assuming a trunk-only drive"
  - "a 5-persona document review 'verified' and corrected the pin to 30000 (trunk + 2x branch), reading the branch block but missing the outer guard"
  - "engine truth: costCalls() == 0 under SymTrap — the if(!sym_trap_call) guard at optimizer_manager.cpp:927 wraps trunk AND branches"
  - "orientationSymTrapUpdated relay count is 61 (60 sweep + 1 restore emit), not 60"
  - "end_frame_index_ uninitialized on the SymTrap path (UB read; the tibia-after-femur oracle was its first-ever executor)"
root_cause: logic_error
resolution_type: test_fix
severity: medium
tags: [sym-trap, cost-calls, pin-first, characterization, oracle, optimizer-manager, outer-guard, bit-identity]
related_components:
  - testing_framework
  - tooling
---

# SymTrap costCalls pin wrong twice: the outer guard wraps trunk AND branches

## Problem

The multi-stage oracle's SymTrap costCalls pin was specified **wrong twice** before the engine's control flow was actually read. Both wrong pins would have forced a deterministic failure of a multi-minute GPU oracle run — or worse, tempted a "fix" to make the engine match the pin. The correct value, measured on the engine, is **0**.

## Symptoms

- The research synthesis (`.panoptes/optimizer-deep-dive/synthesis.org`, Finding 2) pinned `costCalls() == 20000` under SymTrap ("the leaf is skipped, so costCalls() lands at 20000, NOT 35000") — it assumed the tibia pass drives trunk only.
- A 5-persona document review of the plan then "verified" the pin as **30000** ("trunk 20000 + 2x5000 branch run under SymTrap; only the leaf SEARCH is skipped") — it read the branch block in isolation.
- The engine truth (U6 brace analysis + measurement): `costCalls() == 0`, `stageText` stays `"Idle"`, the final `UpdateDisplay` never fires, 60 uncounted analysis evals run.

## What Didn't Work

- **Trusting the synthesis's pin**: it derived the value from a mental model of the stage loop ("SymTrap skips the leaf search") without checking which sections the `sym_trap_call` flag actually gates.
- **Trusting the review's "verification"**: the document-review personas read the trunk block (unconditional) and the branch block (gated only on `enable_branch_`) and concluded branches run — they never looked for an *outer* guard. The `if (!sym_trap_call)` at `optimizer_manager.cpp:927` wraps **both** sections; only the leaf-CFM init (:1111–1132), `CalculateSymTrap` (:1134–1136, 60 evals via `EvaluateCostFunctionAtPoint` — which does NOT increment `cost_function_calls_`), the epilogue, and the early return at :1201 (skipping the final `UpdateDisplay`) execute.
- **The relay pin was also off by one**: `CalculateSymTrap` emits `onUpdateOrientationSymTrap` 60 times in the loop + 1 unconditional restore emit = **61**, not 60.

## Solution

**Pin-first characterization (U6, supervisor-approved):** run 1 of `jtml.oracle_multistage` recorded what the engine *actually* does before asserting anything:

- assert `costCalls() == 0` under SymTrap (recorded in `test/golden/baseline.json` as the pre-Cut-B characterization, `symtrap_relay_count: 61`, `tibia_cost_calls: 0`);
- assert the relay count ≥ 60 (61 on the happy path; a count of 0 catches `CalculateSymTrap`'s zero-pose early return at ~:1304–1308);
- record `stageText` stays `"Idle"` (the :1201 early return skips the final `UpdateDisplay`);
- plan for the side effects: `Results.csv` / `Results.xyz` / `Results2D.xy` written into the process CWD + ~5 s of sleeps.

Then the container work consumed the characterization: `BuildStageScript` maps Sym_Trap → `[{Leaf, repeat=0}]` (U7), and the script-driven loop (U9) preserves the behavior **bit-identically** — the multistage oracle passes unchanged through the relocation, which is the bit-identity proof.

One production fix rode along: `end_frame_index_` was uninitialized on the SymTrap directive path (UB read in `create_image_indices`); the tibia-after-femur oracle was the path's first-ever executor, so the one-line init landed with a comment (plan 008 U6).

## Why This Works

The `sym_trap_call` flag gates the *entire search* — trunk, branches, and leaf search alike — because the tibia SymTrap pass is an **analysis** (sweep 60 rotations around the femur-recovered pose at leaf dilation), not a search. Any pin derived from "which inner block is skipped" is structurally fragile: the correct method is to read the outermost control-flow guard first, then characterize the engine by running it. The characterization-first protocol (run 1 records, later runs enforce — the run's pin-first doctrine) converts "what should the engine do" into "what the engine does", and the bit-identity target becomes "preserve today's behavior", never an assumed number.

## Prevention

- **Read the outer control-flow guards before pinning inner blocks.** A pin derived from a section read is a hypothesis; a pin derived from the enclosing `if` is a fact. Brace-analysis the region before asserting any call-count expectation.
- **Characterize-then-assert**: run 1 of any new instrument records the engine's real values (and enforces nothing); later runs enforce. Wrong pins surface as recorded characterizations, not deterministic failures.
- **When a pin is corrected, record the correction trail in the code comment** (the plan's execution notes carry "both the synthesis's 20000 and the review's 30000 readings were wrong — the outer guard was missed").
- **The bit-identity target is "preserve today's behavior"**, not a number from a spec — the U9 relocation was proven by the U6 oracle passing *unchanged*.
- **Relay counts: count the restore emit.** Loop emits + unconditional trailing emits (61 = 60 + 1) are the norm in this codebase; prefer `>= n` over `== n` when only the zero-count early return is the failure to catch.

## Related Issues

- `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md` — the cumulative-budget doctrine this case extends ("20k → 25k → 30k → 35k... verified against code, not prose; do NOT fix to per-stage caps").
- `docs/solutions/logic-errors/jtml-heatmap-guard-allocator-preconditions-2026-08-12.md` — the uninitialized-member class (`end_frame_index_` is the same family: a guard's precondition must itself be initialized).
- **Stale contradiction (research artifact, not a solution doc):** `.panoptes/optimizer-deep-dive/synthesis.org` still carries the wrong "20000" SymTrap pin — any future reader must use `baseline.json`'s `oracle_multistage` characterization (0) instead.
- Plan: `docs/plans/2026-08-12-008-refactor-optimizer-graph-container-plan.md` (Execution Notes, SymTrap bullet) — the in-repo correction trail.
