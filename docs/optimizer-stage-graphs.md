# Creating Custom Stage Graphs (the optimizer run shape)

The optimizer no longer has a hard-coded trunk/branch/leaf shape. The run is a
**stage graph**: an ordered list of stages (a *StageScript*), each describing
one DIRECT search — its cost function + parameters, its search box, its
budget, and how many times it runs. The manager's `Optimize()` loop iterates
the script (`src/coordinator/optimizer_manager.cpp` — `for (const
jta::StageSpec& spec : stage_script_)`). This guide tells you how to read a
graph, change the run shape, and register your own named graph.

The production shape, as data (`jtml-production`, the v1 graph):

```
[Trunk,  range (35,35,35,35,35,35),       budget 20000, repeat 1, cfm 0]
[Branch, range (15,15,25,25,25,25),       budget  5000, repeat 2, cfm 1]
[Leaf,   range (3,3,15,3,3,3),            budget  5000, repeat 1, cfm 2]
cumulative caps: 20000 / 25000 / 30000 / 35000
```

## The StageSpec fields

`include/coordinator/optimizer_stage_script.h` — `struct StageSpec { kind,
range, budget, repeat, cfm_index }`:

| Field | Meaning | Notes |
|---|---|---|
| `kind` | The stage's search flavor: `Trunk`, `Branch`, or `Leaf` | `StageKind` enum; a future polish kind is a reserved stub. The kind selects the per-kind init/dilate/emit/search gating in the loop — keep the lineage ordering (trunk first, leaf last) unless you know why you're changing it |
| `range` | The search box, physical units (mm, mm, mm, deg, deg, deg) | Applied per repeat via `SetSearchRange`. Ranges are **seed-relative**: each stage re-seeds from the previous stage's optimum, then searches ±range around it |
| `budget` | Evaluations for this stage | **Cumulative semantics**: only the FIRST stage resets the running counter; every later stage *adds* (`budget_ += spec.budget`). The oracle's caps gate (costCalls on 20/25/30/35k) depends on this — never "fix" a stage to reset the counter |
| `repeat` | How many times the stage's `RunDirectStage` runs | Branch = your branch count (re-seed from the *current* optimum per repeat). `repeat = 0` is the **no-search leaf** pattern (the SymTrap analysis pass: init + dilate + emit + `CalculateSymTrap`, zero DIRECT evals). Normal stages use `repeat >= 1` |
| `cfm_index` | Which `CostFunctionManager` the stage uses: 0 → trunk, 1 → branch, 2 → leaf | The manager owns exactly three CFMs; a bad index fails fast. The CFM **is** the cost variant + its parameters — `cfm_index` + the CFM's active cost function + its parameter registry fully determine the cost |

## Rules that are load-bearing (do not break them)

- **Cumulative budget**: the first stage resets the counter; everything after
  accumulates. `CumulativeStageCaps(script)` tells you the caps a run will
  land on — assert them in tests.
- **Seed chaining**: every stage re-seeds from the current optimum; branch
  repeats re-seed *per repeat* from the latest recovery. The oracle asserts
  the recovered-pose *sequence*, not just the final pose.
- **Group-once dilation**: a branch group's init + dilate + emit happens once;
  the repeats only re-seed and search.
- **Cover is intentionally dropped across stages** (the convergence
  tradeoff): each stage's box does NOT cover the previous stage's domain
  (branch ⊂ trunk, leaf ⊂ branch). Do not set branch/leaf ranges equal to the
  trunk range — restoring cover "would negate the explore and exploit
  structure" and requires its own pin. The design note lives in
  `optimizer_stage_script.h`.
- **The channel never says "Leaf"**: the UI/observation layer reports
  `Trunk`, `Branch 1`, `Branch 2`, `Extra Z-Translation` (the leaf), and
  `Finished`.
- **Per-stage cost parameters live in the CFM registry** (dilation, weights,
  etc.), set via `getActiveCostFunctionClass()->setIntParameterValue(...)` —
  the production path (`SettingsBridge.cpp:586`). Do NOT use the
  `updateCostFunctionParameterValues` family — it is a silent no-op
  (documented in `docs/solutions/logic-errors/jtml-cost-function-update-parameter-int-noop-2026-08-12.md`).

## Changing the run shape TODAY (the settings path)

The manager builds the run script in `Initialize` from the settings +
directive: `stage_script_ = jta::BuildStageScript(settings, directive)`
(`src/coordinator/optimizer_manager.cpp:220`). `BuildStageScript` transcribes
the `OptimizerSettings` flags verbatim:

- `enable_branch_` + `number_branches` → the Branch stage's presence and
  `repeat` (0 or absent → no branch stage);
- `enable_leaf_` → the Leaf stage's presence;
- the budgets and ranges come straight from the settings.

So the **practical way to experiment** is to change the settings — the UI
(widgets or QML settings), or the defaults in
`include/domain/settings_constants.h` — and the run shape follows. Examples:

- 2 branches → 3 branches: raise `number_branches` (the branch `repeat`
  follows).
- Trunk-only run: `enable_branch_ = false; enable_leaf_ = false` →
  `[Trunk]` with caps `[20000]`.
- SymTrap (tibia trap analysis): the `"Sym_Trap"` directive yields the
  leaf-only `[{Leaf, repeat=0}]` script — no search, costCalls stays 0.

## Registering a NAMED graph (the registry)

The named-graph registry (`include/coordinator/optimizer_stage_script.h` +
`src/coordinator/optimizer_stage_script.cpp`) is the "configuration library":
`StageGraph { name, stages }`, `ListStageGraphs()`, `StageGraphByName(name)`.
`jtml-production` is the only registered graph; requesting a reserved stub
name (polish, ML-initializer, flood-direct-jta) or an unknown name fails fast
with `std::invalid_argument`.

To add your own, mirror the `JtmlProductionGraph()` factory
(`src/coordinator/optimizer_stage_script.cpp:70`): write a factory returning
`StageGraph{"my-graph", {...}}` and add it to `ListStageGraphs()`. Example —
the lineage's flood shape (3 branches, the 50k/15k×3/50k budget, the
asymmetric 5/5/20/5/5/5 leaf):

```cpp
StageGraph JtmlFloodShapeGraph() {
    return StageGraph{"flood-direct-jta", {
        {StageKind::Trunk,  Point6D(35,35,35,35,35,35), 50000, 1, 0},
        {StageKind::Branch, Point6D(15,15,25,25,25,25), 15000, 3, 1},
        {StageKind::Leaf,   Point6D(5,5,20,5,5,5),      50000, 1, 2},
    }};
}
// ListStageGraphs: return {JtmlProductionGraph(), JtmlFloodShapeGraph()};
```

**Honest wiring note:** today the manager runs the script from
*settings + directive* (the builder), not from a registry name. A registered
graph is validated by the unit pins (`test/unit/test_stage_script.cpp`:
`CumulativeStageCaps`, builder semantics, fail-fast paths) and
`StageGraphByName`, but the production run does not consume it yet — wiring
the manager to set `stage_script_` from `StageGraphByName` instead of the
builder is a small, pinned change (the loop already consumes `stage_script_`).
Until that lands, treat the registry as the tested, ready-to-wire
configuration surface.

## Recipes

1. **1-stage trunk-only graph** (flat run): `[Trunk, (35)^6, 20000, 1, cfm 0]`
   — caps `[20000]`. Equivalent settings path: disable branch + leaf.
2. **More refinement after the leaf** (a second leaf-style pass): append
   `{Leaf, (3,3,15,3,3,3), 5000, 1, cfm 2}` — but note each extra stage adds
   to the cumulative caps, and the UI's stage labels will show the extra
   `Extra Z-Translation` pass. Keep the cover doctrine in mind (a smaller
   range is refinement, an equal range is a re-search needing a pin).
3. **Different per-stage cost**: point a stage at a different `cfm_index`
   (same CFM, different cost function via `setActiveCostFunction`, or
   different parameters) — the cost variant + dilation are data, not code.
4. **What NOT to do**: negative budgets or a bad `cfm_index` (fail fast by
   design); `repeat = 0` on a Trunk/Branch stage (the no-search pattern is
   documented for the leaf/SymTrap only); equal-or-larger ranges downstream
   (cover restoration — pin it first).

## Verifying your graph

- Unit pins: `test/unit/test_stage_script.cpp` (+ `_properties.cpp`) — builder
  mapping, caps, fail-fast, PBT invariants.
- The bit-identity gate: `jtml.oracle_multistage` (production shape, ~15 s on
  the GPU) — any change that alters the production shape's behavior must keep
  the oracle green.
- `pixi run test` for the headless suite; the oracle targets run via
  `pixi run ctest --test-dir .build -R "oracle_multistage|qml_parity"`.

## References

- Schema + design notes: `include/coordinator/optimizer_stage_script.h`
- Registry + builders: `src/coordinator/optimizer_stage_script.cpp`
- The consuming loop: `src/coordinator/optimizer_manager.cpp` (the
  script-driven stage loop, ~:944+)
- Plan: `docs/plans/2026-08-12-008-refactor-optimizer-graph-container-plan.md`
  (U7 schema, U9 PoC, execution notes)
- Related learnings: `docs/solutions/logic-errors/jtml-symtrap-outer-guard-pin-2026-08-12.md`,
  `docs/solutions/logic-errors/jtml-cost-function-update-parameter-int-noop-2026-08-12.md`
