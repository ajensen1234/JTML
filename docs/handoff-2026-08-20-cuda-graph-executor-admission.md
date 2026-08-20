# Handoff: CUDA-Graph Executor Admission and Lifecycle (Plan 012)

**Prepared:** 2026-08-20 — replacement plan for remaining Plan 011 U6–U8; no implementation yet

- **Active plan:** `docs/plans/2026-08-20-012-feat-cuda-graph-executor-admission-plan.md`
- **Do not implement:** Plan 011 U6 as written. 011 U1–U5 stay landed.
- **Requirements:** `docs/brainstorms/2026-08-19-cuda-graph-greedy-evaluation-executor-requirements.org`
- **Planning-completeness learning:** `docs/solutions/workflow-issues/jtml-deepened-unit-not-implementation-ready-2026-08-20.md`
- **Anti-stub (code gate):** `docs/solutions/logic-errors/jtml-cuda-graph-stub-failure-2026-08-19.md`
- **Architecture blueprint:** `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md`
- **Layered-correctness draft:** `docs/solutions/architecture-patterns/graph-tiered-correctness-2026-08-19.md` — still **aspirational** until Plan 012 U6 proves it
- **Historical 011 handoff:** `docs/handoff-2026-08-19-cuda-graph-greedy-evaluation-executor.md` (status only; not the implementation source of truth)

---

## First turn for the next agent

1. Read root `AGENTS.md`.
2. Run `ctx_index` on `docs/` if `jtml-docs` is not indexed; then `ctx_search` for:
   - `"Plan 012 BatchOutcome GraphAdmissionPolicy U12"`
   - `"CaptureCoordinator ForceRelease LeavePoisoned"`
   - `"completeFromPins graph_layer_verdict"`
3. Read Plan 012 **Key Technical Decisions + U1** before any code. U1 is fail-closed admission, not graph launch.
4. `jj st` and `jj log --no-graph -r '@-::@-'`. Docs-only planning/handoff changes should already be checkpointed. Do not mix them with U1 code.
5. `pixi run build`; use `ctest` target filters. Never raw git, cmake, make, or nvcc.

---

## Why 012 exists

A CUDA-aware review of deepened 011 U6 found unresolved **lifecycle/admission design**. Coding that unit would either stub it or ship the wrong R8/R7 semantics. Plan 012 resolves C1–C11 in the plan, then implements them as new U1–U7.

011 U1–U5 remain real (pool, frozen rev-2 fixture, capture probe, persistent workers, monoplane recipe).

---

## What to implement first

**012 U1 only:** typed `BatchOutcome`, default-deny `GraphAdmissionPolicy`, delete the null-recipe `useExecutor=true` override, catch abort/`invalid_argument` in `RunDirectStage` → `OptimizerError`. Do **not** allocate the dummy 8 GiB graph pool. Do **not** start capture, hooks, or U7 rewrite in the same change.

Then U2 (key + `CaptureGeneration`) beside or after U1. Capture coordinator / wrappers are U3.

---

## Known reds (do not “fix” outside their units)

- `jtml.graph_throughput_oracle` — expected red: rev-1 `triangle_count=300000` vs frozen 12412. Belongs to **012 U7**.
- `jtml.qml_lint` — known pre-existing headless red.
- Ordinary `jtml.cut0_measurement` must not mutate `test/golden/cut0_measurement.md` unless `JTML_UPDATE_GOLDEN=1`.

---

## Residual product judgments (not blockers for U1)

- U7 still measures after machinery U3–U5 exists; a no-go keeps default-deny.
- Retain is paired batch throughput (proposed 1.20× N=2), not end-to-end POH wall.
- Recapture tax is reported, not a hard gate.

---

## jj

`jj describe -m "<scope>: <msg>"` then `jj new` per logical change. Never `jj restore` over unverified work.
