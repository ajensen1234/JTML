# Handoff: CUDA-Graph Executor Admission and Lifecycle (Plan 012)

**Prepared:** 2026-08-20 — replacement plan for remaining Plan 011 U6–U8; **U1 landed** (typed BatchOutcome, default-deny policy, null-recipe override removed, guarded RunDirectStage)

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

## Status

**U1 landed** (one jj change): `include/compute/batch_outcome.h` (BatchOutcome + CoordinatorBatchAbort + MaterializeOrderedScores), `include/compute/graph_admission_policy.h` (default-deny policy + DecideGraphAdmission), typed `EvaluationExecutor::RunBatch/RunBatchWithCost`, `RunDirectStage` now: (a) leaves the U12/serial adapter installed on every deny path, (b) no longer allocates the 8 GiB dummy executor pool (lazy per C10 — executor stays uninitialized, poolSize()==0), (c) runs through `jta::RunDirectStageGuarded` (CoordinatorBatchAbort + invalid_argument → OptimizerError). Headless 58/59 green (only pre-existing `qml_lint`). Reviewer follow-ups (non-blocking): add a manager-level SetBatchCost count characterization before U3 prepare lands; U12-survival is currently proven at unit level.

**U2 landed** (next jj change): `CaptureGeneration` (C7 identity incl. upload epoch), header-only `graph_key_assembler.h` (AssembleGraphRecipeKey, HashCameraCalibrationParams FNV-1a, AssembleCaptureGeneration, ValidateGraphKeyVsInputs), CostFunctionManager upload-epoch + `GetGraphRecipeCaptureInputs` provider, GPUModel `GetPrimaryRenderEngine`, full-key assembler + epoch bump wired in optimizer_manager. Headless 59/60 green (pre-existing qml_lint). Adversarial review: no blockers; both should-fixes applied (canonical dilation read via getActiveCostFunctionClass; provider test now asserts out.dilation==4).

**U3 landed** (next jj change): CUDA-free `CaptureCoordinator` (timed_mutex lock + park registry, reentrant, rollback), `ForceRelease`/`LeavePoisoned`/`IsPoisoned`/`InitForTest` on `EvaluationContextPool` (Checkout skips poisoned, Shutdown skips poisoned per C8), executor `graphExecs_` wrapper map + `Prepare`/`InstallPrepareHook`/`InstallDestroyHook` (batch checkout, cleanup on failure, never firstSubmission, re-prepare destroys stale wrapper). Headless 61/62 green (pre-existing qml_lint). Review: no blockers; re-prepare leak fixed (destroy stale wrapper before overwrite); Shutdown poison-skip + ctx.graph_exec-null confirmed. NOTE: CaptureCoordinator/Prepare NOT yet wired into optimizer_manager RunDirectStage — that's U4.

**Next: U4** (hook-driven greedy feeder + no-sync completion). Carry into U4: manager-level SetBatchCost count characterization (U1 review); coordinator stage_id/frame_index wiring test before `gen` consumed (U2 review); install CaptureCoordinator + Prepare into the admission/prepare transaction; wire real curvature_capacity/graph_overhead_bytes; empty park registry until VTK/UI producers enumerated (refuse capture if any producer can't park).

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
