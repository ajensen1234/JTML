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

**U4 DONE** (`1011f9cf` + oracle change): hook-driven greedy feeder + no-sync completion. LANDED + verified: `ComposeDirectDilationScore` (CUDA-free shared composition), `completeFromPins` = `complete()` minus sync (both share helper — recipe split verified), EvaluationExecutor hook seam (PollResult + enqueue/poll/completeFromPins/teardown + Install*), `RunBatchWithCost` hook-driven loop (poll-all for OOO, indexed input-order store, firstSubmission only after successful enqueue, abort drains+ForceRelease, overflow/non-finite=>PostLaunchAbort, no re-poll), `.cu` `InstallCudaFeederHooks` (real cudaGraphLaunch+EventRecord, EventQuery tri-state, completeFromPins). hook_feeder_test.cpp 8 cases green incl. AE1 OOO; no U1 regression; full headless green (only pre-existing qml_lint).

**U4 GPU anti-stub oracle DONE + VERIFIED:** `test/oracle/evaluation_executor_graph_test.cu` (oracle;gpu) drives the real EvaluationExecutor + real recipe + real `InstallCudaFeederHooks`; runs real `cudaGraphLaunch` + event query + `completeFromPins`, returning **distinct finite non-zero scores 247167.97 / 247148.97** (real metric composition, not a constant). Passed independently on RTX 3090 Ti (21 assertions). No per-eval `cudaStreamSynchronize`/`cudaDeviceSynchronize` in the admitted path (completeFromPins is sync-free; complete() keeps its once sync for the serial helper).

**U5 DONE** (next jj change): watchdog poison, teardown, terminal recovery. `EvaluationExecutor` gains `isPoisoned()` + `std::atomic<bool> poisoned_`; `Prepare`/`RunBatch`/`RunBatchWithCost` refuse at entry (return WatchdogPoisoned no-op) once poisoned; hook-driven watchdog poll-sweep path now `LeavePoisoned` each in-flight + `poisoned_.store(true)` (hang = terminal, NOT ForceRelease); early-expiry watchdog also latches poison; executor `Shutdown()` skips `destroyHook_` for poisoned contexts (leak hung wrappers per C8). Tests: `hook_feeder` 4 new `[u5][poison]` (hang poisons all in-flight + kept-checked-out, refuse-after-poison proves no hook/lease taken, Shutdown-safe, ordinary Error = PostLaunchAbort-not-poison). Headless fully green (only pre-existing qml_lint). Reviews: no blockers; count==0 early-return fixed; Shutdown wrapper-destroy skip fixed.

**Carry into U6:** wire production `prepareHook` (stage_manager.GetGraphRecipeCaptureInputs + recipe->createGraph) into optimizer_manager decision.install block (currently only InstallCudaFeederHooks → production graph path stays default-deny NotSubmitted); add coordinator `if (evaluation_executor_->isPoisoned())` early-exit before U12/graph work (poisoned session refuses all GPU work per C8 — currently executor-level only, U12 not yet gated); document poisoned-session restart; manager SetBatchCost count characterization (U1); stage_id/frame_index wiring test (U2); real curvature_capacity/graph_overhead_bytes.

**Carry from reviews into U5+:** production `prepareHook` (capture via stage_manager.GetGraphRecipeCaptureInputs + recipe->createGraph) wired into optimizer_manager decision.install block (currently only InstallCudaFeederHooks there — so production graph path stays default-deny NotSubmitted until U5/U6); manager SetBatchCost count characterization (U1 review); stage_id/frame_index wiring test (U2 review); real curvature_capacity/graph_overhead_bytes.

**Next: U4 GPU oracle first, then U5** (watchdog poison/teardown — builds on U4's loop).

---

## Known reds (do not “fix” outside their units)

- `jtml.graph_throughput_oracle` — expected red: rev-1 `triangle_count=300000` vs frozen 12412. Belongs to **012 U7**.
- `jtml.qml_lint` — known pre-existing headless red.
- Ordinary `jtml.cut0_measurement` must not mutate `test/golden/cut0_measurement.md` unless `JTML_UPDATE_GOLDEN=1`.

---

## Residual product judgments (not blockers for U1)

- U7 measured (plan 013 U3, 2026-08-20): **reverted 0.095×** with the corrected (paced) feeder. Plan 013 U0 probe explains it: the graph path is host-bound — `cudaGraphLaunch` costs ~29.4 µs host-side per launch, per-pose host floor ~73 µs ≳ serial ~104 µs wall, GPU busy ~1.5 %. **No-go: default-deny stays** (machine-qualified evidence in `test/golden/graph_performance_baseline.json` + `test/golden/probe_measurement.md`).
- Retain remains paired batch throughput (proposed 1.20× N=2); plan 013 U0 shows it is **unreachable on this fixture/machine** (host-bound graph path) — recorded `probe=unreachable`.
- Recapture tax is reported, not a hard gate.

---

## jj

`jj describe -m "<scope>: <msg>"` then `jj new` per logical change. Never `jj restore` over unverified work.
