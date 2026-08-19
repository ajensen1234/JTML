# Handoff: CUDA-Graph Greedy Evaluation Executor (Plan 011)

**Date:** 2026-08-19
**Plan:** `docs/plans/2026-08-19-011-feat-cuda-graph-greedy-evaluation-executor-plan.md`
**Requirements:** `docs/brainstorms/2026-08-19-cuda-graph-greedy-evaluation-executor-requirements.org`
**Compound learning:** `docs/solutions/logic-errors/jtml-cuda-graph-stub-failure-2026-08-19.md`

---

## Honest state: what's real, what's stub, what's next

All 8 plan units are `[ ]` unchecked. Here's the verified state:

### Real (keep)

| Unit | What's real | File:line evidence |
|---|---|---|
| U1 (partial) | `EvaluationContext` struct, `GraphRecipe` interface, `GraphRecipeRegistry`, `BankFootprintInput.graph_overhead_bytes`, `RenderBuffers`/`MetricBuffers` types, `bank_state.cuh` admission math | `include/compute/evaluation_context.h`, `include/compute/graph_recipe.h`, `include/compute/bank_state.cuh` |
| U2 | `TEST_IMPACT_MATRIX.md` + `test/golden/graph_pre_registration.json` (rev 2, real Kneel_1 fixture: 12,412 tris, 1024×1024, dilation 6/4/1) | `docs/TEST_IMPACT_MATRIX.md`, `test/golden/graph_pre_registration.json` |
| U3 (partial) | Synthetic micro-graph capture probe works; `ProbeSyntheticMicroGraph` captures a dummy kernel + memset and instantiates successfully | `src/compute/graph_preflight.cu` `ProbeSyntheticInternal` |
| U4 (partial) | `StridePrefixPersistentKernel`/`FillTrianglePersistentKernel`/`OverflowCheckKernel` **defined** (but never launched); `dev_nextCandidate`/`dev_nextChunk`/`dev_overflowFlag` allocated in `RenderEngine` (but not in `EvaluationContextPool`) | `src/compute/render_engine.cu:841/847/872`, `:332-335` |
| U5 (partial) | `DirectDilationMonoplaneRecipe` skeleton: `isEligible`, `preflight`, `keyForContext`, `createGraph`/`launch`/`destroyGraph` lifecycle (but captures dummy kernel, `complete()` returns `0.0`); `GraphExecWrapper` fixes the dangling-`d_out` UAF bug | `src/compute/graph_recipe_direct_dilation.cu` |
| U6 (partial) | `EvaluationExecutor` greedy loop bookkeeping (lease/checkout/recycle, `firstSubmission` atomic flag, watchdog, ordered store) — but `pollOneLease` calls serial cost, not `cudaGraphLaunch`/`cudaEventQuery` | `src/compute/evaluation_executor.cpp` |
| Real kernels | `RenderPhase(BankState&)` / `CompleteRenderPhase(BankState&)` / `EnqueueDirectDilationOnBank` / `CompleteDirectDilationOnBank` / `FastImplantDilationMetric` / `DistanceMapMetric` — all real, stream-explicit, work in production | `src/compute/render_engine.cu:1182-1340`, `src/compute/CostFunctionManager.cpp:306-386` |

### Stub (must be replaced with real implementation)

| Unit | What's stub | Evidence | What real looks like |
|---|---|---|---|
| U1 | `EvaluationContextPool::Initialize` returns null-init contexts (no `cudaStreamCreateWithFlags`, no `cudaMalloc` for counters) — **the U1 fresh-round real allocation was verified correct (208 lines, `cudaStreamCreateWithFlags`, `cudaEventCreate`, `cudaMalloc` for counters, null-safe Shutdown, `REQUIRE(stream != nullptr)` test) but was destroyed by a `jj restore` accident** | `src/compute/evaluation_context.cpp` is 162 lines (stub), should be 208 (real) | Mirror `BankStatePool::BankAllocation::Create` (`cost_capacity_service.cu:130-190`): real `cudaStreamCreateWithFlags(NonBlocking)`, `cudaEventCreateWithFlags(DisableTiming)`, `cudaMalloc` for `dev_nextCandidate`/`dev_nextChunk`/`dev_overflowFlag`, `cudaHostAlloc` for `host_overflowFlag`, zeroed via `cudaMemsetAsync` on context stream. Test: `REQUIRE(ctx->stream != nullptr)`, `REQUIRE(a->stream != b->stream)`, `Shutdown()` idempotent. |
| U3 | `ProbeCurrentSerialPath` returns hardcoded `kSyncBlocker` constant (`graph_preflight.cu:116-126`), not an actual capture attempt of the real render op-set | `src/compute/graph_preflight.cu:116` | Actually `cudaStreamBeginCapture(Global)` over `RenderPhase` kernels + `cub::DeviceScan::ExclusiveSum` + `cudaMemcpyAsync` and report the real capture result |
| U4 | 3 persistent kernels **defined but zero launch sites**; `RenderPhase(EvaluationContext&)` / `CompleteRenderPhase(EvaluationContext&)` return `cudaErrorNotReady` stubs (`render_engine.cu:1346-1348`); `gpu_metrics.cu:285-286` EvaluationContext overloads return `cudaErrorNotReady`; no `cudaMemsetAsync(0)` clear node for counters; host `fragment_fill` read at `:1298` + `cudaStreamSynchronize` at `:1286` still present | `grep -rn 'StridePrefixPersistentKernel<<<' src/` = 0 hits | Wire `RenderPhase(EvaluationContext&)` to the existing kernel chain on the context's stream (same kernels as `RenderPhase(BankState&)` but stream-explicit and without the host sync/fragment_fill readback); launch the 3 persistent kernels with `atomicAdd` chunk claim + `dev_overflowFlag` predicate; metric crops derived from device AABB with fixed-max grid + early exit |
| U5 | `U5_DummyKernel` writes `42` (`:22-24`); `createGraph` captures only the dummy (`:95`); `updateParams` `(void)ctx; return true;` (`:127`); `complete()` `return 0.0` (`:143`) | `src/compute/graph_recipe_direct_dilation.cu` | Capture the REAL `EnqueueRenderPrimaryCamera` → `EnqueueFastImplantDilationMetric` → `EnqueueDistanceMapMetric` chain as one `cudaGraphExec_t`; `updateParams` calls `cudaGraphExecKernelNodeSetParams` for pose; `complete()` reads pinned scores and returns `white_sum + fidm + distance` (same as `CompleteDirectDilationOnBank`) |
| U6 | `RunBatch` calls serial cost per pose (`evaluation_executor.cpp:50-66`); `.cu` is `EvaluationExecutorCudaDummy()` (`:27`); no `cudaGraphLaunch`, no `cudaEventQuery` | `src/compute/evaluation_executor.cpp`, `src/compute/evaluation_executor.cu` | Real `cudaGraphLaunch(exec, ctx.stream)` per checked-out context; `cudaEventQuery(ctx.completion_event)` poll loop with `cudaErrorNotReady` vs real-error discrimination + watchdog; `cudaEventRecord` after launch; ordered `result[lease.input]` store; `firstSubmission` atomic `R7/R8` split (preflight → serial fallback, post-launch → clear vector + wait + `OptimizerError`) |
| U7 | "passes" because `complete()==0.0` makes graph-vs-serial trivially identical | `test/oracle/layered_correctness_test.cpp` | Graph vs serial over real Kneel_1 frames (`1024/2806.tif`); Layer A rendered image byte-identical; Layer B raw `int` reductions (`pixel_score`, `distance_score`/`edge_count`, `intersection`/`union`, `white_count`) identical; Layer C `double` within frozen `abs 1e-12 / rel 1e-9` |
| U8 | Times `cudaMemsetAsync` on 4 bytes per pose; `MakeCompatibilityContext` doesn't exist; `nsys` logic is `which nsys` | `test/oracle/graph_throughput_oracle_test.cu` | Real `BuildGpuCostAdapter` (N=1) vs `EvaluationExecutor::RunBatch` (N=2/max) on Kneel_1 fixture; `nsys profile` → `cuda_gpu_kern_sum` shows `FillTriangle`/`DeviceScan`/`Dilate` overlap; populate `graph_performance_baseline.json` with `measured_P`/`amdahl_S_N2`/`nsight_concurrent_percent_N2`/`max_gap_us`/`zero_sync`; retain vs `jj abandon` per R14 |

### Compound doc created during stub work (flag as aspirational)

- `docs/solutions/architecture-patterns/graph-tiered-correctness-2026-08-19.md` — written during the U7 stub round; describes a layered-correctness convention that was never actually implemented (U7 was circular). Should be marked as "aspirational, not yet proven" or deleted when U7-real lands.

---

## Production wiring (verified)

The app entry is:
```
OptimizerBridge.cpp:71
  → controller_(new OptimizerRunController(jta::CreateOptimizerManagerRunDriver, this))
    → OptimizerManagerRunDriver creates a fresh OptimizerManager per run
      → OptimizerManager::Optimize() → RunDirectStage()
        → BuildGpuCostAdapter (serial, always)
        → IF capacity_service_->poolSize() > 1 && !biplane && DIRECT_DILATION:
            RunCostBatchGreedy (U12 bank path — real GPU work)
        → IF evaluation_executor_->registry().FindEligible("DIRECT_DILATION", false):
            RunBatch (U6 graph path — currently serial passthrough)
```

**Key finding:** production never registers a recipe (`CreateDirectDilationMonoplaneRecipe()` is only called in `multistage_oracle_test.cpp:1598`), so `FindEligible` returns `nullptr` and the graph path is never selected in production. The U12 bank path IS selected when `poolSize() > 1` (real `cudaMemGetInfo` admission at `optimizer_manager.cpp:775-795`).

---

## Real fixture facts (corrected from invented values)

| Parameter | Old (invented) | Real (Kneel_1) | Source |
|---|---|---|---|
| Implant | 300k tri representative | `KR_right_7_fem.stl` = 12,412 facets | `grep -c 'facet normal' example_studies/Kneel_1/KR_right_7_fem.stl` |
| Frame | 512×512 | 1024×1024 (`1024/2806.tif`) | `test/oracle/multistage_oracle_test.cpp:114-115` |
| Dilation | 6 | 6/4/1 (Trunk/Branch/Leaf) | `multistage_oracle_test.cpp:370-372` |
| Per-eval latency | ~98µs (claimed) | CPU median 96.9µs / GPU-event 97.2µs | `test/golden/cut0_measurement.md` |
| Batch shape | Fixed 8/16/32 | 2×|POH-boxes| per DIRECT iteration (variable) | `direct_optimizer.cpp` `TrisectPotentiallyOptimal` |

`test/golden/graph_pre_registration.json` was updated to rev 2 (real Kneel_1 values). The old rev 1 (300k/512²) was removed.

---

## jj restore incident

`jj restore graph-greedy.nsys-rep graph-greedy.sqlite` (intended to untrack two binary nsys output files) **destroyed uncommitted working-copy changes** — the U1 real allocation code (208 lines, verified correct) and the pre-reg rev2 JSON were reverted to stubs. Likely cause: the large-PDF snapshot refusal (`papers/*.pdf` > 1MB) created an incomplete snapshot, and `jj restore` reverted against that incomplete snapshot.

**Lesson:** `jj describe` + `jj new` before any `jj restore`. Add large files to `.gitignore` first. Never `jj restore` with uncommitted verified work in the working copy.

---

## Anti-stub verification protocol (from compound doc)

For the next agent executing U3-U8:

1. **Read the actual `.cu`/`.cpp` after each subagent returns** — confirm the real kernel/API is on the hot path.
2. **Run `ctest -L oracle` on the GPU machine** before marking done — headless alone is insufficient.
3. **Run `nsys profile` on the test** — empty `cuda_gpu_kern_sum` = no real work.
4. **Check for circular tests** — if the test re-implements the math instead of launching the real kernel, it's circular.
5. **Check `complete()` return** — if `0.0` or constant, graph-vs-serial is meaningless.
6. **Check launch sites** — `grep -rn 'KernelName<<<' src/` must show real launch sites.

---

## Recommended execution order for fresh round

1. **U1-real** (redo): `EvaluationContextPool::Initialize` allocates real `cudaStream_t`/event/counters per context. Test: `REQUIRE(stream != nullptr)`. Mirror `cost_capacity_service.cu:130-190`. The verified 208-line version was destroyed by `jj restore` — regenerate with the same prompt.
2. **U3-real**: `ProbeCurrentSerialPath` does an actual `cudaStreamBeginCapture` over the real render op-set.
3. **U4-real** (the hard one): wire `RenderPhase(EvaluationContext&)` / `CompleteRenderPhase(EvaluationContext&)` to the existing kernel chain on the context's stream; remove `:1286` sync + `:1298` host read; launch the 3 persistent kernels with `atomicAdd` chunk claim + `dev_overflowFlag` predicate; metric crops from device AABB with fixed-max grid + early exit. **This is the R5 core deliverable.**
4. **U5-real**: capture the REAL `EnqueueRenderPrimaryCamera` → `EnqueueFastImplantDilationMetric` → `EnqueueDistanceMapMetric` chain as one `cudaGraphExec_t`; real `updateParams` (`cudaGraphExecKernelNodeSetParams` for pose); real `complete()` reading pinned scores. Delete `U5_DummyKernel` and `return 0.0`.
5. **U6-real**: real `cudaGraphLaunch` + `cudaEventQuery` poll + watchdog + ordered store in `evaluation_executor.cu`. Replace serial passthrough. Register recipe in production (`optimizer_manager.cpp`).
6. **U7-real**: graph vs serial over real Kneel_1 frames. Layer A/B exact, Layer C within frozen tolerance. Delete circular `complete()==0.0` comparison.
7. **U8-real**: real `BuildGpuCostAdapter` (N=1) vs `RunBatch` (N=2/max) on Kneel_1. `nsys` timeline → `measured_P`/`amdahl_S_N2`/`≥30%` concurrent. Retain vs `jj abandon` per R14.

---

## jj stack

```
ymzxxxpp feat(oracle): paired throughput/latency + Nsight proof (U8)  ← @ (working copy, all [ ])
ykoqtvrv feat(oracle): layered oracle gate (U7)
lyuxqzrn feat(compute): greedy batch wiring + ordered assembly (U6)
wuwmzttq feat(compute): monoplane DIRECT_DILATION graph recipe (U5)
ylxnrytu feat(compute): device-driven persistent chunk workers (U4)
vkpzolso feat(compute): graph capture compatibility probe (U3)
pspuslty feat(test): test-impact matrix + frozen baseline harness (U2)
nwkkrnqv feat(compute): private EvaluationContext + generic GraphRecipe surface (U1)
xrzppnqx docs: add CUDA-Graph greedy evaluation executor plan and requirements
smtxvqwk feat(coordinator): wire U12 greedy bank scheduler into production stages
```

The working copy @ contains: plan uncheck (8 `[ ]`), pre-reg rev2 (re-applied after jj restore), plus leftover stub-round modifications (graph_recipe_direct_dilation.cu dangling-pointer fix, graph_throughput_oracle_test.cu RPATH/cut0 fixes, cut0_measurement.md/z_profiles.json recorder regenerations, graph_performance_baseline.json stub, docs/architecture + handoff updates). The nsys binary files have been removed.

**Next:** `jj describe -m "docs: honest handoff + compound learning + plan uncheck + pre-reg rev2"` to capture this state, then `jj new` for the fresh U1-real round.
