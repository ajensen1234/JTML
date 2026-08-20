---
title: "CUDA-Graph Greedy Evaluation Executor"
type: feat
status: active
date: 2026-08-19
origin: docs/brainstorms/2026-08-19-cuda-graph-greedy-evaluation-executor-requirements.org
---

# CUDA-Graph Greedy Evaluation Executor

## Overview

Replace JTML's host-sized raster/metric launch boundary with a generic CUDA-Graph-backed greedy execution layer that evaluates a POH pose batch faster while leaving DIRECT's synchronous iteration contract untouched. The executor owns private `EvaluationContext`s (pose + mutable device write set + `cudaStreamNonBlocking` + completion event + pinned result + status), feeds the POH vector greedily, launches a reusable per-cost-function graph per context, polls completion without blocking other contexts, and returns costs in original input order.

The first admitted recipe is monoplane `DIRECT_DILATION`. Unsupported cost families and biplane retain the exact serial adapter. The work introduces a device-driven persistent-worker raster/metric design that eliminates the current 5-integer D2H packet barrier (`AABB[4] + fragment_fill`) and `cudaStreamSynchronize`, making the full cost chain graph-capturable with layered bit-exact correctness and paired throughput/latency gating.

## Problem Frame

JTML's `DirectOptimizer` calls a CUDA-backed cost function through `OptimizerManager` -> `CostFunctionManager` -> `GPUModel::RenderPrimaryCamera` -> `GPUMetrics::FastImplantDilationMetric/DistanceMapMetric`. Each DIRECT iteration submits a set of potentially-optimal hyperbox (POH) poses and must receive their costs before the next selection.

The current monoplane `DIRECT_DILATION` path produces pose-dependent raster work: `PrepareLaunchPacketKernel` derives `fragment_fill = size[last] + prefix[last]` and the aggregate `AABB`; the host then copies that 5-int packet (`src/compute/render_engine.cu:1159-1173` `cudaMemcpyAsync` + `cudaStreamSynchronize`), computes `fill_grid = ceil(fragment_fill/256)`, and launches `StridePrefixKernel`/`FillTriangleKernel` (`src/compute/render_engine.cu:1185-1224`). Metric kernels similarly derive crop grids from the host AABB (`src/compute/fast_implant_dilation_metric.cu:304-341`, `src/compute/distance_map_metric.cu:145-165`). That interruption prevents a whole evaluation from being a reusable CUDA Graph and caps the parallelizable fraction `P` (Amdahl). The product outcome is faster POH-batch cost acquisition without changing DIRECT's synchronous algorithm or silently compromising registration correctness. (see origin: docs/brainstorms/2026-08-19-cuda-graph-greedy-evaluation-executor-requirements.org)

---

## Requirements Trace

- R1. Domain contract remains synchronous and ordered: DIRECT submits one ordered POH vector and receives the cost vector in the same input order before the next iteration.
- R2. Executor greedily feeds poses to available private contexts; callers must not pre-partition into fixed-size sub-batches.
- R3. Generic cost-function graph-recipe interface represents eligibility, preflight, launch, completion, cleanup. First admitted recipe is monoplane `DIRECT_DILATION`; unsupported recipes retain the serial path.
- R4. In-flight evaluation owns private mutable state: pose inputs, device write set, stream, completion state, result storage. Immutable geometry/comparison data may remain shared only when verified read-only.
- R5. Admitted recipe removes the device-to-host packet barrier from the critical render-to-metric path: the device drives pose-dependent raster/metric work after graph launch.
- R6. Graph recipe is reusable for repeated evaluations with stable topology and per-evaluation parameter/buffer updates; compatible with concurrently active private contexts.
- R7. Runtime CUDA/graph error after graph-backed submission aborts the stage through the existing error path; no partial results combined with serial fallback.
- R8. Recipe that fails preflight capture/instantiation before any graph-backed submission is deterministically marked unavailable and uses the serial path.
- R9. Layered correctness: rendered output and raw integer metric reductions are bit-exact vs serial baseline; only final floating-point composition may use a pre-registered, tightly bounded tolerance with written rationale.
- R10. Every affected existing test is classified before modification as retained unchanged, retained with graph coverage, intentionally superseded with rationale, or genuinely obsolete; no deletion merely because internals change.
- R11. Graph-backed path preserves replay-ordered DIRECT bookkeeping: calls, optimum sequence, non-finite handling, callback order match serial semantics.
- R12. Performance claims use fixed-work paired experiments after warmup: serial vs graph-backed greedy at multiple context counts, repeated trials; throughput primary, POH-batch latency also reported and must not materially regress.
- R13. Nsight Systems evidence must confirm real device overlap and identify remaining sync/launch gaps; stream/context count alone is not evidence.
- R14. Graph-backed recipe retained only when it passes correctness gates and demonstrates pre-registered performance result; measured no-go keeps serial path supported.

**Origin actors:** A1 (DIRECT optimizer), A2 (greedy evaluation executor), A3 (cost-function graph recipe), A4 (registration developer), A5 (test and measurement harness)

**Origin flows:** F1 (graph-backed POH evaluation), F2 (unsupported recipe or graph setup), F3 (runtime CUDA failure)

**Origin acceptance examples:** AE1 (covers R1, R2, R11 — ordered replay), AE2 (covers R3, R4, R6 — private contexts), AE3 (covers R5, R9 — no barrier + bit-exact), AE4 (covers R7, R8 — preflight fallback vs runtime abort), AE5 (covers R10, R12, R13, R14 — test matrix + measurement)

**Traceability — Actors / Flows / Acceptance Examples → Units**

| Requirement / Actor / Flow / AE | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 |
|---|---|---|---|---|---|---|---|---|
| R1 ordered domain contract |  | ✓ |  |  |  | ✓ |  |  |
| R2 greedy feeding |  |  |  |  |  | ✓ |  |  |
| R3 generic recipe / monoplane admission | ✓ |  | ✓ |  | ✓ | ✓ |  |  |
| R4 private mutable state | ✓ |  |  | ✓ |  | ✓ |  |  |
| R5 device-driven barrier removal |  |  | ✓ | ✓ | ✓ |  | ✓ |  |
| R6 reusable topology + concurrent contexts | ✓ |  | ✓ |  | ✓ | ✓ |  |  |
| R7 runtime abort (no partial) |  |  |  |  |  | ✓ |  |  |
| R8 preflight fallback |  |  | ✓ |  | ✓ | ✓ |  |  |
| R9 layered bit-exact |  | ✓ |  | ✓ | ✓ |  | ✓ | ✓ |
| R10 test-impact matrix |  | ✓ |  |  |  |  | ✓ |  |
| R11 replay-ordered bookkeeping |  | ✓ |  |  |  | ✓ |  |  |
| R12 paired throughput (frozen) |  | ✓ |  |  |  |  |  | ✓ |
| R13 Nsight overlap (quantitative) |  |  |  |  |  |  |  | ✓ |
| R14 retain only on gates |  | ✓ |  |  |  |  | ✓ | ✓ |
| F1 graph-backed POH |  |  |  |  |  | ✓ | ✓ | ✓ |
| F2 unsupported/setup | ✓ |  | ✓ |  | ✓ | ✓ |  |  |
| F3 runtime CUDA failure |  |  |  |  |  | ✓ |  |  |
| AE1 ordered replay (R1,R2,R11) |  | ✓ |  |  |  | ✓ |  |  |
| AE2 private contexts (R3,R4,R6) | ✓ |  |  | ✓ | ✓ | ✓ |  |  |
| AE3 no barrier + bit-exact (R5,R9) |  |  |  | ✓ |  |  | ✓ |  |
| AE4 fallback vs abort (R7,R8) |  |  | ✓ |  | ✓ | ✓ |  |  |
| AE5 matrix + measurement (R10,R12-R14) |  | ✓ |  |  |  |  | ✓ | ✓ |
| A1 DIRECT optimizer |  | ✓ |  |  |  | ✓ |  |  |
| A2 greedy executor | ✓ |  |  |  |  | ✓ |  |  |
| A3 graph recipe | ✓ |  | ✓ |  | ✓ |  |  |  |
| A4 registration developer (fallback diagnostics) |  |  | ✓ |  | ✓ | ✓ |  |  |
| A5 test/measurement harness |  | ✓ | ✓ |  |  |  | ✓ | ✓ |

> Every row has at least one ✓; no actor/flow/AE is orphaned (see coherence finding). U6's lifecycle `QSignalSpy` test covers A1/R11 signal ordering that the earlier plan omitted.



---

## Scope Boundaries

- First implementation supports only monoplane `DIRECT_DILATION`; biplane and other cost families (`DIRECT_MAHFOUZ`, `sym_trap_function`, `DD_NEW_POLE_CONSTRAINT`, etc. in `include/compute/CostFunctionManager.h`) remain serial until each has its own verified recipe. The `CostCapacityService::RunCostBatchGreedy` monoplane gate (`src/coordinator/optimizer_manager.cpp:1304-1310`) is the admission precedent.
- This work does not change DIRECT selection, partitioning, budgets, callback semantics, or the synchronous iteration boundary (`include/domain/direct_optimizer.h:30-41` batch sibling contract, cumulative caps `20k->25k->30k->35k` trunk/branch/leaf).
- Not an adoption of Laine--Karras's general-purpose bin/coarse/fine/multisample rasterization pipeline. Target is a narrowly scoped device-driven replacement for the current host-sized raster/metric launch boundary, preserving the existing flattened candidate-fragment representation where possible.
- No unconditional performance claim follows from context count, CUDA Graph adoption, or successful stream overlap alone; Cut-0 `~98us CPU + ~98us GPU-event per eval` (`test/golden/cut0_measurement.md`) is a gate, not a speedup (see origin: R12-R14).
- No silent fallback inside an already-submitted graph-backed batch; runtime failure aborts via `OptimizerError` (see origin: R7, existing `cost_capacity_service.cu:403-413` abort semantics).
- No caching/memoization of cost results; no UI wiring for graph selection.

### Deferred to Follow-Up Work

- Additional cost-function graph recipes (biplane, mahfouz, pole-constraint variants) — each needs its own measured recipe and admission after this plan's interface lands.
- Full biplane dual-camera context/metric graph coverage (secondary `RenderBuffers` path in `include/compute/bank_state.cuh`).
- Adaptive `N_MAX` tuning beyond the existing `BankAdmission` budget (`bank_state.cuh:250-270` half-free-memory) — measure after first recipe's `P` is known.
- CUDA Graph node priority / capture-mode refinements once the baseline reusable graph is proven.

---

## Context & Research

### Relevant Code and Patterns

- `include/domain/direct_optimizer.h:30-41` / `src/domain/direct_optimizer.cpp` — pure DIRECT loop with optional `BatchCostFunction` sibling; replay-ordered bookkeeping is the contract. Domain must remain CUDA-free.
- `include/compute/bank_state.cuh` — Stage-1 write-set contract (`RenderBuffers`, `MetricBuffers`, `BankState` with opaque `void* stream/completion_event`, `BankFootprintInput`/`BankFootprint`/`BankAdmission` footprint and half-memory admission). Allocation-free; keeps this plan's `EvaluationContext` math exact.
- `src/coordinator/optimizer_manager.cpp:1288-1337` — script-driven `Optimize()` loop and the U12 batch bridge: `BuildGpuCostAdapter` (single-eval) + `SetBatchCost` lambda that constructs `BankEnqueue`/`BankComplete` around `CostFunctionManager::EnqueueDirectDilationOnBank`/`CompleteDirectDilationOnBank`. The batch is admitted only when `capacity_service_->poolSize()>1 && !biplane && DIRECT_DILATION`.
- `src/compute/render_engine.cu:1069-1224` — render hot path: `RenderPhase` (pose->project->bbox->CUB `ExclusiveSum`->`PrepareLaunchPacketKernel`->D2H packet + `cudaStreamSynchronize` at `:1173`) and `CompleteRenderPhase` (host `fill_grid = ceil(fragment_fill/256)` -> `StridePrefixKernel`/`FillTriangleKernel`). Existing fill kernels already guard on `i < prefix[last]+size[last]` (`:726-742`, `:698-703`), enabling safe fixed-grid/persistent-worker alternatives.
- `src/compute/fast_implant_dilation_metric.cu:304-353` / `src/compute/distance_map_metric.cu:145-180` / `src/compute/gpu_metrics.cu` — metric chains: host AABB-derived crop `left/bottom/right/top` drives `dim3 grid` and `cudaMemcpyAsync` of `int` scores + `cudaStreamSynchronize` in `Complete*` (`:351`, `:178`). Same packet dependency as raster.
- `src/compute/cost_capacity_service.cu:305-422` — greedy scheduler `RunCostBatchGreedy` (checkout->enqueue->record event->poll/recycle with ordered `result[input]`). Precedent for input-indexed store (`result[lease.input]`) and `cudaEventSynchronize` vs `cudaEventQuery` distinction.
- `src/compute/CostFunctionManager.cpp:283-382` — `TrySetActiveBank`, `EnqueueDirectDilationOnBank`, `CompleteDirectDilationOnBank` over `gpu_principal_model_`/`gpu_metrics_` via bank stream. Current bank path still uses mutable `SetActiveBank` alias.
- Test harness: `test/CMakeLists.txt` — `Catch2` pure logic vs `QtTest` lifecycle seams; default `LABELS headless` (`ctest -L headless --timeout 600`, `QT_QPA_PLATFORM=offscreen`) and `oracle`/`gpu` gated fixtures (`test/oracle/bit_identity_test.cpp`, `multistage_oracle_test.cpp`, `cost_capacity_oracle_test.cu`). `test/golden/cut0_measurement.md` paired measurement precedent.

### Institutional Learnings

- `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md` — **Direct blueprint.** Prescribes `DirectOptimizer -> synchronous BatchCostFunction -> EvaluationExecutor -> EvaluationContext{pose,bank-owned mutable state,stream,event,pinned result,status} -> RenderContext/MetricContext`, async-interior / sync-boundary shape, 6 conditions for multi-instance pools (private mutable state, explicit non-default stream, pinned private host result, no blocking memcpy/legacy stream, destruction waits for stream/event, share read-only), greedy `for pose in order / acquire / enqueue / record / while in flight poll cudaEventQuery` loop, Amdahl `S(N)=1/((1-P)+P/N)` with N=2 caps `1.33x@P0.5 / 1.60x@P0.75 / 1.82x@P0.9`, and `CUB DeviceScan` capture caution.
- `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md` — Hybrid `Catch2`+`QtTest` + two-tier oracle (Tier-1 analytic bit-exact + Tier-2 silhouette IoU>0.85) + cumulative DIRECT budget doctrine; anti-patterns: false confidence, circular tests; plan must keep `sync BatchCostFunction wrapper` (compatibility `MakeCompatibilityContext->Submit->Complete`) for bit-identity proof.
- `docs/solutions/logic-errors/jtml-cost-function-update-parameter-int-noop-2026-08-12.md` + sibling `cost-function-parameter-double-truncation-2026-08-08.md` — Generic recipe wiring must not mutate by-value `Parameter<T>` vectors (`get*Parameters()` returns copy). Wire recipes to `getActiveCostFunctionClass()->set*ParameterValue`. Add hegel `GraphRecipe->Dilation` round-trip PBT.
- `docs/solutions/conventions/jtml-layered-lib-split-2026-08-08.md` — 6 targets `jtml_domain(STATIC, Qt/GPU-free) <- jtml_services <- jtml_coordinator <- jtml_view <- app` + `jtml_compute(SHARED, owns CUDA)`; `file(GLOB CONFIGURE_DEPENDS)` + explicit `.cpp/.cu` list; include prefixes `domain/x.h`, `compute/gpu_*.cuh` with grep gates; `CUDA_SEPARABLE_COMPILATION ON` only on `jtml_compute`.
- `docs/solutions/logic-errors/jtml-heatmap-guard-allocator-preconditions-2026-08-12.md` — Every guard that skips allocation must null-init destructor-freed members and validate guard preconditions; destruction waits for stream/event; run `compute-sanitizer` on teardown after zero-work graph run.

### External References

- CUDA Programming Guide 13.3: `cuda-guide/04-special-topics/cuda-graphs.md` (reusable topology, `cudaGraphExecKernelNodeSetParams`/`cudaGraphExecUpdate`, `cudaGraphLaunch` stream, device-launch restrictions), `cuda-guide/02-basics/asynchronous-execution.md` (streams/events/`cudaEventQuery` vs `cudaEventSynchronize`, per-thread default stream, capture `cudaStreamBeginCapture`/`EndCapture` invalidation via sync/query on captured stream).
- CUDA Runtime API 13.3.1: `cuda-runtime-docs/modules/group__cudart__graph.md` (instantiate `cudaGraphInstantiate`, graph update constraints, `cudaGraphInstantiateFlagDeviceLaunch`).
- CUDA Best Practices Guide: `best-practices-guide/11.5-concurrent-kernel-execution.md` (non-default `cudaStreamNonBlocking` required for concurrency; resources bound concurrency, not just stream count), `best-practices-guide/9.1-timing.md`, `best-practices-guide/4.1-profile.md` + `performance-traps.md`.
- Papers: `papers/Laine and Karras - 2011 - High-performance software rasterization on GPUs.pdf` (sort-middle, bin/coarse/fine, `fragment_fill` analogue is variable per-triangle fragment count resolved before raster); `papers/Laine et al. - 2020 - Modular Primitives for High-Performance Differentiable Rendering.pdf` (differentiable raster primitives — not adopted, but cited in origin as considered binning alternative).

---

## Key Technical Decisions

- **Generic recipe surface, one initial recipe.** The `GraphRecipe` interface is cost-function-oriented from day one (eligibility/preflight/build/update/launch/complete/destroy + budget/footprint keys), but only `direct_dilation_monoplane` is implemented and admitted. This keeps `R3` testable without pretending unverified compatibility. (see origin: Key Decisions)
- **EvaluationContext owns private mutable state; no active-bank alias as primary design.** `RenderEngine`/`GPUMetrics` APIs gain explicit `EvaluationContext`/`RenderContext`/`MetricContext` views; `SetActiveBank` becomes a compatibility shim. This resolves the aliasing gap in `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md`.
- **Device-driven persistent chunk workers over the flattened candidate list (Option B).** Replaces the 5-int packet barrier. Workers claim `chunk_size` via `atomicAdd(&next_candidate, chunk_size)` while `start < fragment_fill` (device-resident), using the existing `FillTriangleKernel:726-742` and `StridePrefixKernel:698-703` guards. This is substantially narrower than Laine--Karras bin/coarse/fine/multisample rerasterization; it preserves the same candidate-fragment list and advances the plan's preferred implementation.
- **Metric crop also device-resident.** Derive `left/bottom/right/top`, `sub_cropped_width/height`, `crop_width/height` entirely from the device AABB written by the bbox phase and launch metric kernels with a **fixed-max grid + early-exit guard** (`if (x>=cropW||y>=cropH) return`, matching current `698-703`/`726-742` semantics). No host `bounding_box` read between render and metrics within a graph. The host-load alternative is rejected; U3 probes the fixed-max grid for capturability.
- **Reusable graph topology per context, per-evaluation param/buffer updates.** Capture once per compatible key (recipeId + biplane flag + frame dims + triangle count + camera calib + dilation + cub_storage_bytes + curvature_capacity + footprint), instantiate **one private `cudaGraphExec_t` per `EvaluationContext`** (no sharing), relaunch many times via `cudaGraphExecKernelNodeSetParams`/`cudaGraphExecMemcpyNodeSetParams` for pose/buffer addresses (pinned host twins remain graph nodes where needed). Preflight probes `cudaStreamBeginCapture`/`cudaGraphInstantiate` legality for this workload (including `cub::DeviceScan::ExclusiveSum`) and surfaces a deterministic fallback to serial. A pooled-Exec alternative is rejected — it would mutate node params while concurrent contexts are in-flight, violating R4/R6.
- **Layered correctness.** Bit-exact `renderer_output` image and raw `int` metric reductions vs the serial baseline. For the monoplane `DIRECT_DILATION` recipe the per-eval reductions are `pixel_score` (FID) and `distance_score`/`edge_count` (distance-map); `comparison_image_white_sum` is a capture-time constant, NOT a per-eval reduction, and `intersection/union/white_count` belong to other cost paths, not this one. Only the final `double score = comparison_image_white_sum + (-pixel_score) + distance/(edge_count+0.1)` may use a tight tolerance **pre-registered in U2** (`abs 1e-12` / `rel 1e-9` landed in `test/golden/graph_pre_registration.json` with rationale) — U4/U7 must not derive tolerance after the fact. This advances the origin's `Split the gate by layer` decision.
- **Test stewardship: matrix before mutation.** Every affected test gets one of four dispositions (retained unchanged / retained with graph coverage / superseded with named replacement+Rationale / genuinely obsolete) **approved before U4 mutates any `src/compute` file**; an `obsolete` row requires code-owner sign-off. Internal changes alone never justify discarding oracle/bit-identity coverage. `TEST_IMPACT_MATRIX.md` is the gate for U4.
- **Measurement: paired warmup + Nsight Systems overlap proof.** Serial vs graph-greedy at admitted `N=1,2,4` (clamped by `BankAdmission` half-memory **plus graph object VRAM, probed by an instantiate trial before admission**), warmup `N` launches discarded, repeated trials, throughput `evals/sec` primary, `p50/p99` pose latency and POH-batch latency also reported, Nsight Systems timeline as overlap artifact **with quantitative gates: ≥30% concurrent kernel time at N=2, max host-to-device launch gap <50us, zero `cudaStreamSynchronize`/`cudaEventSynchronize`/`cudaMemcpy` on admitted graph path (grep + timeline)** (origin: R12-R13).
- **Error: preflight fallback, runtime abort.** Setup failure (capture/instantiate/update) before the first `cudaGraphLaunch` (`firstSubmission` flag clear) marks that recipe unavailable for the run and selects the serial `BuildGpuCostAdapter` at the `OptimizerManager` branch (no sentinel inside `DirectOptimizer`). After first submission (`firstSubmission` set atomically on first successful `cudaGraphLaunch`), any real CUDA error (`cudaEventQuery`/`cudaStreamQuery` ≠ `cudaErrorNotReady`) aborts the whole POH batch: **clear the ordered result vector, wait for all context streams/events, then report via the existing `Optimize() bool/String` + `OptimizerError` signal** (no partial vector, no sentinel inside `BatchCostFunction`). Add a watchdog timeout for `cudaEventQuery` spin. (origin: R7-R8, `cost_capacity_service.cu:403-413` precedent).

---

## Open Questions

### Resolved During Planning

- **Should the first graph-backed executor be limited to monoplane `DIRECT_DILATION`?** Yes, but expose the generic `GraphRecipe` interface now so other recipes can be added without re-plumbing the executor. (see origin: Scope Boundaries + Key Decisions)
- **Is Laine--Karras bin/coarse/fine/multisample rasterization adopted?** No. A narrowly scoped device-driven chunk-worker redesign over the existing flattened candidate list is sufficient to remove the packet barrier (see origin: Scope Boundaries).
- **Should the host packet be kept for metrics?** No. The admitted recipe must make metric crop grids device-resident/graph-internal, not host-driven between render and metrics (see origin: R5).
- **Is the Laine--Karras read the right input to plan choice?** Yes — it identifies the variable-work/fragment-count root cause and confirms why a simpler persistent-worker design solves the host-sized-launch problem without the paper's full pipeline.
- **Should overflow gating be host-branch or device-predicated?** Device-predicated via `dev_overflowFlag` with DAG `memset -> Prepare -> overflowCheck -> workers` (see HA-11); host-branch would reintroduce the packet barrier.

### Deferred to Implementation

- Whether to keep the legacy `cost_capacity_service.cu` bank pool as a fallback or consolidate entirely under `EvaluationExecutor` — defer until U1's ownership shape is measured; U1 must record an explicit keep-as-serial-fallback-shim vs remove decision before U6 wiring (see scope guardian S06).
- Secondary `RenderBuffers` empty-in-monoplane invariant and read guard for generic metric path — enforce `secondary==empty` assertion for the monoplane recipe and add a null-guard CI check so a generic metric read does not touch secondary `dev_fragment_fill`/`dev_bounding_box` when empty (see adversarial FM-12).

> Note: Numeric fixed-grid ceiling, chunkSize, graph keying, tolerance, and representative workloads are now **resolved** (see Key Technical Decisions and U2). The items above are the only remaining deferred decisions; the previous five deferred items are frozen in U2's `test/golden/graph_pre_registration.json`.


---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```text
DirectOptimizer (pure)
  │  BitCostFunction        (R12 fallback)
  │  BatchCostFunction      ──► EvaluationExecutor (greedy, sync boundary)
  │                              │  GraphRecipeRegistry (generic, admits direct_dilation_monoplane)
  │                              │  EvaluationContextPool (private mutable per context)
  │                              │   ├─ RenderContext (banks: output, bbox, prefix, triangles, CUB scratch)
  │                              │   ├─ MetricContext (pixel/distance/edge/union/white/curvature twins)
  │                              │   ├─ cudaStreamNonBlocking + cudaEvent (completion)
  │                              │   └─ pinned host result slot + status + pose + input_index
  │                              └─ greedy: for pose in input order -> lease ready context
  │                                      update graph node params (pose/buffer addrs)
  │                                      cudaGraphLaunch(context.graphExec, context.stream)
  │                                      while in flight: cudaEventQuery (NotReady vs error)
  │                                      on cudaSuccess: copy pinned value -> result[input_index]
  │
  └─ synchronous BatchCostFunction: poses -> ordered double[] -> replay-ordered bookkeeping
```

Device-driven persistent fill within one graph (Option B):

```text
Graph (captured once per context/key, relaunched per pose):
  clear output (Async)
  WorldToPixelKernel + BoundingBoxForTrianglesKernel + BoundingBoxSizesKernel
  cub::DeviceScan::ExclusiveSum  (probed for graph capture; fallback path if uncapturable)
  PrepareLaunchPacketKernel      (writes device fragment_fill)
  StridePrefix persistent chunk workers:  while nextChunk < fragment_fill
  FillTriangle persistent chunk workers:  while nextCandidate < fragment_fill -> map candidate->triangle/pixel -> barycentric test -> write output
  FIDM chain (device AABB bounds) + distance-map chain (device AABB bounds)  (no host bbox read)
  cudaMemcpyAsync raw int reductions -> pinned host twins
  (completion event is NOT captured here — the executor calls cudaEventRecord on the
   context stream AFTER cudaGraphLaunch, outside capture; querying an event recorded
   inside a capture is prohibited)
```

---

## Implementation Units

- [x] U1. **Private EvaluationContext + generic GraphRecipe surface**

**Goal:** Durable ownership that makes greedy graph execution safe by construction; legacy bank alias is compatibility-only.

**Requirements:** R3, R4, R6

**Dependencies:** None

**Files:**
- Create: `include/compute/evaluation_context.h`
- Create: `src/compute/evaluation_context.cpp`
- Create: `include/compute/graph_recipe.h`
- Modify: `include/compute/bank_state.cuh` (keep Stage-1 math; note `EvaluationContext` replaces `BankState` as primary executed type)
- Modify: `src/compute/CMakeLists.txt` (explicit `.cpp/.cu` list — GLOB trap)
- Modify: `include/compute/render_engine.cuh` / `src/compute/render_engine.cu` (add explicit-context overloads; keep old API as shim)
- Modify: `include/compute/gpu_metrics.cuh` / `src/compute/gpu_metrics.cu` (same)

**Approach:**
- `EvaluationContext` = `{ pose, RenderBuffers (+ dev_nextCandidate/dev_nextChunk/dev_overflowFlag), MetricBuffers, cudaStream_t, cudaEvent_t, cudaGraphExec_t, pinned int score twins, status, input_index }`. Pool size = `bank_state_math::admit(free_bytes, footprint(frameDims, triCount, cubBytes), n_max).bank_count` clamped by half-memory, floored at `1` (serial fallback). Every constructor null-inits destructor-freed members; destruction waits for its stream/event before `cudaFree`/`cudaGraphExecDestroy` (per `jtml-heatmap-guard-allocator-preconditions-2026-08-12.md`).
- `GraphRecipe` = interface `{ isEligible(costName, biplane), preflight(frame/model/cudaProps), key(frameDims, calib, dilation, triCount), createGraph(stream), updateParams(graphExec, context), launch(graphExec, stream), complete(context) -> double }`. `GraphRecipeRegistry` enumerates recipes; admits only `DIRECT_DILATION monoplane` initially. Reads dilation via `getActiveCostFunctionClass()->get*ParameterValue` (not `updateCostFunctionParameterValues` — `jtml-cost-function-update-parameter-int-noop-2026-08-12.md`).
- Render/metric overloads take `EvaluationContext&` instead of relying on `SetActiveBank`; old `TrySetActiveBank` path stays for serial compatibility.

**Patterns to follow:** `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md` 6-pool conditions + `DirectOptimizer -> synchronous BatchCostFunction -> EvaluationExecutor` shape; `docs/solutions/conventions/jtml-layered-lib-split-2026-08-08.md` lib split + grep-gate for `compute/` prefixes; `docs/solutions/logic-errors/jtml-heatmap-guard-allocator-preconditions-2026-08-12.md` null-init + destruction wait.

**Test scenarios:**
- Happy path: `EvaluationContext` null-init leaves every destructor-freed pointer null/false; zero-work construction (`0 triangles` or `0 curvature`) returns `initialized_correctly_=true` with no allocation.
- Happy path: `BankAdmission` half-memory math matches `bank_state.cuh` precedent for the real Kneel_1 fixture (`12412` tri / `1024x1024` frame / `cubStorageBytes` from `RenderEngine::GetCubStorageBytes()`).
- Happy path: `GraphRecipeRegistry` admits only `DIRECT_DILATION && !biplane`; unknown cost name or biplane returns ineligible with serial-fallback sentinel, never instantiates a graph.
- Error path: `U1` construct with `curvature_capacity` derived from a garbage-initialized input fails (mirrors `frame.h` `num_curvature_keypoints_` guard lesson) — but correct path requires fully-initialized `BankFootprintInput`.
- Integration: `RenderEngine::Render(EvaluationContext&)` routes through the same `WorldToPixelKernel`/`BoundingBox*Kernel` as `RenderPhase` but without the internal `cudaMemcpy/cudaStreamSynchronize` dispersal — prove by inspecting that the bank path uses explicit stream only.

**Verification:** `pixi run test` headless unaffected (U1 adds no new behavior to admitted path); `pixi run configure` picks up new headers via `CONFIGURE_DEPENDS`; `grep -rIn '#include "(core/|gui/|gpu/|cost_functions/)' include src` clean; explicit `.cpp/.cu` list builds.

---

- [x] U2. **Test-impact matrix + fixed-work baseline harness (characterization-first)**

**Goal:** Preserve-by-default stewardship and a pinned serial baseline so graph work is gateable before any behavior changes.

**Requirements:** R10, R11, R12, R14

**Dependencies:** U1 (contexts/recipes exist to reason about impact, even if not yet executable)

**Files:**
- Create: `test/unit/evaluation_context_test.cpp` (Catch2, headless)
- Create: `test/unit/graph_recipe_preflight_test.cpp` (Catch2, headless)
- Create: `test/oracle/evaluation_executor_oracle_test.cu` (oracle)
- Modify: `test/CMakeLists.txt` (new targets: `LABELS headless` vs `LABELS "oracle;gpu"` `TIMEOUT 3600` `WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}`, direct-compile pattern for pure domain pieces)
- Modify: `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md` if a new guardrail is proven (compound doc follow-up)
- Test: `test/unit/test_direct_optimizer_batch.cpp` — extend with ordered-replay checks (existing Tier-0)

**Approach:**
- Produce a `TEST_IMPACT_MATRIX.md` (or appendix in this plan's follow-up) enumerating every affected test (`unit/*`, `lifecycle/*`, `oracle/bit_identity_test.cpp`, `cost_capacity_oracle_test.cu`, `multistage_oracle_test.cpp`, `z_profile_test.cpp`, `qml/*`) as retained/retained-with-coverage/superseded/obsolete with rationale + replacement.
- Characterization-first: record the pre-unit serial `pose->score` sequence and rendered images over `test/golden` frames and `test/oracle` fixtures (bit-identity baseline), plus `DirectOptimizer` replay-ordered bookkeeping (`GetCostFunctionCalls` + `GetOptimumLocation/Value` + `SetCallOffset` cumulative caps).
- Define and **freeze** the fixed representative POH workloads for U8 as a checked-in `test/golden/graph_pre_registration.json` — `8/16/32` pose batches from the real Kneel_1 fixture (`12412`-triangle implant at `1024x1024`), dilation `6`, `N=1,2,4` (clamped by `BankAdmission` half-memory **including graph VRAM, probed by an instantiate trial before admission**), discard first `3` launches per config, `10` trials, report `p50/p99`, material latency regression `>10%` p99 **AND** `>5%` stage-level wall-time. Define Layer-C tolerance there (`abs 1e-12` or `rel 1e-9` with rationale). This committed artifact is what U8 verifies against; any change requires re-approval as a scope change.

**Patterns to follow:** `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md` two-tier oracle (Tier-1 analytic + Tier-2 silhouette IoU≥0.85, vertically flipped, repo-root `WORKING_DIRECTORY`, cumulative `20k->35k` budget) + `direct_data_storage.cpp`/`data_structures_6D.cpp` direct-compile isolation; `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md` wrapper pattern `MakeCompatibilityContext->Submit->Complete`.

**Test scenarios:**
- Covers AE5. Happy path: `TEST_IMPACT_MATRIX` covers every `test/unit` and `test/oracle` file touching `CostFunctionManager`/`GPUModel`/`RenderEngine`/`GPUMetrics`/`direct_optimizer` — a test that is missing from the matrix is a failure.
- Happy path: serial characterization run through `BuildGpuCostAdapter` (plan 008 U9 `src/coordinator/optimizer_manager.cpp:1568-1594`) produces a pinned `bit_identity_baseline` that later graph runs diff against, with empty diff.
- Happy path: `BatchCostFunction` replay-ordered bookkeeping: with a recording fake `CostFunction` vs `BatchCostFunction` over the same POH, `GetCostFunctionCalls()` (including `SetCallOffset`), `GetOptimumLocation/Value()`, and `GetNonFiniteCount()` are identical regardless of which seam was used (Tier-0 contract `include/domain/direct_optimizer.h:30-41`).
- Edge case: trivial `1`-pose and `0`-pose batches never require `N>1` to succeed; they complete with ordered result and the harness counts them as `N=1` comparable.
- Integration: the U2 harness itself runs as `ctest -L headless` (zero GPU/VTK) for the domain/contract parts and `ctest -L oracle` separately for the GPU baselines.

**Verification:** `ctest -L headless --timeout 600` green; matrix reviewed and landed; baseline artifacts checked into `test/golden/` with `jj describe` rationale.

---

- [x] U3. **Graph capture compatibility probe (preflight truth)**

**Goal:** Deterministic yes/no for whether the cost recipe's operation set is graph-capturable on the real `CUDA 12.9` build, before the device-driven redesign is hardened.

**Requirements:** R6, R8

**Dependencies:** U1 (recipe surface), U2 (baseline to compare fallback against)

**Files:**
- Create: `include/compute/graph_preflight.h`
- Create: `src/compute/graph_preflight.cu`
- Modify: `src/compute/CMakeLists.txt`
- Test: `test/oracle/graph_capture_probe_test.cu`

**Approach:**
- Probe, on the `oracle` label, a minimal capture sequence: `cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)` -> `RenderPhase` fragment of the candidate operation set (including `cub::DeviceScan::ExclusiveSum` with its `dev_cub_storage` alias) -> `cudaStreamEndCapture` -> `cudaGraphInstantiate` -> `cudaGraphDestroy`/`cudaGraphExecDestroy`. Also probe `cudaMemcpyAsync` pinned-host edges and `atomicAdd` chunk workers as isolated graph nodes.
- A failed capture is not a crash: probe returns `GraphPreflightResult{ capturable, reasonCode, failingNodeHint }` (derived from the Runtime API's `group__cudart__graph` `cudaGraph*` errors). The executor uses this to mark that recipe unavailable for the run (R8) vs aborting a batch (R7).
- Confirm `cuda-skill/references/cuda-guide/02-basics/asynchronous-execution.md` capture-invalidators: `cudaDeviceSynchronize`/`cudaStreamSynchronize`/`cudaStreamQuery` on a captured stream or legacy-stream use while a non-blocking captured stream exists will invalidate capture — so the probe must prove the *proposed* operation set has none of those.

**Patterns to follow:** `cost_capacity_service.cu` error-return discipline (`cudaGetLastError` immediately after launch, `cudaMemcpyAsync` + stream, no blocking `cudaMemcpy` on bank path); `performance-traps.md` frequency of syncs.

**Test scenarios:**
- Happy path: on the current serial operation set, probe returns non-capturable with the known blocker (`cudaStreamSynchronize` in `RenderEngine::RenderPhase` `:1173` + host AABB dependency) — this is the expected red result that justifies U4.
- Happy path: a synthetic capturable micro-graph (`WorldToPixelKernel` -> `cudaMemcpyAsync` -> `FillTriangleKernel` with fixed grid) captures and instantiates successfully, proving the graph path is viable *in principle* on this toolchain.
- Edge case: probe on a `0`-triangle or `maximum_stride_size` overflow case reports capturable=false with the correct reasonCode rather than an unhandled CUDA error.
- Error path: probe cleans up (`cudaGraphDestroy`/`cudaGraphExecDestroy`) even after `cudaStreamEndCapture` returns an error-graph-`NULL` (capture invalidation path per `cudaStreamIsCapturing`/`EndCapture` docs).

**Verification:** `ctest -L oracle` probe test passes on the GPU machine, reporting preflight result without side-effecting global state; headless suite unaffected.

---

- [x] U4. **Device-driven persistent chunk workers (remove the packet barrier)**

**Goal:** Make raster and metric work device-resident so the full cost chain is graph-internal.

**Requirements:** R5, R9

**Dependencies:** U3 (probe confirms which operation set needs replacing), U1 (private write sets so workers can be private), U2 (**TEST_IMPACT_MATRIX approval + baseline to prove bit-identity** — U4 must not mutate `src/compute` until matrix is landed)

**Files:**
- Modify: `src/compute/render_engine.cu` (new kernels: `StridePrefixPersistentKernel`, `FillTrianglePersistentKernel` + device `next_candidate` counter)
- Modify: `include/compute/render_engine.cuh`
- Modify: `src/compute/fast_implant_dilation_metric.cu` / `include/compute/gpu_metrics.cuh` (device-AABB overloads or graph-internal crop)
- Modify: `src/compute/distance_map_metric.cu`
- Test: `test/oracle/evaluation_executor_oracle_test.cu` (extend — bit-identity diff)
- Test: `test/unit/render_pipeline_builder_test.cpp` (extend — chunk math, overflow guard)

**Approach:**
- Replace the two-phase `RenderPhase` / `CompleteRenderPhase` split at the packet with one graph-internal flow: after `PrepareLaunchPacketKernel` writes device `fragment_fill`, persistent workers claim ranges:
  ```
  __global__ StridePrefixPersistentKernel(nextChunk, chunkSize=256, fragment_fill, ...) {
    while (true) { int base = atomicAdd(nextChunk, chunkSize); if (base>=fragment_fill) break; // stride work }
  }
  __global__ FillTrianglePersistentKernel(nextCandidate, chunkSize, fragment_fill, triCount, prefix, sizes, ...) {
    while (true) { int start = atomicAdd(nextCandidate, chunkSize); if (start>=fragment_fill) break;
      for (int i=start; i<min(start+chunkSize, fragment_fill); ++i) { // map candidate->triangle/pixel -> barycentric test -> write }
    }
  }
  ```
  Chunk size candidates `256` vs `512` measured; `NX` = `min(maxBlocksPerSM * SMcount, ceil(SAFE_CAP/256))` as a **fixed measured upper-bound grid** (not host-computed per pose) that self-retires. Each launch is preceded by a **captured `cudaMemsetAsync(0)` node clearing `dev_nextCandidate`/`dev_nextChunk` (both `int32` device counters, `1*4` bytes each, added to `RenderBuffers` and `BankFootprintInput`/`BankFootprint` so `BankAdmission` half-memory accounts for them; U1 null-inits and destructor `cudaFree`/`cudaGraphExecDestroy` wait covers them)**; explicit DAG edges **`cudaMemsetAsync(0) -> PrepareLaunchPacketKernel -> overflowCheckKernel (writes `dev_overflowFlag`) -> {StridePrefixPersistentKernel, FillTrianglePersistentKernel}` predicated on `!dev_overflowFlag`** enforce ordering and close HA-11 TOCTOU. `NX` uses per-kernel `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for Stride vs Fill, taking the **min**; `SAFE_CAP = maximum_stride_size*256` alongside overflow threshold `maximum_stride_size*(256-1)` is documented in the footnote. For metrics, derive `sub_left/right/top/bottom`, `crop_width/height` from the *device* AABB written by the bbox phase and launch with a **fixed-max 2-D grid derived from `max(2048x2048, frameDims)` in the graph key** + `if (x>=cropW||y>=cropH) return` guard (matching the current `dim3 grid(ceil(crop/ sqrt(256)))` arithmetic but without host input), probed in U3 for topology stability. Overflow flag is a **device int -> pinned int `cudaMemcpyAsync` node at graph tail** checked in `Complete` path.
- Keep the current `src/compute/render_engine.cu:726-742` `if (i < prefix[last]+size[last])` guard semantics — no-op threads preserve bit-identity while allowing overlap.
- Overflow guard stays device-visible: `fragment_fill > maximum_stride_size * (256-1)` still yields `fragment_overflow_` and `cudaErrorMemoryAllocation`, but now the check reads the *device* value within the graph (or via a pre-launch status kernel) rather than a host read + immediate error return.
- Keep the capacity-service `CapacityGrid`/`gridFor` path as a no-op for `FillTriangleKernel` (already `src/compute/render_engine.cu:922-941` strict no-op) and allow its future removal once U4's persistent grid is proven.

**Patterns to follow:** `include/compute/bank_state.cuh` footprint math for `maximum_stride_size` / `cub_storage_bytes` upper bound; `src/compute/render_engine.cu:698-703` and `:726-742` existing guards; `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md` condition 6 (share read-only, private mutable).

**Test scenarios:**
- Covers AE3. Happy path: same pose through U4's persistent fill vs serial host-sized fill produces bit-identical `renderer_output` image (byte compare) over `test/golden` frames.
- Covers R9 (layered). Happy path: same pose through full U4 raster+metric produces bit-identical raw `int` reductions (`pixel_score`, `distance_score/edge_count`, `intersection/union`, `white_count`) and therefore bit-identical host scores after composition.
- Edge case: tiny `fragment_fill <256` and large `fragment_fill ~ few-M` both complete without deadlock; `nextCandidate` chunking never skips or duplicates a candidate index.
- Edge case: `1x1` and `2048x2048` image bounds with `dilation` near-width/height produce the same crop semantics as `max/min` clamping in `fast_implant_dilation_metric.cu` / `distance_map_metric.cu`.
- Error path: overflow case remains detected (now via device value) and propagates as `fragment_overflow_` without producing a partial image/metrics result.

**Verification:** `ctest -L headless` green; `ctest -L oracle` bit-identity diff empty for the U4-patched kernels run serially (N=1) before greedy is enabled.

---

- [x] U5. **Monoplane `DIRECT_DILATION` graph recipe (reusable topology)**

**Goal:** One reusable graph recipe that embodies the entire `DIRECT_DILATION` cost chain for monoplane, capturable and relaunchable per evaluation context.

**Requirements:** R3, R5, R6, R9

**Dependencies:** U1, U3, U4 (device-driven workers make full capture possible)

**Files:**
- Create: `include/compute/graph_recipe_direct_dilation.h`
- Create: `src/compute/graph_recipe_direct_dilation.cu`
- Modify: `include/compute/graph_recipe.h` / `src/compute/graph_recipe.cpp` (registry, preflight key)
- Modify: `src/compute/CMakeLists.txt`
- Test: `test/oracle/graph_recipe_direct_dilation_test.cu`

**Approach:**
- Recipe `direct_dilation_monoplane`:
  - `isEligible` returns true only when `CostFunctionManager::getActiveCostFunction()=="DIRECT_DILATION" && !calibration_.biplane_calibration` (mirrors `src/coordinator/optimizer_manager.cpp:1308-1310`).
  - `preflight` runs the U3 probe for this recipe's operation set + `BankFootprint` check for graph node memory; returns `GraphPreflightResult` that U8/U1 use for selection.
  - `key = { recipeId, biplaneFlag, width,height,triangle_count, dilationParam, cameraCalibHash, cub_storage_bytes, curvature_capacity, maximum_stride_size, graphRecipeVersion }` — pose-independent, so the same `cudaGraphExec_t` can be relaunched for any pose in the same stage; `recipeId`/`biplaneFlag` prevent cross-family reuse (see coherence R3).
  - Build: `cudaStreamBeginCapture(stream, Global)` over the U4 operation set (clear, project, bbox, CUB scan, prepare packet, persistent fill/metric, `cudaMemcpyAsync` raw ints) -> `cudaStreamEndCapture` -> `cudaGraphInstantiate` (flags: none initially; `cudaGraphInstantiateFlagUseNodePriority` deferred) -> cache per `key`+context.
  - Per-eval update: `cudaGraphExecKernelNodeSetParams` for pose constants (`model_pose_`, `model_rotation_mat_`), buffer addresses from `EvaluationContext`, `cudaMemcpyAsync` pinned twins; topology never changes.
  - **One private `cudaGraphExec_t` per `EvaluationContext`** — concurrent contexts never share an `Exec` (pooled alternative rejected; it would mutate node params while in-flight, violating R4/R6). Verified by U5 two-contexts-in-flight test.
- Unsupported recipes keep their graph node absent; the registry entry exists but preflight returns `admitted=false`.

**Patterns to follow:** `cuda-guide/04-special-topics/cuda-graphs.md` topology stability + `cudaGraphExec*NodeSetParams` reuse; `cost_capacity_service.cu` half-memory admission (`BankAdmission.admitted`).

**Test scenarios:**
- Happy path: capture+instantiate of `direct_dilation_monoplane` graph succeeds for the real Kneel_1 fixture (`1024x1024` frame + `12412` tri implant) on RTX 3090/4090 class hardware; `cudaGraphLaunch` on an unpopulated context succeeds (smoke).
- Happy path: graph relaunch with a different pose reuses the same topology after only node-param updates — second pose's `PrepareLaunchPacketKernel` output differs (new `fragment_fill`) yet the `Exec` relaunch succeeds without re-capture.
- Happy path: two `EvaluationContext`s each with their own `cudaGraphExec_t`+`cudaStreamNonBlocking`+`pinned twins` can be in-flight simultaneously and each produces the correct pose->score mapping when run serially (correctness before concurrency).
- Edge case: `preflight` with biplane or `DIRECT_MAHFOUZ` returns deterministic `capturable=false` with a stable reasonCode; caller retains `BuildGpuCostAdapter` serial path.
- Error path: a graph that was not instantiated for device launch and attempts to vary a device-side `cudaGraphLaunch` node correctly rejects via the Runtime API's `cudaGraphExecUpdate`/`Instantiate` error path — the recipe marks itself unavailable for the run.

**Verification:** `ctest -L oracle` capture test green on GPU machine; headless unchanged; no new `cudaDeviceSynchronize` introduced (grep gate).

---

- [ ] U6. **Greedy batch wiring + ordered result assembly (real CUDA graph path)**

**Goal:** Make one synchronous `BatchCostFunction` call feed an ordered POH vector greedily to real per-context CUDA graphs, while preserving DIRECT replay/callback semantics. This realizes the 08-17 executor pattern: private `cudaStreamNonBlocking` contexts, executor-recorded completion events, `cudaEventQuery` polling (`cudaErrorNotReady` is pending, any other non-success is a real error), and result storage at original input indices.

**Requirements:** R1, R2, R3, R4, R5, R6, R7, R8, R11.

**Dependencies:** U1 (pool), U2 (frozen replay contract), U5 (real graph recipe), and the U2 test-impact matrix.

**Post-U5 reality / constraints:**
- `EvaluationExecutor::RunBatch` in `evaluation_executor.cpp` is still a serial-cost simulation; `evaluation_executor.cu` is a dummy. Headless tests intentionally link only the host `.cpp` path.
- `DirectDilationMonoplaneRecipe::createGraph` returns a caller-owned `GraphExecWrapper*` (owns both `cudaGraph_t` and `cudaGraphExec_t`) via `void** out_graphExec`; `recipe->destroyGraph` is its only correct deallocator.
- `EvaluationContextPool::Shutdown()` treats `ctx.graph_exec` as a raw `cudaGraphExec_t`. **Never store a `GraphExecWrapper*` in `ctx.graph_exec`**; the executor owns wrappers in a separate per-context collection.
- The current OptimizerManager U6 null-recipe fallback sets `useExecutor=true`, which replaces the real U12 bank batch with serial passthrough. This is a regression: no eligible/preflight-failed recipe must leave the existing U12 bank batch or serial adapter intact (R8).
- The U5 recipe's `complete()` synchronizes its stream. U6 must move greedy completion to event-query + a no-sync `completeFromPins()` path; otherwise U8 cannot honestly claim overlap or zero per-eval synchronization.

**Key design decisions:**
1. **Executor-owned wrapper lifetime.** Add `graphExecs_` and `graphKeys_` collections indexed by context index. Entries contain the `GraphExecWrapper*`, full `GraphRecipeKey`, and a frame/parameter generation token. On executor shutdown: drain known-pending streams, call `recipe->destroyGraph(wrapper)` once per non-null entry, clear every entry, then let the pool release raw buffers/events. `Shutdown()` is idempotent, so destructor-after-explicit-Shutdown is a no-op.
2. **Headless/real CUDA split — use injected hooks, not a direct `.cpp → .cu` call.** Headless tests link `evaluation_executor.cpp` but not the `.cu` TU; a direct `RunBatchGpu` symbol call would not link. Keep one ordered greedy loop in `.cpp`, driven by CUDA-free hook members declared in `evaluation_executor.h`: `enqueue(ctxIdx, pose) -> bool`, `poll(ctxIdx) -> {Pending, Done, Error}`, `completeFromPins(ctxIdx) -> double`, and `teardown()`. `evaluation_executor.cu` defines/installs the real hooks through `InstallEvaluationExecutorGpuPath(EvaluationExecutor&)`; hooks remain null for headless/unsupported paths, leaving exact serial passthrough. `.cpp` never references a `.cu`-only symbol.
3. **Capture-input provider and complete key assembler.** `CostFunctionManager` owns stage/frame-specific inputs. Add `GetGraphRecipeCaptureInputs(GraphRecipeCaptureInputs&)` and `GPUModel::GetPrimaryRenderEngine()`. The provider fills render engine, metrics, rendered image, dilated comparison frame, distance map, and active dilation; executor supplies `inputs.context`. Add one shared `GraphRecipeKey` assembler used by both OptimizerManager admission and executor capture: active-frame width/height, model triangle count, `maximum_stride_size`, `cub_storage_bytes`, `curvature_capacity`, active dilation, camera-calibration hash, graph overhead, recipe id, and biplane flag. Validate `key.dilation == inputs.dilation` and non-null `rendered_image` before capture. A zero/default key is a deterministic R8 not-submitted failure, never a reason to weaken recipe preflight.
4. **Invalidate and prepare graphs at a quiescent pre-batch boundary.** U5 captures with `cudaStreamCaptureModeGlobal`; do NOT lazily call `createGraph` during a live greedy feed while other contexts/UI CUDA work may be active. Before any launch for a batch/frame/stage, assemble the complete key and a frame/parameter generation token for every needed context, compare with stored entries, drain/destroy stale wrappers, then capture/recreate required graphs on the feeder thread while no graph contexts are in flight. `createGraph` failure in this preparation phase is `NOT_SUBMITTED` (R8 → U12/serial). This also prevents reuse of frame-A comparison pointers, dilation, or `comparison_image_white_sum` for frame B.
5. **Greedy state machine (single interleaved lease loop; no recycled-lease re-poll):**
```
graphExecs_ := executor-owned map contextIndex -> GraphExecWrapper*  // NOT ctx.graph_exec
pre-batch quiescent preparation for each pool context that may be used:
  key := assembleKey(context)
  if wrapper absent OR storedKey != key OR storedGeneration != currentGeneration:
      destroy stale wrapper if present
      inputs := provider(); inputs.context = context
      if !recipe->createGraph(key, context.stream, inputs, &graphExecs_[context]):
          return NOT_SUBMITTED      // R8: no graph launch; caller retains U12/serial
      storedKey := key; storedGeneration := currentGeneration
leases := []
for input in input order:
  ctx := checkout(); if none free: finishOneOldestLease()
  ctx.in_flight = true; ctx.status = InFlight; ctx.input_index = input
  copy poses[input] into ctx pose fields
  if !recipe->updateParams(graphExecs_[ctx], ctx):
      if firstSubmission: abortPostSubmission() else return NOT_SUBMITTED
  if !recipe->launch(graphExecs_[ctx], ctx.stream):
      if firstSubmission: abortPostSubmission() else return NOT_SUBMITTED
  firstSubmission := true                    // only after a successful launch
  cudaEventRecord(ctx.completion_event, ctx.stream)  // executor-side, outside capture
  leases.push({ctx, input, done=false})
while any lease is not done:
  progress := false
  for each unfinished lease:
      r := cudaEventQuery(lease.ctx.completion_event)
      if r == cudaSuccess:
          score := recipe->completeFromPins(lease.ctx) // no CUDA sync
          if lease.ctx.status == Failed: abortPostSubmission()
          result[lease.input] := score
          lease.done = true; recycle(lease.ctx, true); progress = true
      else if r == cudaErrorNotReady:
          continue
      else:
          abortPostSubmission()
  if !progress and watchdog since last progress expired:
      abortPostSubmission()
```
`finishOneOldestLease()` marks that lease done and removes it from the active set before re-checkout; a context can never be re-polled under an old input index.
6. **No-sync completion API.** Add `virtual double GraphRecipe::completeFromPins(EvaluationContext&) const`; default implementation delegates to `complete()` for non-graph recipes. `DirectDilationMonoplaneRecipe::completeFromPins()` is `complete()` minus `cudaStreamSynchronize`: after event success it reads only pinned scores and `host_overflowFlag`, marks `EvaluationStatus::Failed` on overflow/null/non-finite, and calls the same composition helper as `complete()` (`comparison_image_white_sum + (-pixel_score) + distance_score/(edge_count+0.1)`). Test parity between the two completion entry points on identical pins.
7. **R7/R8 and watchdog semantics.** Before first successful launch, provider/key/create/update/launch failure returns `NOT_SUBMITTED`; OptimizerManager keeps U12/serial and never installs graph `SetBatchCost`. After first launch, any launch/event/watchdog/overflow error clears the ordered vector and reports a structured failure to `RunDirectStage`; `RunDirectStage` catches/converts it to the existing bool/String + `OptimizerError` path rather than allowing `std::invalid_argument` to escape the worker thread. A watchdog failure marks the executor poisoned until re-initialized; do not blocking-sync a suspect hung stream. A true wedged GPU may need device recovery.
8. **Admission and rollout.** Replace OptimizerManager's hard-coded 8 GiB/zero graph-overhead executor initialization with measured `cudaMemGetInfo` capacity minus already committed pools and U5 graph-overhead measurement. Resolve the U12 coexistence explicitly: keep it as the graph-unavailable serial/bank fallback but never run both schedulers for one `SetBatchCost`. Keep production graph admission disabled behind an explicit experimental/runtime flag until U7 passes and U8 records a retained verdict; U6 itself must not silently ship an unmeasured graph path.

**Files:**
- Modify: `include/compute/evaluation_executor.h` (hook seam; executor-owned wrapper/key collections; capture-input provider; poisoned/error state)
- Modify: `src/compute/evaluation_executor.cpp` (single ordered greedy loop, headless serial fallback, R7/R8 split)
- Modify: `src/compute/evaluation_executor.cu` (real event-query feeder hooks, post-launch `cudaEventRecord`, no `cudaDeviceSynchronize`)
- Modify: `include/compute/graph_recipe.h` + `src/compute/graph_recipe_direct_dilation.cu` (`completeFromPins`, shared composition helper, provider validation)
- Modify: `src/compute/CostFunctionManager.cpp` + `include/compute/CostFunctionManager.h` (`GetGraphRecipeCaptureInputs`; explicit-context path)
- Modify: `include/compute/gpu_model.cuh` (`GetPrimaryRenderEngine()` accessor)
- Modify: `src/coordinator/optimizer_manager.cpp` (register recipe, install provider/hooks, populate real key, remove null-recipe override, catch/emit graph failure)
- Test: `test/unit/test_direct_optimizer_batch.cpp` (ordered leases, no re-poll, non-finite/replay, R7/R8 hook injection)
- Test: `test/lifecycle/optimizer_run_controller_test.cpp` (QSignalSpy callback and `OptimizerError` ordering on graph success/failure)
- Test: new `test/oracle/evaluation_executor_graph_test.cu` (real two-context launch/poll/key-invalidation on the real Kneel_1 fixture)

**Test scenarios:**
- Keep green: `2*|POH|`, degenerate 0/1 batches, batch `< N`, repeated `>=3x` determinism, wrong-sized batch contract.
- New R8: no recipe, incomplete key, or null `rendered_image` leaves `firstSubmission` false and preserves U12/serial instead of overwriting `SetBatchCost`.
- New R7: injected post-launch real CUDA error returns no partial vector, keeps `firstSubmission`, and reaches `OptimizerError` without serial re-fallback or an escaped throw.
- New lifetime: checkout a context, create a live wrapper, call executor Shutdown, assert `destroyGraph` exactly once; destructor-after-Shutdown remains safe.
- New invalidation: reuse one pool across two frame/parameter generations; frame two recreates wrappers and matches serial.
- New overflow: graph overflow/status `Failed` follows the serial hard-error path, not an ordinary DIRECT non-finite cost.
- GPU oracle: two real graph contexts complete out of order yet return scores in input order; event polling, `completeFromPins`, and no-sync path are exercised.

**Verification:** `ctest -L headless --timeout 600` green; lifecycle `OptimizerError`/callback-order QtTest green; GPU executor oracle green on the target GPU; grep production sources (`src/compute/evaluation_executor.cu`, executor/recipe admitted path) for no `cudaDeviceSynchronize`; do not grep test serial-reference code where D2H validation is intentional.

---


- [ ] U7. **Layered oracle gate (bit-exact image/raw-int, tolerated composition)**

**Goal:** Make graph admission enforceable with a REAL graph-vs-serial oracle: rendered image and recipe-owned raw reductions exact, final score bounded only by the frozen Layer-C tolerance. A serial-passthrough executor comparison is explicitly NOT evidence.

**Requirements:** R5, R9, R10, R14.

**Dependencies:** U2 (matrix + frozen rev-2 workload/tolerance), U4 (device workers), U5 (recipe), and U6-real (real graph launch/poll/completion). U8 may not measure or retain the graph path until this gate passes.

**Current reality / why this unit remains open:**
- `test/oracle/layered_correctness_test.cpp`, the U7 block in `bit_identity_test.cpp`, and a multistage admission fence already exist, but they call `exec.RunBatch(poses, serialCost)`. While RunBatch is serial passthrough, `graph == serial` is tautological plumbing coverage, not a graph oracle.
- The current test has no Layer-A image byte compare, no Layer-B device/pinned raw-int compare, and fabricated `triangle_count=300000` / 4 MB overhead values rather than the frozen real Kneel_1 workload.
- `docs/solutions/architecture-patterns/graph-tiered-correctness-2026-08-19.md` is explicitly aspirational because the prior U7 path was circular. It remains a design sketch until this unit proves the gate.

**Layer contract (do not blur layers):**
1. **Layer A — exact image.** For the same real Kneel_1 frame and pose, copy the graph `ctx.primary.output` and serial render output to host after their respective completion points. Require identical dimensions and byte-for-byte equality (`diff == 0`). The test's validation D2H copy is allowed; it is not part of the admitted production graph path.
2. **Layer B — exact recipe reductions.** Compare only reductions actually produced by monoplane `DIRECT_DILATION`: `pixel_score` (FID), `distance_score`, and `edge_count` (distance-map). Graph pinned host twins must equal serial bank/metric hosts exactly. `comparison_image_white_sum` is a capture-time per-frame constant, not a per-eval reduction. `intersection`/`union`/`white_count` belong to other cost paths and are out of this recipe's Layer-B gate; do not claim they were compared.
3. **Layer C — bounded composition.** Use the single recipe composition helper: `comparison_image_white_sum + (-pixel_score) + distance_score/(edge_count+0.1)`. Read frozen `abs=1e-12` and `rel=1e-9` from `test/golden/graph_pre_registration.json`; never derive or widen them after observing a diff. Layer-C tolerance never excuses a Layer-A/B mismatch.
4. **Fixture identity.** Use the rev-2 frozen workload: Kneel_1 femur, 12412 triangles, 1024x1024, Canny 3/0/150, dilation 6, backface off, real calibration/key values. Remove every U7 `300000`/`4MB` fabricated fixture constant.
5. **Concurrency/determinism.** Execute neighboring poses through at least two real contexts and repeat the same workload >=3 times. Assert context-private device reduction pointers differ, Layer A/B remain exact on every run, and every ordered result vector is identical. This catches write-set aliasing that a host-struct-address check misses.
6. **Retention gate.** U7 is an R14 prerequisite: if any Layer A/B mismatch or Layer-C out-of-bound occurs, graph admission stays disabled and U8 is blocked. Do not "fix" the oracle by changing frozen tolerance or reference data.

**Files:**
- Rewrite: `test/oracle/layered_correctness_test.cpp` (real graph arm, real fixture, Layer A/B/C assertions, three-run stress)
- Modify: `test/oracle/bit_identity_test.cpp` (U7 section drives real graph executor and checks the same frozen contract)
- Modify: `test/oracle/multistage_oracle_test.cpp` (only graph-admitted monoplane DIRECT_DILATION may enter the staged graph path; unsupported paths remain serial)
- Update only after proof: `docs/solutions/architecture-patterns/graph-tiered-correctness-2026-08-19.md` (aspirational → proven, with exact recipe reduction set and fixture)

**Test scenarios:**
- Happy path: graph vs serial Layer-A rendered image diff is zero over the golden frame.
- Happy path: Layer-B `pixel_score`, `distance_score`, and `edge_count` match exactly; Layer-C meets frozen tolerance using the shared composition helper.
- Edge: tiny/flat `fragment_fill < 256` and high-detail poses retain Layer A/B exactness.
- Concurrency: N=2, adjacent poses, >=3 repeated batches; device reduction targets are distinct and ordered outputs are deterministic.
- Failure: intentionally shared reduction target or wrong frame-generation token yields a deterministic Layer A/B failure; a failed U7 gate leaves graph admission disabled.

**Verification:** `ctest -L oracle` passes on a REAL graph executor (not serial passthrough); Tier-2 IoU>=0.85 remains required on the GPU machine; every changed/deleted oracle has a `TEST_IMPACT_MATRIX` disposition. Only then mark the graph-tiered-correctness solution doc as proven.

---

- [ ] U8. **Paired throughput/latency + Nsight Systems overlap proof**
**Goal:** Make a measured, reversible retain/revert decision for the graph-backed `DIRECT_DILATION` executor. Nothing synthetic, no fabricated throughput, and no production admission retained without correctness and timeline evidence.

**Requirements:** R9, R12, R13, R14.

**Dependencies:** U6-real (real graph launch/event polling/no-sync completion), U7-real (Layer A/B/C gate), U2 (frozen rev-2 workload and pre-registration).

**Current state / expected red:**
- `test/oracle/graph_throughput_oracle_test.cu` is synthetic scaffolding: fake percentile data, fabricated wall times/P=0.85, `which nsys`, dummy CUDA work, no `EvaluationExecutor::RunBatch`, no `BuildGpuCostAdapter` serial arm, and a fabricated 1.5x p99 gate.
- **BLOCKER (fails the oracle suite right now — expected red; `ctest -L headless` is unaffected):** it asserts rev-1 `triangle_count=300000` / 512x512 against rev-2 `graph_pre_registration.json` (`12412`, 1024x1024, Kneel_1 `2806.tif`). U8 must remove every rev-1 assertion before measurement.
- `graph_performance_baseline.json` contains placeholders, not evidence. `pending_manual_gpu_run` is not a retained verdict.
- U8 is blocked until U6 removes per-eval stream synchronization and U7 passes; current `CompleteRenderPhase`/recipe `complete()` sync behavior cannot support honest overlap claims.

**Measurement contract (freeze before the first U8 run):**
1. **Rev-2 fixture only.** Use Kneel_1 femur, 12412 triangles, 1024x1024 `2806.tif`, Canny 3/0/150, dilation 6, stage Trunk; batch sizes 8/16/32; discard 3 warmups; run 10 trials. Do not change `graph_pre_registration.json` after seeing results.
2. **Four paired arms, not three.** (a) serial N=1 via `BuildGpuCostAdapter`/compatibility path; (b) graph N=1, which isolates graph launch/reuse overhead from concurrency; (c) graph N=2, which measures overlap; (d) graph N=max admitted. Record admitted N per hardware. If max admitted N<2, record the overlap premise as not achievable on that hardware; do not substitute N=1 for the N=2 overlap gate, and retain only if the pre-registered N=1 launch-overhead criterion is met.
3. **Metrics and timing.** Record wall_time_ms, per-pose p50/p99, evals/sec, stage wall time, and GPU event time. The executor completion events are `cudaEventDisableTiming` and MUST NOT be used for elapsed timing; the harness creates its own timing-enabled event pair around each measured run. Compute measured P and Amdahl `S(N)=1/((1-P)+P/N)` only as a consistency ceiling, never as proof of benefit.
4. **Minimum benefit is a gate, not an afterthought.** Before the first measurement, add an owner-approved explicit threshold to `graph_pre_registration.json`. Default proposed threshold for review: graph N=2 on the 16-pose workload must deliver >=1.20x paired throughput versus serial N=1 AND a non-negative stage-wall result; measured speedup must also fall within the pre-registered Amdahl-consistency band. If the owner selects a different threshold, record it in the frozen artifact before running. A marginal gain that merely "fits Amdahl" is not sufficient to retain this complexity.
5. **R13 timeline gate is mandatory.** Run `nsys profile -o graph-greedy <real-harness>` then `nsys stats --report cuda_gpu_kern_sum graph-greedy.nsys-rep`, plus inspect the timeline. Retention requires real `cudaGraphLaunch` work, >=30% concurrent kernel time at N=2, max host-to-device launch gap <50us, and production-path absence of `cudaStreamSynchronize`/`cudaEventSynchronize`/blocking `cudaMemcpy`. Scope the source grep to the admitted production executor/recipe/render sources; do not flag test-side serial references or validation D2H copies. Stream count alone is not overlap. If `nsys` is unavailable, the retain decision is **BLOCKED**: `decision.status` remains `pending_manual_gpu_run`, graph admission stays disabled, and serial remains supported. A deferred timeline gate never counts as a pass.
6. **Latency, error, and availability behavior.** p99 must not regress >10% and stage-level `DirectOptimizer` wall time must not regress >5%. Probe-unavailable retains serial without failing the suite. Any post-submission CUDA/watchdog error aborts that measurement and records `failed`, never `0 evals/sec` or a partial number.
7. **R14 retain/revert.** Retain only when U7 passes, the explicit minimum-benefit + Amdahl consistency gates pass, latency gates pass, and the mandatory Nsight timeline gate passes. Otherwise keep graph admission disabled and `jj abandon`/revert only the functional admission change while retaining TEST_IMPACT_MATRIX, frozen workloads, and the measured no-go artifact. Cut-0 (~98us) is context only; never present CPU/GPU Cut-0 ratio as N-way speedup.

**Files:**
- Rewrite: `test/oracle/graph_throughput_oracle_test.cu` (real four-arm paired harness; remove rev-1/synthetic proofs; correct error/availability paths)
- Rewrite: `test/golden/graph_performance_baseline.json` (real fixture, threshold, machine/driver/commit, admitted N, all measured values, Nsight evidence, `retained`/`reverted`/`pending_manual_gpu_run` decision)
- Modify: `test/golden/graph_pre_registration.json` BEFORE running U8 (owner-approved minimum-benefit/Amdahl-consistency threshold; never post-hoc)
- Modify after measurement: `docs/handoff-2026-08-12-optimizer-path.md` and `docs/architecture/jtml-cost-evaluation-execution-graph.org` (measured P, hardware, decision, final worker strategy)

**Test scenarios:**
- Real 16-pose graph N=2 passes the pre-registered benefit/latency gates and Nsight shows overlapping kernels with no production-path serializing sync.
- Graph N=1 quantifies graph overhead; graph N=2 isolates overlap; graph N=max records saturation; 8/32 pose batches report the wall-time/throughput tradeoff.
- N<2 hardware path records overlap as unavailable and does not claim an N=2 win.
- Probe-unavailable retains serial; injected CUDA/watchdog failure is recorded as `failed` with no fabricated throughput.
- `cut0_measurement.md` remains context, never a speedup claim.

**Verification:** Land a real report artifact with raw trial summaries, machine/toolkit/driver/commit, admitted N, measured P, timing events, Nsight stats/timeline reference, and explicit retained/reverted/blocked status — or land a measured no-go note with graph admission disabled. U8 is never "done" merely because the harness compiles.

---

## System-Wide Impact

- **Interaction graph:** `Optimize()` -> `RunOptimizedStage` -> `DirectOptimizer` (with `Options`)-> injected `BatchCostFunction` -> `EvaluationExecutor` -> `GraphRecipe` (monoplane `DIRECT_DILATION`) -> `EvaluationContext` pool -> `RenderContext`/`MetricContext` on private streams/events -> pinned reductions -> ordered `double[]`. `UpdateDisplay`/`UpdateOptimum`/`OptimizerError` signal contract unchanged; iteration/improvement callbacks fire from the same replay points.
- **Error propagation:** Unimplemented variant options fail fast at stage start; graph preflight failure deterministically selects serial (R8); wrong-sized batch violates contract (fail fast); any real CUDA error inside the graph-greedy batch aborts via the one existing error path (`OptimizerError`, no partial vector, no silent serial re-fallback).
- **State lifecycle risks:** `EvaluationContext` pool allocated once after frame dims/model/trig counts are known and before any graph instantiate; freed with destructor waiting for its stream/event; `cudaMalloc`/`cudaGraphExecDestroy` guards are null-safe (`jtml-heatmap-guard-allocator-preconditions-2026-08-12.md`). Admission via `bank_state_math::admit` half-memory ceiling prevents VRAM growth on smaller GPUs. `gb` phase state remains per-`DirectOptimizer` instance, not shared.
- **API surface parity:** `BuildGpuCostAdapter` keeps 3 consumers (production runner, Tier-2 oracle twin, z-profile probe) with identical body; batch sibling is additive and all unsupported cost families keep the single-eval path.
- **Integration coverage:** Multistage oracle on the GPU machine arbitrates both the host-sized serial baseline and the graph-greedy `direct_dilation_monoplane` recipe via the same staged `20k/25k/30k/35k` harness; headless suite cannot see launch behavior by design.
- **Unchanged invariants:** `jtml-production` shape, cumulative caps, four lineage invariants, `Sym_Trap [{Leaf, repeat=0}]` script, domain layer's zero-CUDA contract, reserved stub graph names.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Persistent chunk `atomicAdd` contention or under-occupancy regresses throughput | Measure `chunkSize` (256 vs 512) and `NX` persistent grid against the host-sized baseline under `maximum_stride_size`/`SAFE_CAP` bounds; no-op threads preserve correctness, throughput gate decides retention |
| `cub::DeviceScan::ExclusiveSum` uncapturable in the graph | U3 probes it first; if uncapturable, either capture the scan as a child graph or keep that single kernel outside the graph but still device-driven (still avoids the host `fragment_fill` sizing packet) and document the exception |
| Shared per-eval-written buffer race under concurrency (nondeterministic diff) | U4/U1 binding audit enumerates every `dev_*` write (stride, prefix, triangles, bbox, output) as pool resources; share only `triangles/normals/comparison images`; hashed diff stress `>=3x` with neighboring poses on different contexts |
| Legacy default-stream implicit sync serializes contexts | `cudaStreamNonBlocking` on every bank path stream + grep-gate asserting zero `cudaMemcpy`/default-stream launches on the admitted path; `CUDA_API_PER_THREAD_DEFAULT_STREAM` considered at the pool layer |
| Runner wiring divergence if a variant copies the batch adapter | Keep one `EvaluationExecutor::RunBatch` surface; any runner extension is via the `GraphRecipe` registry, not a copied `RunDirectStage` body |
| `nvcc 13.2` vs manifest `12.9` environment drift invalidates capture conclusions | Probe and U8 report the actual `nvcc --version` with results; resolve drift before claiming version-gated graph performance |
| Dilation `6` default hides recipe wiring failure (by-value `Parameter` no-op) | Wire through `getActiveCostFunctionClass()->set*ParameterValue` and add `hegel` `GraphRecipe->Dilation` round-trip PBT (see sibling truncation/noop fixes) |
| Battery/runtime cost before recipe proves out | Keep U8 as an `oracle`/`gpu`-only `TIMEOUT 3600` harness; CI prunes to `headless`; full battery only nightly |

---

## Documentation / Operational Notes

- `docs/plans/2026-08-14-010-feat-direct-variants-capacity-launch-plan.org` gains a Phase-B successor note: U12 compatibility banks are superseded by this executor plan as the measured durable design.
- `docs/architecture/jtml-cost-evaluation-execution-graph.org` updated with the final persistent-worker strategy, graph keying, and measured `P`/throughput.
- `docs/handoff-2026-08-12-optimizer-path.md` workstream statuses updated per unit.
- `docs/solutions/` compound entry only if a new convention is proven (e.g., device-driven raster guard pattern vs Laine--Karras).
- Every unit is one logical `jj` change (`jj describe -m "<scope>: <msg>" && jj new`); a gated unit that fails its gate is abandoned in the same change (`jj abandon`).

---

## Sources & References

- **Origin document:** [docs/brainstorms/2026-08-19-cuda-graph-greedy-evaluation-executor-requirements.org](docs/brainstorms/2026-08-19-cuda-graph-greedy-evaluation-executor-requirements.org)
- **Related plans:** [docs/plans/2026-08-14-010-feat-direct-variants-capacity-launch-plan.org](docs/plans/2026-08-14-010-feat-direct-variants-capacity-launch-plan.org) (U12 compatibility prototype this supersedes), [docs/plans/2026-08-12-008-refactor-optimizer-graph-container-plan.md](docs/plans/2026-08-12-008-refactor-optimizer-graph-container-plan.md) (graph registry seam)
- **Guide:** [docs/architecture/jtml-cost-evaluation-execution-graph.org](docs/architecture/jtml-cost-evaluation-execution-graph.org)
- Related code: `include/domain/direct_optimizer.h`, `include/compute/bank_state.cuh`, `src/coordinator/optimizer_manager.cpp`, `src/compute/render_engine.cu`, `src/compute/fast_implant_dilation_metric.cu`, `src/compute/distance_map_metric.cu`, `include/compute/gpu_metrics.cuh`
- Papers (via `papers/`): `Laine and Karras - 2011 - High-performance software rasterization on GPUs.pdf`; `Laine et al. - 2020 - Modular Primitives for High-Performance Differentiable Rendering.pdf` (considered, not adopted)
- CUDA skill refs: `cuda-guide/04-special-topics/cuda-graphs.md`, `cuda-guide/02-basics/asynchronous-execution.md`, `cuda-runtime-docs/modules/group__cudart__graph.md`, `best-practices-guide/11.5-concurrent-kernel-execution.md`
