---
title: "JTML CUDA cost evaluation: prefer explicit evaluation contexts and an executor"
date: 2026-08-17
last_refreshed: 2026-08-20
status: refreshed
category: docs/solutions/architecture-patterns
module: JTML CUDA cost evaluation
problem_type: architecture_pattern
component: service_object
severity: medium
related_components:
  - RenderEngine
  - GPUMetrics
  - GPUModel
  - CostFunctionManager
  - DirectOptimizer
tags:
  - cuda
  - evaluation-context
  - explicit-streams
  - gpu-concurrency
  - render-engine
  - bank-state
  - performance-architecture
---

# JTML CUDA cost evaluation: prefer explicit evaluation contexts and an executor

## Context

JTML's `DirectOptimizer` calls a CUDA-backed cost function through `OptimizerManager`, `CostFunctionManager`, `GPUModel`, `RenderEngine`, and `GPUMetrics`.

The serial production path is conceptually:

```text
DirectOptimizer
  -> OptimizerManager cost adapter
    -> CostFunctionManager::callActiveCostFunction()
      -> costFunctionDIRECT_DILATION()
        -> GPUModel::RenderPrimaryCamera()
        -> GPUMetrics::FastImplantDilationMetric()
        -> GPUMetrics::DistanceMapMetric()
        -> scalar double
```

A single evaluation reuses mutable render and metric state. Persistent read-only inputs include model triangles, model normals, comparison images, dilated images, and distance maps. Per-evaluation writes include projected triangles, snapped triangles, backface flags, bounding boxes, prefix/scratch buffers, rendered output, metric reductions, and pinned result copies.

The first U12 implementation used service-owned extra banks, explicit streams, and active-bank rebinding. That work established useful ownership and correctness seams, but architecture review found that it is a tactical compatibility migration rather than a durable executor design. The current feeder still has blocking completion points, including the render packet boundary and production event/stream synchronization; its N>=2 oracle proves lifecycle/in-flight behavior, not production throughput. The intended executor must therefore be treated as a target architecture, not as a measured speedup result:

- active-bank pointer aliases remain mutable object-global state;
- the host must synchronize at the fragment-packet boundary before calculating the fill grid;
- the compatibility bank and extra-bank count semantics need to be explicit;
- the current tests prove lifecycle and correctness, but not trustworthy end-to-end throughput speedup;
- an executor that merely creates multiple objects still needs private mutable state and explicit non-default streams inside every object.

The Cut-0 measurement is a gate, not a speedup result. The latest artifact records roughly 98 microseconds of GPU-event time per evaluation. The same workload must still be timed serially versus N-way before claiming improvement.

## Guidance

Use a synchronous domain boundary and an asynchronous compute interior:

```text
DirectOptimizer
  -> synchronous BatchCostFunction
    -> EvaluationExecutor
      -> EvaluationContext per in-flight pose
        -> RenderContext
        -> MetricContext
        -> cudaStream_t
        -> cudaEvent_t
        -> ordered result/status
```

An `EvaluationContext` should make the evaluation identity explicit. It should carry the pose, bank-owned mutable state, stream, completion event, result location, and failure status. Render and metric functions should consume explicit contexts or explicit state views instead of relying on one mutable `SetActiveBank()` alias as the primary design.

The outer optimizer does not necessarily need a future-based API. DIRECT must finish the current POH evaluation set before choosing the next set. The executor can remain synchronous to the caller while submitting independent poses asynchronously to CUDA streams internally.

An executor-owned pool of independent `RenderEngine`/`GPUMetrics` instances is a valid implementation option, but only when all of the following hold:

1. Each instance owns private mutable render and metric state.
2. Each instance uses an explicit non-default stream for bank-path launches.
3. Host result buffers are pinned and private to that instance/context.
4. Blocking `cudaMemcpy` and legacy default-stream launches are absent from the concurrent path.
5. Destruction waits for the instance's stream/event before freeing its state.
6. Read-only geometry and comparison inputs are shared where safe instead of duplicated unnecessarily.

Multiple instances are therefore an alternative way to realize `EvaluationContext`; they are not a shortcut around stream-aware execution.

Within one context, preserve CUDA's in-order stream semantics:

```text
submit context
  -> render preparation
  -> packet completion boundary
  -> fill launches
  -> FIDM chain
  -> distance-map chain
  -> async pinned result copies
  -> completion event
```

### Corrected poll discipline (2026-08-20)

Event polling is the correct CUDA API usage for distinguishing `cudaErrorNotReady` from real errors. The host must never hot-spin the poll loop at GHz cadence. Each `cudaEventQuery` is a host-to-driver round trip (~0.5 us), and a zero-delay busy poll starves the GPU:

- **The hot spin, not the graph, is the measured bottleneck.** The nsys profile of the greedy CUDA-graph feeder (plan 012 U7) shows 1,526,320 `cudaEventQuery` calls — 81.8% of all CUDA API time in the measured window — while the GPU was only 1.6% busy (27.6 ms of kernels over a 1764 ms span, ~2.36 us average kernel on the 12412-triangle Kneel_1 mesh; launch-to-launch host gap p50 = 712 us).
- **This is a host-bound polling artifact, not a graph-capability limit.** With the GPU 60x under-utilized, the measured graph-vs-serial ratio of 0.114x is not an architecture verdict.
- **Stop the spin, keep polling correct.** Use bounded backoff between re-polls (10–25 µs for µs-scale evals — the 50–200 µs band is too coarse: ~4–10 eval-residencies wide on a 97 µs per-eval fixture; plan 013 measured 10 µs works), OR block on the completion event (`cudaEventSynchronize`) for a dedicated serialized step / for the only remaining in-flight context, OR sweep a small event set and service only its Done events while the others keep running. NEVER busy-poll the entire lease set at zero delay.
- **Admission (N) must saturate the device.** A half-memory bank ceiling capping N=4 on a small fixture destroys the concurrency story before the poll loop even matters.

Cross-references: `docs/solutions/performance-issues/jtml-graph-feeder-cudaeventquery-hotspin-2026-08-20.md` (compound analysis) and `test/golden/graph_performance_baseline.json` (measured reverted artifact). Normalize the earlier absolute on `cudaEventSynchronize`: it is acceptable for the serialized step / last remaining context, not on a per-context non-blocking stream that other live work shares (capture-invalidator rule).

**Target design:** Across contexts, use non-default streams and event polling. `cudaEventQuery`/`cudaStreamQuery` distinguish `cudaErrorNotReady` from real errors. Poll with bounded backoff, do not hot-spin. This paragraph replaces the earlier over-broad "never synchronize" guidance; the current implementation's poll loop is being corrected toward the bounded contract above.

## Why This Matters

CUDA streams express concurrency but do not guarantee it. Multiple non-default streams can execute concurrently when device resources permit; that is a property of the device and the kernels, not a statement we can assume from the caller alone.

The host-dependent fragment packet boundary is a structural serialization point: the host must receive the fragment count before choosing the final fill grid. Other contexts can run while one waits at that boundary.

Use Amdahl's law for expectations:

```text
S(N) = 1/((1 - P) + P/N)
```

For two balanced contexts, ideal upper bounds are approximately:

- P=0.5: 1.33x
- P=0.75: 1.60x
- P=0.9: 1.82x
- P=1.0: 2.0x

The actual `P` must be measured. The Cut-0 GPU/CPU ratio must not be reported as an N-way speedup. A host-bound measurement is never a graph verdict.

## When to Apply

Apply this architecture when:

- a batch contains independent evaluations that could overlap;
- the existing public/domain contract should remain synchronous;
- multiple streams, events, and pinned result buffers are required;
- correctness depends on keeping per-evaluation reductions isolated;
- the executor admission policy lets enough contexts saturate the device.

Use the simpler compatibility-bank path only as a bounded transitional step (e.g. monoplane `DIRECT_DILATION`). Revisit before adding more metrics, biplane support, or performance claims.

## Examples

### Serial compatibility wrapper

```cpp
EvaluationContext bank0 = MakeCompatibilityContext();
Submit(pose, bank0);
return Complete(bank0);
```

The wrapper preserves the old call shape.

### Corrected greedy executor (bounded poll)

```text
for pose in input order:
    acquire a free context
    enqueue pose work on context.stream
    record a completion event

while contexts remain in flight:
    if cudaEventQuery(context.event) == cudaErrorNotReady:
        yield or sleep a bounded backoff (10-25 us, adaptive); # NOT a hot spin
        continue feeding other contexts
    if cudaEventQuery(context.event) == cudaSuccess:
        read only this context's pinned result
        store result at its original input index
        recycle context
    otherwise:
        abort the complete batch

when only one context remains: cudaEventSynchronize on that event instead.
```

### Architecture alternatives

| Option | Strength | Risk | Recommended use |
|---|---|---|---|
| Active-bank compatibility migration | Smallest code delta | Mutable aliasing, hidden ownership | Tactical prototype only |
| Explicit `EvaluationContext` / `EvaluationExecutor` | Clear ownership and lifecycle | Larger API migration | Recommended long-term |
| Data-oriented batch arrays / CUDA Graph | Highest upside | Larger change; packet boundary complicates capture | Later, after measurement |

## References

- `cuda-skill/references/cuda-guide/02-basics/asynchronous-execution.md`
- `cuda-skill/references/best-practices-guide/11.5-concurrent-kernel-execution.md`
- `cuda-skill/references/best-practices-guide/9.1-timing.md`
- `cuda-skill/references/best-practices-guide/4.1-profile.md`

Environment: project target is CUDA 12.9; during implementation the manifest declared 12.9 while active `nvcc --version` reported 13.2.

## Related

- `docs/solutions/performance-issues/jtml-graph-feeder-cudaeventquery-hotspin-2026-08-20.md` (plan 012 handle 2026-08-20)
- `test/golden/graph_performance_baseline.json`
- `src/compute/evaluation_executor.cpp`, `src/compute/evaluation_executor.cu`
- `docs/architecture/jtml-cost-evaluation-execution-graph.org`
- `docs/plans/2026-08-14-010-feat-direct-variants-capacity-launch-plan.org`
- `test/golden/cut0_measurement.md`
- `docs/handoff-2026-08-12-optimizer-path.md`

## Refresh

2026-08-20: this doc is a REPLACE-in-place corrected guidance for the CUDA `EvaluationExecutor` feeder. The corrected poll discipline (bounded backoff / `cudaEventSynchronize` / sweep-Done) replaces the earlier hot-spinning; admission (N) must saturate the GPU; measurement discipline requires GPU-busy% + timeline, never a spin-loop rate. All 14 cross-references in handoffs/plans still target this path.