---
title: "JTML CUDA cost evaluation: prefer explicit evaluation contexts and an executor"
date: 2026-08-17
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

The Cut-0 measurement is a gate, not a speedup result. The latest artifact records roughly 98 microseconds of CPU host time and 98 microseconds of GPU-event time per evaluation. The same workload must still be timed serially versus N-way before claiming improvement.

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

**Target design:** Across contexts, use non-default streams and event polling. `cudaEventQuery` or `cudaStreamQuery` should distinguish `cudaErrorNotReady` from real errors. Do not use `cudaEventSynchronize` in the feeder when the goal is to keep other contexts moving. The current implementation does not yet satisfy this target everywhere; that gap is the reason for the new EvaluationExecutor workstream.

## Why This Matters

CUDA streams express concurrency but do not guarantee it. NVIDIA's CUDA Programming Guide and Best Practices Guide state that multiple non-default streams can execute concurrently when device resources permit. A stream count is not a throughput result.

The host-dependent fragment packet boundary is a structural serialization point: the host must receive the fragment count before choosing the final fill grid. Other contexts can run while one context waits at that boundary, but the boundary limits the parallel fraction.

Use Amdahl's law for expectations:

```text
S(N) = 1 / ((1 - P) + P/N)
```

For two concurrent contexts, ideal upper bounds are approximately:

| Overlappable fraction P | Ideal N=2 speedup |
|---:|---:|
| 0.50 | 1.33x |
| 0.75 | 1.60x |
| 0.90 | 1.82x |
| 1.00 | 2.00x |

The actual `P` must be measured. The Cut-0 GPU/CPU ratio must not be reported as N-way speedup.

## When to Apply

Apply this architecture when:

- one GPU-backed evaluation object currently owns mutable scratch state;
- a batch contains independent evaluations that could overlap;
- the existing public or domain contract should remain synchronous;
- multiple streams, events, and pinned result buffers are required;
- correctness depends on keeping per-evaluation outputs and reductions isolated.

Use the simpler compatibility-bank migration only as a bounded transitional step, such as a monoplane `DIRECT_DILATION` experiment. Revisit it before adding more metrics, biplane support, host-thread concurrency, or performance claims.

## Examples

### Serial compatibility wrapper

```cpp
EvaluationContext bank0 = MakeCompatibilityContext();
Submit(pose, bank0);
return Complete(bank0);
```

The wrapper preserves the old call shape while routing through the explicit context internally.

### Greedy executor

```text
for pose in input order:
    acquire a free context
    enqueue pose work on context.stream
    record context completion event

while contexts remain in flight:
    if cudaEventQuery(context.event) == cudaErrorNotReady:
        continue feeding other contexts
    if cudaEventQuery(context.event) == cudaSuccess:
        read only that context's pinned result
        store result at its original input index
        recycle context
    otherwise:
        abort the complete batch
```

### Architecture alternatives

| Option | Strength | Main risk | Recommended use |
|---|---|---|---|
| Active-bank compatibility migration | Smallest code delta | Mutable aliasing and hidden ownership | Tactical prototype only |
| Explicit `EvaluationContext` / `EvaluationExecutor` | Clear ownership and lifecycle | Larger API migration | Recommended long-term design |
| Data-oriented batch arrays / CUDA Graph | Highest possible upside | Largest change; packet boundary complicates capture | Later, after profiling |

CUDA references used for this guidance:

- `cuda-skill/references/cuda-guide/02-basics/asynchronous-execution.md`
- `cuda-skill/references/best-practices-guide/11.5-concurrent-kernel-execution.md`
- `cuda-skill/references/best-practices-guide/9.1-timing.md`
- `cuda-skill/references/best-practices-guide/4.1-profile.md`

The project target is CUDA 12.9. During the implementation session, the project manifest declared 12.9 while the active `nvcc --version` reported 13.2; resolve or document that environment discrepancy before relying on version-specific performance conclusions.

## Related

- `docs/architecture/jtml-cost-evaluation-execution-graph.org`
- `docs/plans/2026-08-14-010-feat-direct-variants-capacity-launch-plan.org`
- `test/golden/cut0_measurement.md`
- `docs/handoff-2026-08-12-optimizer-path.md`
