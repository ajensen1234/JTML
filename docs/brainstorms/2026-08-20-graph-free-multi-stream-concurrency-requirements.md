---
date: 2026-08-20
topic: graph-free-multi-stream-evaluation-concurrency
---

# Graph-Free Multi-Stream Evaluation Concurrency

## Problem Frame

Plan 011–013 built and measured a CUDA-graph evaluation path for DIRECT_DILATION. Plan 013's
U0 probe delivered the decisive measurement: the graph path is **host-bound, not poll-bound**
— `cudaGraphLaunch` costs ~29.4 µs host-side per launch, the per-pose host floor is ~73 µs 
(≳ serial ~104 µs wall), GPU busy ~1.5%, and the corrected (paced, watchdog-porous) feeder
re-qualified as `reverted` (0.095×). The graph verdict is honest: **on this fixture, the
CUDA-graph machinery itself is a per-pose host tax**.

The exploration (5 CUDA persona lenses, call-stack grounded, source-verified) found the
decisive structural fact: **the graph is a thin `cudaStreamBeginCapture` wrapper around the
same three plain per-stream enqueues** (`EnqueueRenderPhase(EvaluationContext&)`,
`EnqueueFastImplantDilationMetric`, `EnqueueDistanceMapMetric` — all `<<<...,stream>>>` +
`cudaMemcpyAsync`). A graph-free multi-stream executor calls those same enqueues without the
capture window, overlaps N in-flight evals on N per-context non-blocking streams, and completes
via the event-poll machinery already built in plan 013 U1–U3. The kernels, buffers, streams,
pinned twins, and `completeFromPins` all carry over unchanged.

**The durable product capability is host-starvation removal via multi-stream overlap.** The
graph was one implementation of that; this brainstorm replaces the graph path with the
graph-free production executor that unlocks the same overlap at a strictly lower host floor and
without graph lifetime/admission/capture complexity.

---

## Actors

- A1. **Greedy evaluation feeder** — the host loop (today `EvaluationExecutor::RunBatch`)
  that checks out contexts, enqueues poses, polls completions, and returns ordered scores.
- A2. **Render + metric pipeline** — the existing kernels (`GPUModel`, `GPUMetrics`, CUB scans)
  that render a pose and compute the DIRECT_DILATION cost.
- A3. **Context pool / admission** — `EvaluationContextPool` + `bank_state_math::admit`, which
  owns per-context streams, events, buffers, and the half-memory admission rule.

---

## Key Flows

- F1. **Graph-free batch evaluation**
  - **Trigger:** `EvaluationExecutor::RunBatchWithCost` with ≥2 poses.
  - **Actors:** A1, A2, A3.
  - **Steps:** for each pose: checkout a free context → set pose on ctx → replay the three
    per-context enqueues on ctx.stream → `cudaEventRecord(ctx.completion_event, ctx.stream)`.
    Then poll all in-flight events (bounded/backoff, watchdog) → any Done →
    `completeFromPins(ctx)`. Recycle free contexts.
  - **Outcome:** ordered scores, input-order preserved, all converter sets event-based (no
    `cudaStreamSynchronize` on the admitted path).
  - **Covered by:** R1, R2, R3, R5.
- F2. **Multi-stream overlap**
  - **Trigger:** ≥2 contexts in flight (batch > pool size, or concurrent stages).
  - **Actors:** A2, A3.
  - **Steps:** eval B's kernels are enqueued on stream B while eval A's stream still runs the
    tail of its chain; the per-context events gate when B's results are read. GPU packs the
    inter-Kern overhead windows; host is not the binding constraint beyond N≈4–8.
  - **Outcome:** per-pose GPU-busy rises from ~1.5% to device-kernel-bound (~2.5–3× vs serial
    on the frozen fixture); no per-pose host packet barrier.
  - **Covered by:** R1, R4, R5.

---

## Requirements

**[Execution path — graph-free]**

- R1. Replace the graph `enqueueHook` (recipe->updateParams+`cudaGraphLaunch`) with a
  graph-free replay of the three per-context enqueues: render → fast metric → distance metric
  → `cudaEventRecord` on ctx.stream. The kernels, `completeFromPins`, pinned host results, and
  the bounded-pacing/event-poll discipline from plan 013 U1–U3 stay. Zero new kernels.
- R2 — The production executor must not use `cudaStreamSynchronize` on the admitted path. All
  completion goes through the per-context disable-timing completion event; the sole
  `cudaStreamSynchronize` remains only in the serial `RenderPhase(BankState&)` backward
  compat path (render_engine.cu:1401), which is not used for the concurrent path.
- R3 — Use the existing pure-enqueue render path
  (`EnqueueRenderContext(EvaluationContext&)`, render_engine.cu:U4 — "NO D2 copy / NO sync")
  — not the serial `RenderPhase(BankState&)` which ends in `cudaStreamSynchronize` at
  render_engine.cu:1398. This is the single structural change that makes overlap possible.

**[Memory / admission]**
- R4 — Memory is NOT the admission cap on this card. `admit()` gives N≈233 for the frozen
  fixture (bank_state.cuh:287-289; verified). The observed N=4 in every baseline came from the
  harness `makeGraphExec(4,...)` (graph_throughput_oracle_test.cu:411), not memory. The
  executor should admit N that saturates the GPU (≈4–8 for this fixture), not cap at 4.
- R5 — Keep geometry shared read-only (the mesh triangles/normals are already uploaded once and
  read via launch args — NO per-context mesh copy; "copy the STL" is a net loss). Only the
  per-eval mutable write-set is per-context, which the pool already allocates
  (evaluation_context.cpp:87-144).
- R6 — Batch the four per-eval pinned D2H copies (pixel/ distance/ edge/ overflow = ~16–20 B
  total) into one `cudaMemcpyAsync` of a per-context struct → removes ~3 calls ≈ **~36 µs/pose
  host** (measured 12.14 µs/call).
- R7 — The 40 MB `dev_stride_prefixes` buffer is write-only and unused by the persistent
  workers (render_engine.cu:1041-1043 `(void)dev_stride_prefixes`). Reduce
  `maximum_stride_size` (cuda_launch_parameters.h:10) to a safe per-fixture bound (e.g. 2 M →
  8 MB, or 512 K → 2 MB) after measuring the real fragment_fill span; every consumer uses the
  same constant so it is consistent; the existing overflow gate fails loud.

**[Measurement / gate]**
- R8 — Re-qualify with the same honest discipline as plan 013 U3: verified layered pass,
  NCU ≥30% concurrent at N=2, host-device gap <50 µs, trials ≥50, readback of the gold
  baseline. The throughput gate stays the frozen 1.20× N=2 vs serial N=1 (graph_pre_registration.json).
- C1 — Target N-scaling: N=2 → 1.2–1.7×; N=4 → 1.7–2.4×; plateau ~2.5–3× at N≈8 (device-kernel
bound). N=12 / N=100 / N=233 give no further speedup; only a memory tax. This is a hard
ceiling (≈31 µs/eval of GPU kernel time), NOT a tuning knob.

---

## Acceptance Examples

- AE1. **Covers R1, R2, R3.** Given a batch of 8 poses on the frozen Kneel_1 fixture, when the
  executor runs graph-free on N=4 streams, scores match serial (input-ordered, non-zero,
  distinct, within Layer-C tolerance), the nsys timeline shows ≥30% concurrent kernels at N=2,
  host-device gap <50 us, and the GPU busy fraction is ≥50% across the batch. No
  `cudaStreamSynchronize` in the admitted path. Verdict advances.
- AE2. **Covers C1** — N=8 throughput on the frozen fixture is ≥1.5× and ≤3.5× serial; N=32 is
  within noise of N=8 (plateau by device-boundness) — proving speedup is device-bound, not
  N-bound.

---

## Success Criteria

- The graph-free executor replaces the graph path in production and delivers honest paired
  throughput ≥ 1.2× at N=2 serial N=1 on the frozen 16-pose workload, with GPU busy >50% as
  the evidence that the device (not the host) is the constraint.
- A downstream planner can take this doc and plan the executor without inventing product
  behavior (the three-enqueue replay, event completion, admission N, memory routing, gates).

## Scope Boundaries

- **Non-goals:** No new kernels or changes to the rendering/metric algorithms. No
  `cudaMallocAsync` swap (the current one-time `cudaMalloc` pool init is fine; do not add
  lazy/hot-loop allocation). No multi-host-thread enqueue — the alias dance
  (`GPUModel`/`GPUMutator` member rebinding around each ctx enqueue) is safe only under a
  single feeder thread. No per-context mesh copy. No graph-capture reuse (the graph path is
  removed from production once the graph-free executor passes).

## Key Decisions

- **Graph-free replay is the executor:** the graph adds per-pose host tax (29.4 µs launch +
  2.55 µs setParams) with no throughput benefit over the same-enqueue replay; the win over
  serial comes from overlap (device residency ~97 µs per eval), not from bypassing graph launch
  serial comes from overlap (device residency ~97 µs per eval), not from bypassing graph launch
  cost.
- **Stream admission targets GPU saturation (~4–8), not max N:** more streams beyond the
  device-kernel-bound point (≈8) are a memory tax, not throughput.
- **The single structural enabler:** use the U4 pure-context render enqueue
  (EnqueueRenderPhase(EvaluationContext&)) as the actual kernel feed, not the serial
  BankState/sync path. That is what removes the host packet barrier.

## Dependencies / Assumptions

- The three `Enqueue*Context` functions (render + two metrics). Must remain zero-sync and use ctx.stream
  everywhere (code point verified in render_engine.cu:1495-1532 metrics overload).
- The `completeFromPins` function from the graph recipe is the validated no-sync reader; it
  must be reusable verbatim for plain launches (it reads only the per-context pinned twins,
  which the same `cudaMemcpyAsync` on ctx.stream fills).

## Outstanding Questions

- [ ] Measure fragment_fill on the real Kneel_1 fixture before choosing the stride
  cap (2 M vs 512 K) — the overflow gate protects correctness, but a too-small bound would
  abort on larger implants.
- [ ] Is the distance-metric pixel score (int) reduced by two different atomic paths between
  serial and graph paths, or is the result identical? (Plan-013 U6 layer-B bit-exactness was
  validated for the graph path; confirm the same for the graph-free replay.) → carries into
  planning, not out-of-scope.

## Next Steps

- **Step 1 — N=2 overlap POC:** a standalone harness (no rewrite): enqueue eval A + eval B on
  two streams from one thread, poll both events, measure the 2-window vs 194 µs; the f = 1 −
  win/(2×97) gate before building the full executor.
- **Step 2:** If f ≥ 0.3 (i.e., N=2 win ≤ 135 µs), plan the graph-free executor (plan 014).
- **Step 3 — Default-deny until evidence:** default-deny stays until the machine-qualified, layered-verified
  retained verdict; the graph path remains the reference oracle.