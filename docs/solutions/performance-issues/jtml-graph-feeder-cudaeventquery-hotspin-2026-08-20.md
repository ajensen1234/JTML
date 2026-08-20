---
title: "JTML greedy CUDA-graph feeder hot-spins cudaEventQuery: U7 benchmark verdict is host-bound, not a graph-capability limit"
date: 2026-08-20
category: docs/solutions/performance-issues
module: JTML compute executor (EvaluationExecutor CUDA feeder)
problem_type: performance_issue
component: tooling
severity: high
symptoms:
  - "nsys verified GPU busy at 1.6%: only ~27.6ms of kernels executed across a 1764ms span, ~2.36us average kernel on the 12412-triangle Kneel_1 mesh"
  - "nsys API profile counted 1,526,320 cudaEventQuery calls = 81.8% of all CUDA API time; host launch-to-launch gap p50 = 712us"
  - "graph-greedy measured 0.114x serial throughput; graph_performance_baseline.json records verdict \"reverted\", reason \"gate failed: benefitOk=0 wallOk=0 p99Ok=0\" (benefit_N2_vs_serial 0.151855 on batch16)"
  - "graph N=2/N=4 arms collapse evals/sec vs serial N=1: batch16 serial_eps 9009.32 vs gN2_eps 1368.11"
root_cause: async_timing
resolution_type: code_fix
tags: [cuda, cuda-event-query, hot-spin, greedy-feeder, throughput, performance, plan-012, graph-executor]
related_components:
  - compute
  - oracle-test
  - golden-baseline
---

# JTML greedy CUDA-graph feeder hot-spins cudaEventQuery: U7 benchmark verdict is host-bound, not a graph-capability limit

## Problem

Plan 012 U7's greedy CUDA-graph `EvaluationExecutor` feeder replaces blocking completion with a busy spin-poll on `cudaEventQuery` in its hook-driven poll loop, so the host burns a full driver round-trip per poll while the GPU sits mostly idle. The measured 0.114x graph-vs-serial result — recorded as a `reverted` verdict in `test/golden/graph_performance_baseline.json` — is a **host-bound measurement artifact, not a fundamental concurrency limit**: nsys shows roughly 60x GPU headroom, so the fix is to wait properly (and raise admitted N) instead of concluding the CUDA-graph approach cannot deliver throughput.

## Symptoms

Live nsys-verified measurement (Kneel_1 `12412`-triangle implant at `1024x1024`, dilation 6):

- `1,526,320` `cudaEventQuery` calls = **81.8% of all CUDA API time**. The host thread is busy posting non-blocking status queries, not launching or reading work.
- GPU busy only **1.6%**: only ~`27.6ms` of kernels executed over a `1764ms` span; **~2.36us average kernel**. A single DIRECT_DILATION cost eval is microseconds of device work.
- Launch-to-launch host gap **p50 = 712us**: the host takes far longer to notice a completion and launch the next eval than the GPU takes to execute it.
- Measured graph-vs-serial = **0.114x** (i.e. roughly 8+ times slower) → harness gate recorded `reverted`.

Frozen golden artifact `test/golden/graph_performance_baseline.json`:

- `test/golden/graph_performance_baseline.json:13` — `"verdict": "reverted"`
- `test/golden/graph_performance_baseline.json:14` — `"reason": "gate failed: benefitOk=0 wallOk=0 p99Ok=0"`
- `test/golden/graph_performance_baseline.json:15` — `"benefit_N2_vs_serial": 0.151855`
- Per-workload arm collapse — `test/golden/graph_performance_baseline.json:10` (16-pose): `"serial_eps": 9009.32` vs `"gN2_eps": 1368.11`, `"gN2_p99_us": 923.213`.

Admission is also a factor: the max arm admits only `N=4` on this small fixture (`graph_performance_baseline.json:10` `"admitted_N": 4`), so even a correct wait can overlap only four in-flight evals per wave.

## What Didn't Work

- **Graph-N=1-as-serial-proxy, and trusting `reverted` at face value.** The harness planned graph N=1 as a launch-overhead isolation arm, but the `reverted` verdict was also weighted by the N=2/N=max arms. With `admitted_N=4` and the spin-poll dominating wall time, none of the graph arms actually measured graph concurrency — they measured the feeder's host spin cost plus low admission. Since nsys showed 1.6% GPU busy and large headroom, treating `reverted` as proof that graph admission cannot help was wrong.
- **Prior stream-0 CUDA-event timing bug (~0.0015ms).** The earlier harness timed graph arms with `cudaEventRecord(event, 0)` on the default stream 0 while graphs run on per-context non-blocking streams — producing a suspiciously impossible 0.0015ms and a falsely-fast N=1 arm. Fixed by moving to host `std::chrono::steady_clock` wall around the synchronous `RunBatchWithCost`.
- **Optimizing the wrong variable.** Prior attempts tightened threshold bookkeeping and event/stream placement rather than the host wait mechanism and the admitted-N ceiling. The measured 0.114x was read as an architecture verdict when the data said the opposite.

## Solution

**Resolved 2026-08-20 (plan 013):** the hot-spin was replaced with an injectable bounded
pacing hook (default yield; CUDA installer = 10 µs bounded sleep), a watchdog-porous
sole-context wait (no `cudaEventSynchronize` anywhere on the admitted path), and the U7
harness now gates `retained` on layered-verdict + NCU/nsys census + readback. The
re-qualification honestly records **`reverted` (0.095×)** — the U0 probe showed the graph
path is **host-bound**, not poll-bound: `cudaGraphLaunch` is ~29.4 µs host-side per
launch, per-pose host floor ~73 µs ≳ serial, GPU busy ~1.5 %. The wait fix was necessary
and is real (cudaEventQuery 0.78 µs avg, no hot-spin); it was simply not sufficient
because the bottleneck moved to the per-pose host launch cost. Default-deny stays.

Follow-up levers (plan 016 / future): cut per-pose `cudaGraphLaunch` host cost (batched
params / multi-param update / `cudaGraphUpload`), reduce per-pose pinned D2H, or move
host work off the critical path on a fixture where host and device floors allow overlap.
Evidence: `test/golden/probe_measurement.md`, `test/golden/graph_performance_baseline.json`.

Replace the hot-spin `cudaEventQuery` poll with a proper event wait or backoff so the host thread is not busy-synchronizing at GHz cadence, then raise admitted N and re-measure.

Exact hot-spin sites (quoted from source):

- `src/compute/evaluation_executor.cpp:298-300` — the greedy loop polls every in-flight lease every iteration:
```cpp
for (auto cur : inFlight) {
    PollResult pr = pollHook_(cur.ctxIdx);
```
- `src/compute/evaluation_executor.cpp:329-338` — when no lease completes, the loop has only `std::this_thread::yield()` plus the watchdog between iterations — no sleep/backoff, so it re-polls everything immediately:
```cpp
inFlight = std::move(survivors);
if (!anyDone) {
    if (std::chrono::steady_clock::now() - watchdogStart > watchdogTimeout_) {
        result.clear();
        ...
        return BatchOutcome::WatchdogPoisoned("watchdog expiry");
    }
    std::this_thread::yield();
}
```
- `src/compute/evaluation_executor.cu:34-42` — the installed pollHook is the busy `cudaEventQuery` tri-state call:
```cu
exec.InstallPollHook([&exec](size_t ctxIdx) -> PollResult {
    ...
    cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(ctx->completion_event);
    cudaError_t q = cudaEventQuery(ev);
    if (q == cudaSuccess) return PollResult::Done;
    if (q == cudaErrorNotReady) return PollResult::Pending;
    return PollResult::Error;
});
```

Fix directions (in priority order), all at the `pollHook_` / greedy loop site:

1. **Blocking wait on a completion event.** Replace `cudaEventQuery` with `cudaEventSynchronize(ev)` for the serial-feeder step so the host actually blocks until the device completes instead of spamming status calls.
2. **Bounded sleep/yield backoff on `Pending`.** When a lease returns `NotReady`/`Pending`, sleep 50-500us (or `std::this_thread::sleep_for` backoff) before re-polling. That collapses ~1.5M driver round-trips to a few dozen per completion.
3. **Multi-event batch + process-completed-while-others-run.** Submit a full work slice, then sweep a small set of events and handle only those `Done` while the rest keep running, with backoff between sweeps. Never hot-poll the entire lease set at zero delay.
4. **Raise admitted N.** With `admitted_N` clamped at 4 (`bank_state_math::admit` half-memory ceiling including graph-overhead bytes), only four in-flight evals can overlap, so larger batches serialize into waves — no wait change can show concurrency until admission rises (realistic/larger fixture, or the ceiling relaxed after a graph-VRAM probe).

Then re-run the U7 paired harness `test/oracle/graph_throughput_oracle_test.cu` and regenerate `test/golden/graph_performance_baseline.json`. The `retained` vs `reverted` gate is meaningful only once GPU-busy is the constrained resource.

## Why This Works

`cudaEventQuery` is a host->driver round-trip: each call threads into the CUDA driver to ask whether an event is ready, and while every eval is still in flight the host issuing these calls (~0.5us each in measured conditions) never returns to launch the next one in time. The device is only ~1.6% busy — the gap starves in the opposite direction, with the GPU idle waiting for a quieter host. Replacing the spin with a blocking wait or backoff turns the same 27.6ms of kernels into a tightly packed launch queue; since the measured launch-to-kernel ratio leaves large headroom, utilization climbs from ~1.6% without device changes.

## Prevention

- **Measure GPU busy% (or a kernel-active window) before accepting a performance verdict.** A verdict computed from a host-spin loop measures the host, not the capability. Ensure nsys shows the device is the constraint (kernels contiguous, not ~27.6ms spread across 1764ms).
- **Use event-driven/backoff polling for host-bound CUDA completion.** Never busy-poll an API with a driver round trip per poll; prefer blocking on one event or bounded sleeps between re-polls.
- **Never treat a host-bound measurement as a graph-capability verdict.** Separate "admission too low" and "spin-poll overhead polluted the timing" (both present here) from "CUDA graphs cannot deliver." Gate on GPU busy + Amdahl consistency, not evals/sec from a spin-loop baseline.

## Residual Risks

- Raising `admitted N` on the current small fixture remains VRAM-ceiling bound; if the fixture stays ~12400 triangles, a larger scratch device or framework change is required to demonstrate overlap, not just the wait fix.
- Backoff polling adds latency when events complete quickly; tune the sleep bound below the per-eval wall target.
- Blocking `cudaEventSynchronize` can erode concurrency if reused on a shared/legacy stream; the plan's capture-invalidator rule (never invoke it on per-context non-blocking streams) applies directly.

## Related Issues

- `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md` — the blueprint that prescribed the `cudaEventQuery` tri-state poll (no backoff); **refresh candidate** (HIGH overlap).
- `docs/solutions/logic-errors/jtml-cuda-graph-stub-failure-2026-08-19.md` — anti-stub verification protocol; keeps this verdict honest.
- `docs/solutions/architecture-patterns/graph-tiered-correctness-2026-08-19.md` — stays **aspirational** until a real retained measurement exists.
- `test/golden/graph_performance_baseline.json` — the frozen `reverted` verdict artifact that must be regenerated after the wait/backoff fix + N raise.
