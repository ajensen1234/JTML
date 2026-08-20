# U0 Probe Measurement — plan 013 (2026-08-20)

**Run:** `nsys profile --stats=true -o /tmp/u7_u0_probe .build/bin/jtml_test_graph_throughput_oracle`
**Device:** NVIDIA GeForce RTX 3090 Ti (sm_86), driver 610.57.04, CUDA 12.9, commit c55a1dec
**Fixture:** 12412-tri Kneel_1, 1024x1024, dilation 6, monoplane DIRECT_DILATION graph

## Verdict: `probe=unreachable` — 1.20× at N=2 is OUT of device reachability

The corrected feeder (U1/U2 pacing, 0.78 µs/query) removed the hot-spin, but the graph
path is host-bound at ~80–90 µs/pose (see below), ≳ the serial floor. Per plan U0 the
reachability predicate (host floor ≤ ~12 µs, or N=2 SM-overlap f ≥ 0.28) FAILS.

## Kernel census (anti-stub: REAL graph work, all expected kernels)

| kernel | count | total (µs) |
|---|---|---|
| FillTriangleKernel | 3802 | 5711 |
| StridePrefixKernel | 2968 | 6017 |
| PrepareLaunchPacketKernel | 2968 | 5850 |
| DilateKernel | 2968 | 19662 |
| EdgeKernel | 2968 | 4485 |
| ResetPixelScoreKernel | 2968 | 2877 |
| DifferenceKernel | 2968 | 25262 |
| DistanceMapMetric_Kernel | 2968 | 9130 |
| CUB DeviceScanKernel | 2968 | 7285 |

47532 kernels total; sum 111 ms across 7.57 s wall → **GPU busy ~1.5 %**.

## Host-side floor (CUDA runtime API, nsys census)

| API | count | avg µs/call | total host ms |
|---|---|---|---|
| **cudaGraphLaunch** | 8904 | **29.41** | **261883** |
| cudaGraphExecKernelNodeSetParams | 8904 | 2.55 | 22738 |
| cudaMemcpy (D2H pins) | 14877 | 12.14 | 180672 |
| cudaLaunchKernel | 48330 | 3.66 | 176936 |
| cudaEventQuery | 233259 | 0.78 | 181334 |
| cudaEventRecord | 8904 | 1.34 | 11898 |

Per-eval host ≈ 29.4 (graph launch) + 2.55 (setParams) + 12.1 (pinned copy) + ~29
(8 kernel launches × 3.66) ≈ **~73 µs/pose host**, vs serial ~104 µs/pose wall with
~97 µs GPU residency. The graph path does NOT lower the host floor; cudaGraphLaunch
remains ~29 µs host-side per launch, and per-context concurrency cannot hide a
host-bound floor behind a 97 µs GPU residency at N=2.

## Conclusions (honest, per plan U0 + compound prevention rules)

1. The hot-spin fix (U1/U2) is real: cudaEventQuery avg 0.78 µs, not the former
   GHz-cadence spin; the sole-tail is bounded and watchdog-porous.
2. **The graph path is host-bound, not poll-bound.** `cudaGraphLaunch` at ~29 µs host
   per launch + per-pose setParams + D2H copies puts the graph host floor at or above
   serial. GPU busy ~1.5 % confirms the device idles because the *host* feeds it
   one pose at a time — concurrency at N=2 (admitted 4) cannot amortize this.
3. **1.20× at N=2 is unreachable on this fixture/machine.** Recorded `reverted`
   (benefit 0.095×) in `graph_performance_baseline.json` is the honest device
   outcome, not a poll artifact. Default-deny stays.
4. Follow-up levers (deferred to plan 016 / future): cut per-pose `cudaGraphLaunch`
   host cost (e.g. one graph with batched params, or cudaGraphUpload + multi-param
   update), reduce pinned D2H per pose, or move host work off the critical path —
   on a machine/fixture where host and device floors allow overlap.

## Registered evidence
- kernel_census: 47532 real kernels; GPU busy 1.5%
- host_floor_per_pose_us: ~73 (graph), ~104 (serial wall), ~97 (GPU residency)
- reachability_predicate: FAIL (host floor ~73 µs ≫ 12 µs bound; f unmeasurable at
  N=2 because both contexts are host-starved)