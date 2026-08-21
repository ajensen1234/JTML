# P13 — Kernel performance lens: DIRECT_DILATION eval path

Lens: occupancy / launch config / register pressure / coalescing / bank conflicts / divergence.
Scope: per-kernel grid+block on the DIRECT_DILATION graph path; whether each saturates ~84 SMs; two-chain overlap; shared-buffer mutation (data hazard); the single most concurrency-limiting change; ncu metrics to confirm overlap.

## Source note

`/tmp/p013/p13` (the send file) does **not exist** on this host (`ls /tmp/p013` and `find /tmp -maxdepth 2 -name 'p13*'` both empty). The probe census that does exist is `test/golden/probe_measurement.md` (same run: nsys profile of `jtml_test_graph_throughput_oracle`, RTX 3090 Ti, commit c55a1dec, 12412-tri Kneel_1 at 1024x1024, dilation 6, DIRECT_DILATION monoplane graph). I reviewed that plus kernel bodies. If the send copy carried facts beyond the census, they were not available to me.

Census facts relied on here (probe_measurement.md:15-44):
- GPU busy ≈ **1.5 %** (47,532 kernels, 111 ms GPU sum over 7.57 s wall).
- Per-pose host floor ≈ **73 µs** (graph) vs ~104 µs serial wall; per-pose GPU residency ≈ **97 µs**.
- Per-launch GPU µs (total/count): FillTriangle 5711/3802≈1.5, StridePrefix 6017/2968≈2.0, PrepareLaunchPacket 5850/2968≈2.0, DilateKernel 19662/2968≈6.6, DifferenceKernel 25262/2968≈8.5, EdgeKernel 4485/2968≈1.5, ResetPixelScore 2877/2968≈1.0, DistanceMapMetric 9130/2968≈3.1, CUB DeviceScan 7285/2968≈2.5.

Device: RTX 3090 Ti → 84 SMs (probe "sm_86"; render_engine.cu:371 reads `cudaDevAttrMultiProcessorCount`). Blocks pinned at `threads_per_block` = 256 (16x16 edges for the 2D kernel). The DIRECT_DILATION graph = `EnqueueRenderPhase` + `EnqueueFastImplantDilationMetric` + `EnqueueDistanceMapMetric` (graph_recipe_direct_dilation.cu:77-90, 354-367), captured once, replayed per eval.

---

## A. Per-kernel grid/block, SM saturation, and two-chain overlap

### Render kernels (EnqueueRenderPhase → render_engine.cu)

| kernel | grid | block | resident | verdict |
|---|---|---|---|---|
| `cudaMemsetAsync` (1024x1024 output) | whole frame | — | memset engine | throughput-correct, once/eval |
| `ResetKernel` | 1x1 | 1 | 1 thread | latency-bound (render_engine.cu:1057) |
| `WorldToPixelKernel` | ceil(sqrt(3·tris/256))^2 ≈ 13x13 = 169 blocks | 256 | <84·4 | latency-bound (render_engine.cu:150-155,1060) |
| `BoundingBoxForTriangles` | ceil(sqrt(4·tris/256))^2 ≈ 14x14 = 196 blocks | 256 | <84·4 | latency-bound (render_engine.cu:161-165) |
| `BoundingBoxSizes` | ceil(sqrt(tris/256))^2 ≈ 7x7 = 49 blocks | 256 | <84 | latency-bound |
| `CUB DeviceScan` | tris=12412, scan all N | — | CUB internal | latency-bound |
| `PrepareLaunchPacket` | 1x1 | 256 | 1 thread | latency-bound (render_engine.cu:1596) |
| `OverflowCheck` | 1x1 | 256 | 1 thread | latency-bound (render_engine.cu:1638) |
| `StridePrefixPersistent` | min(maxActiveStride·84, ceil(maxStride/256)) | 256 | **occupancy-tuned to 84 SM** | **saturates** (render_engine.cu:374-382,408) |
| `FillTrianglePersistent` | min(maxActiveFill·84, ceil(fragments/256)) | 256 | **occupancy-tuned to 84 SM** | **saturates** (render_engine.cu:373-406) |

### Metric kernels (gpu_metrics.cu, fast_implant_dilation_metric.cu)

| kernel | grid (fixed-max) | block |
|---|---|---|
| `ComputeMetricCrop` | 1x1 | 1 (gpu_metrics.cu:360) |
| `${ResetPixelScore}` | 1×1 | 1 (gpu_metrics.cu:364, 474-477) |
| `EdgeKernel_Graph` | (kMaxMetricDim=2048 ÷ 16)^2 = **16,384 blocks** | 16x16=256 (gpu_metrics.cu:376-384) |
| `DilateKernel_Graph` | ceil(4·2048²/256) = **65,536 blocks** | 256 (gpu_metrics.cu:387-396) |
| `DifferenceKernel` | ceil(2048²/256) = **16,384 blocks** | 256 (gpu_metrics.cu:402-406) |
| `DistanceMapMetric_Kernel` | ceil(2048²/256) = **16,384 blocks** | 256 (gpu_metrics.cu:485-489) |

`kMaxMetricDim = 2048` (gpu_metrics.cu:315); frame is 1024, so every metric grid uses `max(2048, 1024) = 2048`.

### Saturation verdict

Overwhelmingly **latency-bound, not throughput-bound**:
- Nine of the render kernels use 1 thread or ≤ a few hundred blocks. Their ~1-2 µs/launch GPU times are essentially fixed launch/teardown + stream overhead, not SM-resident compute.
- The only true saturating kernels are FillTriangle and StridePrefix, both tuned for full-84-SM occupancy at init via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

**Metric fat-grid problem.** Edge/Difference/Distance launch 16,384 blocks and Dilate 65,536 blocks **every** eval, independent of the actual (small) crop. On ~84 SM × ~4 resident blocks (assumed; no `__launch_bounds__`), that is hundreds of early-retire waves. Each `_Graph` kernel already has the crop guard (fast_implant_dilation_metric.cu:224-228, 261, 293, 322), so nearly all launched blocks retire immediately inside the guard. Result: these kernels are NOT "a few SMs latency-bound" — they assert a whole-SM launch footprint with near-total early retire, most of it wasted dispatch.

**Two-reval-chain overlap.**
- The render core (World/BBox/Prepare/Overflow/ResetComputeCrop/ResetPixel) is so small it overlaps trivially into unused SMs. Not the limiter.
- The fat-grid metric kernels and Fill/Stride all target the same full-84 SM dispatch footprint. So two reval chains **compete for the same per-SM kernel-queue slots**, not fill idle SMs. And the probe's host-bound serialization (host ~73 us vs GPU ~97 us/eval) leaves no SM headroom for N=2 GPU overlap anyway — consistent with GPU busy 1.5%.

---

## B. Shared-buffer mutation and data hazards

Generic "FullFrame"/"DirtyMask" names map to these concrete device buffers:

1. **Rendered frame (the true shared output buffer).** Written by render, then same-address MUTATED in-place by the metric:
   - `FillTrianglePersistent`/`FillTriangle` writes output pixels (render_engine.cu:1186, 1684).
   - `EdgeKernel_Graph` writes `dev_image[loc] = EDGE_PIXEL;` (fast_implant_dilation_metric.cu:243).
   - `DilateKernel_Graph` writes `dev_image[location] = DILATED_PIXEL;` (fast_implant_dilation_metric.cu:279).
   Within one eval this is graph-ordered (safe). The hazard appears when **two evals alias the same frame buffer** and their Edge+Dilate kernels overlap the same pixels. Isolation is intended per-bank/per-context (`active_bank_->primary.output`, `ctx.primary.output`), so the invariant is: **a bank must not be reissued before its prior eval's Edge+Dilate+D2H tail has drained.** Any pool path that re-checks-out the same bank while the previous graph is still writing is a silent race on the frame.

2. **`dev_pixel_score_` scalar:** `DifferenceKernel` does `atomicAdd`/`atomicSub(&result[0],1)` on a single address (fast_implant_dilation_metric.cu:146-148, 304-307); ResetPixelScore(1,1) zeroes it. Two evals sharing it double-count.

3. **`dev_distance_map_score_` + `dev_edge_pixels_count_`:** DistanceMapMetric does **two atomicAdds per EDGE pixel** into two 1-int cells (distance_map_metric.cu:54-55; kernel :335-337), reset by single-thread kernels (distance_map_metric.cu:474-477).

4. **`dev_crop` (MetricCropParams):** written 1-thread by `ComputeMetricCrop`, read by the four metric kernels; ordered in-eval, cross-eval hazard only if two evals share it.

5. **Persistent-worker atomics** `nextCandidate`/`nextChunk`/`overflowFlag`: cleared async, consumed by Stride/Fill; single-context.

So the per-kernel mutation set is: Edge + Fill + Dilate all write the frame; Difference writes the score; the distance kernels write the two scalar counters. Item 1 is the "two evals hit the same pixels" data hazard, prevented only by bank-reissue ordering.

---

## C. Single most concurrency-limiting kernel change

Probe says the path is **host-bound** (cudaGraphLaunch 29.4 µs/pose + per-pose setParams 2.55 + pinned D2H 12.1 µs) and GPU busy 1.5 %. So no kernel-launch change alone can recover N=2 via SM saturation.

Still, the single most costly **kernel launch** is the fixed 2048² metric grid:
- `Dilate` 65,536 blocks and each of Edge/Difference/Distance 16,384 blocks, per eval, although the crop is a small device AABB.
- The crop bounds are already on-device (`MetricCropParams` written by `ComputeMetricCropKernel`, gpu_metrics.cu:360); the retire guards already use them — only the launch boundary is left fat.
- **Change: derive the metric `_Graph` grids from the device crop AABB (or the frame 1024) instead of `kMaxMetricDim=2048`.** On this 1024 fixture that removes a ~4× thread over-invocation and ~99% of the wasted wavefront dispatch every eval. Smaller `Dilate` and `Difference` grid → fewer SM slots claimed → the two chains actually overlap in the SMs they leave idle.

Confidence 75. Caveat (50): if the app ever runs frames at 2048 the constant is intentional; on this fixture (1024) it is pure over-invocation.

---

## 4. ncu metrics to confirm overlap

- **Per-kernel resident-block ceiling and true waves:** call `cudaOccupancyMaxActiveBlocksPerMultiprocessor` once for `FillTrianglePersistentKernel`, `StridePrefixPersistentKernel`, `FastImplantDilationMetric_{Edge,Dilate,Difference}_Graph`, `DistanceMapMetric_Kernel_Graph`. None of the bodies sets `__launch_bounds__`; `--ptxas options=v` (register count) is needed to project residency. Wave-per-launch = ceil(grid / (84 × resident-per-SM)).
- **Concurrent-kernel timeline:** both reval chains run on non-default per-bank streams (`ctx.stream`) — required for concurrency. Profile per-stream kernel windows with nsys (`--kernel-type` ranges / nvtx) and compare against host `cudaGraphLaunch` (29.4 µs) gaps. If per-kernel GPU residency is shorter than inter-launch, SMs are idle → overlap exists only if grids/fat/sizing are reduced.
- **Confirm the tail is the host, not the reducer:** `cudaEventQuery` avg 0.78 µs (hot-spin fixed) is fine; the cost is `cudaGraphLaunch` 29.4 + `setParams` + D2H 12.1 µs.
- **Hazard check out path:** put the sync/wait only on the metric tails; never reissue the same bank output until its Edge/Dilate tail has fully drained.

---

## Residual risks / parent notes

- `/tmp/p013/` send missing; census from `test/golden/probe_measurement.md`.
- Occupancy per SM assumed ~4 resident 256-thread blocks (no smem, low registers); register ceiling unverified because no `__launch_bounds__`. Confirm with `--ptxas options=v`.
- Not fully established whether the sampled DIRECT path uses `FillTrianglePersistent` / `StridePrefixRules` (grid ~= thousands of blocks, census Fill count 3802) or the legacy grid = ceil(fragments/256). Either keeps the 84 SM.
- Host floor (29.4 µs/launch) is the actual N=2 blocker; do not chase GPU kernel occupancy alone for reachability.

---

## Findings (JSON)

```json
{
  "findings": [
    {
      "lens": "cuda-kernel-perf",
      "title": "Metric kernels launch fixed 2048x2048 grids (16-65k blocks per eval), over-invoking the SM footprint ~4x and blocking two-chain overlap",
      "severity": "P1",
      "confidence": 75,
      "evidence": [
        "gpu_metrics.cu:315 `constexpr int kMaxMetricDim = 2048;`",
        "gpu_metrics.cu:376-396,485-489 grids sized from ceil(2048^2/256)=16384 blocks (Edge/Difference/Distance) and 4*2048^2/256=65536 blocks (Dilate)",
        "fast_implant_dilation_metric.cu:224-228,261,293,322 the in-kernel crop guards already retire threads past the actual AABB"
      ],
      "owner": "maintainer",
      "suggestedFix": "Size the _Graph metric grids from the already on-device MetricCropParams AABB (fast/or the frame 1024) instead of kMaxMetricDim=2048; the guards make the over-launch pure waste. This removes ~4x thread over-invocation per eval and lets two chains overlap in idle SMs."
    },
    {
      "lens": "cuda-kernel-perf",
      "title": "Edge then Dilate MUTATE the same in-place frame buffer the render writes — cross-eval hazard if a bank is reused while a prior graph is still draining",
      "severity": "P0",
      "confidence": 75,
      "evidence": [
        "fast_implant_dilation_metric.cu:243 `dev_image[loc] = EDGE_PIXEL;`",
        "fast_implant_dilation_metric.cu:279 `dev_image[location] = DILATED_PIXEL;`",
        "render_engine.cu:1186/1684 same frame is the render write target; graph_recipe_direct_dilation.cu:77-90 runs render+metric in one graph"
      ],
      "owner": "maintainer",
      "suggestedFix": "Guarantee a bank frame buffer is not reissued until the prior eval's Edge+Dilate tail has drained (no reuse while the previous eval's dense metrics still write it). Confirm per-bank isolation is absolute."
    },
    {
      "lens": "cuda-kernel-perf",
      "title": "Difference + DistanceMap do per-pixel atomicAdd into ONE scalar cell — heavy single-address contention, worse when two chains overlap",
      "severity": "P2",
      "confidence": 75,
      "evidence": [
        "fast_implant_dilation_metric.cu:146-148 atomicAdd/Sub on &result[0] per matched pixel",
        "distance_map_metric.cu:332-333 two atomicAdds per EDGE pixel into two single-cell 4-byte counters",
        "probe_measurement.md:23-24 Difference 8.5 us, Distance 3.1 us/launch — the reduction cell is the hotspot"
      ],
      "owner": "maintainer",
      "suggestedFix": "Block-local partial reduction into shared memory then one atomic per block (e.g. atomicAdd by block), collapsing the per-pixel single-address contention to (num_blocks) atomics per eval."
    }
  ]
}
```

---

## Residual risk recap

- Missing /tmp/p013/p13 send file; used the probe census from test/golden/probe_measurement.md.
- Exact FillTriangle grid (persistent vs legacy form) needs a probe route check; census count 3802 fits a fill grid of a few thousand blocks either way.
- Register ceiling unconfirmed (no `__launch_bounds__`). Confirm with `--ptxas options=v` before c-size the resident-block-per-SM ceiling used in the waves math.