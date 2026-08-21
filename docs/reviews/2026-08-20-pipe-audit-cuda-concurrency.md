# P2 — Per-eval Kernel Launch Chain & Synchronization Map (CUDA concurrency lens)

**Scope (README was read, not tests):** `src/compute/render_engine.cu`, `src/compute/fast_implant_dilation_metric.cu`, `src/compute/distance_map_metric.cu`, `src/compute/gpu_metrics.cu`, `src/compute/CostFunctionManager.cpp`, plus the harness that actually drives per-eval concurrency: `src/compute/evaluation_executor.cu`, `src/compute/evaluation_executor.cpp`, `src/compute/evaluation_context.cpp`, `src/compute/graph_recipe_direct_dilation.cu`.

**Reference basis (skill-local snapshots, CUDA 13.3):**
- `cuda-guide/02-basics/asynchronous-execution.md` §2.5.6.1 *Legacy Default Stream*: the default (NULL, stream-0) stream is **blocking**; work on it synchronizes with all other blocking streams. §2.5.6/§2.5.5: two ops on different streams cannot overlap if any NULL-stream op is submitted in between **unless the streams are `cudaStreamNonBlocking`**.
- `best-practices-guide/11.5-concurrent-kernel-execution.md`: non-default streams are required for concurrent kernels; a default-stream launch begins only after all prior device work (any stream) completes. `concurrentKernels` capability required.

**Key structural fact.** `EvaluationContextPool::Initialize` creates each context's stream with `cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)` (`evaluation_context.cpp:188`). So **each eval context owns its own independent non-blocking stream** — the precondition for two evals to overlap already exists. `completeFromPins` + event-query completion is present. The remaining question is only which *sync sites* still sit between two enqueues and serialize the host.

---

## A. Per-eval launch chain, kernel by kernel, with STREAM

### A1. Graph path (U4/U6 — the overlap-ready architecture)

Driven by `EvaluationExecutor::RunBatchWithCost` hook loop (`evaluation_executor.cpp:253-…`), with hooks installed by` CUDA glue` (`evaluation_executor.cu:14-70`). A batch prepares `count` contexts once (`Prepare`, `evaluation_executor.cpp:98-169`; one `createGraph` per context — not per eval), then the hot loop per pose does:

`Checkout(ctx)` → set pose → `updateParams(exec, ctx)` → **`recipe->launch(exec, ctx.stream)`** (→ `cudaGraphLaunch(exec, ctx.stream)`, `graph_recipe:453-455`) → **`cudaEventRecord(completion_event, ctx.stream)`** (`evaluation_executor.cu:31`) → poll all in-flight via **`cudaEventQuery(ev)`** (`non-blocking`, tri-state Done/Pending/Error) → `completeFromPins(*ctx)` → `Recycle(idx, true)`.

The graph body itself *is exactly the pure-enqueue replay body* (`EnqueueProductionSet`, `graph_recipe:63-93`; captured at `createGraph:348-371`): `EnqueueRenderPhase(ctx)` then the two metric enqueues. Per launch, all on **`ctx.stream`** (the context's own non-blocking stream):

**Render stage** — `RenderEngine::EnqueueRenderPhase(EvaluationContext)` (`render_engine.cu:1475-1716`):
| # | Op | Grid | Stream |
|---|----|------|--------|
| 1 | `cudaMemsetAsync(output…)` | (full frame) | ctx.stream |
| 2 | `ResetKernel` | `1×1` | ctx.stream |
| 3 | `WorldToPixelKernel` | `dim_grid_vertices_ × threads_per_block` | ctx.stream |
| 4 | `BoundingBoxForTrianglesKernel` | `dim_grid_bounding_box_ × threads_per_block` | ctx.stream |
| 5 | `BoundingBoxSizesKernel` | `dim_grid_triangles_ × threads_per_block` | ctx.stream |
| 6 | CUB `DeviceScan::ExclusiveSum(…)` | internally 1+ kernels over `cub_storage` | ctx.stream (stream passed) |
| 7 | `PrepareLaunchPacketKernel` | `1×1` | ctx.stream |
| 8 | `cudaMemsetAsync(nextCandidate)` | `1 int` | ctx.stream |
| 9 | `cudaMemsetAsync(nextChunk)` | `1 int` | ctx.stream |
| 10 | `cudaMemsetAsync(overflowFlag)` | `1 int` | ctx.stream |
| 11 | `OverflowCheckKernel` | `1×1` | ctx.stream |
| 12 | `StridePrefixPersistentKernel` | `persistent_stride_blocks_ × threads_per_block` | ctx.stream |
| 13 | `FillTrianglePersistentKernel` | `persistent_fill_blocks_ × threads_per_block` | ctx.stream |
| 14 | `cudaMemcpyAsync` overflowFlag→`host_overflowFlag` (pinned) D2H | — | ctx.stream |

**Metric phase** — `GPUMetrics` Enqueue `(EvaluationContext)` overloads, same `ctx.stream`, launched from the same graph body:
15. `ComputeMetricCropKernel` `1×1` (crop derived on device) (`gpu_metrics.cu360`)
16. `FastImplantDilationMetric_ResetPixelScoreKernel` `1×1` (`:364`)
17. `FastImplantDilationMetric_EdgeKernel_Graph` `16×16 blocks`, fixed-max grid (`:380`)
18. `FastImplantDilationMetric_DilateKernel_Graph` 1-D `dilate_grid` (`:392`)
19. `FastImplantDilationMetric_DifferenceKernel_Graph` 1-D `diff_grid` (`:402`)
20. `cudaMemcpyAsync` pixel_score D2H (`:415`)
21. `ComputeMetricCropKernel` `1×1` (`:470`)
22. `DistanceMapMetric_ResetPixelScoreKernel` `1×1` **×2** (`:474,476`)
23. `DistanceMapMetric_Kernel_Graph` `km_grid × km_threads` (fixed max grid) (`:487`)
24. `cudaMemcpyAsync` ×2 (distance_map_score, edge_pixels_count) D2H (`:498,509`)
25. `cudaEventRecord(completion_event, ctx.stream)`

Every single op (1–25) is launched on **`ctx.stream`**. **No default-stream launch exists anywhere in the admitted graph path.** Two eval contexts → two independent non-blocking streams → their kernel chains can execute concurrently. This is the overlap being requested, and it is already what the code constructs.

### A2. Serial / bank path (the non-overlap fallback) — same kernels, but on a **shared** bank stream with sync

- `CostFunctionManager::EvaluateDirectDilationOnBank` → `RenderPrimaryCamera(bank)` → **`EnqueueRenderPrimaryCamera`** calls `RenderPhase(BankState)` (`render_engine.cu1293-1397`) which ends with **`cudaStreamSynchronize(stream)`** (`render_engine.cu:1397`). This is the host barrier inside render, on **`bank.stream`**.
- Then `CompleteRenderPrimaryCamera` → `CompleteRenderPhase(BankState)` (`:1400-1448`) reads the pinned `host_fragment_fill`, launches `StridePrefixKernel` (`:1424`) + `FillTriangleKernel` (`:1438`) on the same bank stream; no host sync — but the *metrics* complete syncs follow:
  - `EnqueueFastImplantDilationMetric(stream)` + **`CompleteFast…(stream)`** with `cudaStreamSynchronize` (`fast_implant_dilation_metric.cu:532`)
  - `EnqueueDistanceMapMetric(stream)` + **`CompleteDistanceMapMetric(stream)`** with `cudaStreamSynchronize` (`distance_map_metric.cu:179`)

So the bank path is 3 host syncs per eval (render-finalize, fidm, distmap) and any scheduling that interleaves two evals on one host thread has them strictly serialized.

---

## B. Every point the CURRENT code serializes — the "walls"

| # | Wall | File:line | Type |
|---|------|-----------|------|
| W1 | `cudaStreamSynchronize(stream)` at end of `RenderPhase(BankState)` | render_engine.cu:1397 | host wait on bank stream after render+bbox/fragment D2H |
| W2 | `cudaStreamSynchronize(stream)` in `CompleteRenderPhase(EvaluationContext)` | render_engine.cu:1731 | host wait — sync wrapper / fallback |
| W3 | `cudaStreamSynchronize(stream)` in `CompleteFastImplantDilationMetric` | fast_implant_dilation_metric.cu:532 | host wait on metric stream |
| W4 | `cudaStreamSynchronize(stream)` in `CompleteDistanceMapMetric` | distance_map_metric.cu:179 | host wait on metric stream |
| W5 | Legacy `FastDestinationMapDilationMetric` host-scalar: kernels on **DEFAULT stream** + blocking `cudaMemcpy` D2H | fast_implant_dilation_metric.cu:351/399/423/454 + 470 | default-stream + host block (device-wide per §A ref) |
| W6 | Legacy `ComputeWhitePixelSum` synchronous: default-stream kernels + blocking `cudaMemcpy` | gpu_metrics.cu:152/165/170 (see also distance:118,123) | default-stream + host block |
| W7 | Legacy synchronous `GpuModel::Render()` trunk (blocking `cudaMemset` :1051, default-stream kernels :1057-1170, blocking `cudaMemcpy` :1116/1121/1206/1234) | render_engine.cu:1048-1240 | default-stream + host block |
| W8 | **Default-stream one-time pump** `ComputeDestinationWhitePixels` (called inside `createGraph` once, `graph_recipe:397`, hunk gpu_metrics:146-178; launches `ComputeWhitePixel…<<<1,1>>>`/`WhiteSum<<<grid,256>>>` on **default stream** and a blocking `cudaMemcpy`) | gpu_metrics.cu:152/165/170 | *one-time per graph at Prepare*, but pumped on the default stream → device-wide fence while it runs |

**Wall semantics.** W1–W4 block the host until that eval's stream drains; with a single driver thread the enqueue of eval B cannot start until eval A's stream sync returns, so **two evals cannot overlap.** W5–W7 are device-wide (default stream blocks every other stream, per §2.5.6.1) plus host-blocking copies. W8 is a one-time fence at Prepare, outside the per-eval loop — the real hot-loop overlap is not compromised by it, but it does fence the whole device once at batch setup.

---

## C. Minimum change set to make two evals overlap

The overlap path is *already constructed by the code*; the change set is therefore mostly **routing**, not new kernels:

1. **Use the graph completion, not the sync wrappers, in the hot loop.** For each eval: `Checkout(ctx) → set pose → updateParams → cudaGraphLaunch(exec, ctx.stream) → cudaEventRecord(ev, ctx.stream)` and complete only after `cudaEventQuery(ev)==Done` via **`completeFromPins(*ctx)`** — reading the already-D2H-pinned `host_overflowFlag / host_pixel_score / host_distance_score / host_edge_count` directly.
   - Do **not** call `CompleteRenderPhase(EvaluationContext)` (its `cudaStreamSynchronize` at render_engine.cu:1731 is exactly the W2 wall).
   - Do **not** call `complete()` on the recipe (it still `cudaStreamSynchronize`es at graph_recipe:465) — `completeFromPins` is the sync-free branch.
   - The poll-one-`lease` (`evaluation_executor.cpp:175-190`) and the "poll ALL in-flight for OOO completion" path (`:298-331`) already exist and are the correct shape.

2. **The pure-enqueue replay body is already the right shape.** `EnqueueRenderPhase(EvaluationContext)` + `EnqueueFastImplantDilationMetric(ctx)` + `EnqueueDistanceMapMetric(ctx)` (the body captured at `graph_recipe:354-367` / `EnqueueProductionSet:63-77`). This is the shape for a graph-free replay as well: enqueue these three on `ctx.stream`, event-record, poll, completeFromPins. **No extra sync must be inserted between them.**

3. **Push render D2H tails off-stream / make them pinned-only.** The `bbox`/`fragmentFill` D2H on the graph tails (`render_engine.cu jeweler async tails 1383-1391 / 1701`) — keep them `cudaMemcpyAsync` **on ctx.stream** into **pinned** host buffers; completion is demarked by the stream event. Do not let any host-read of those tails occur while the graph is in flight (that is what would force a `cudaStreamSynchronize`).

4. **Never put a default-stream op inside the eval-admit window.** Keep `Compute…WhitePixelSum` (standing default-stream default stream + blocking memcpy) confined to `Prepare` (it already is). Do not introduce a default-stream launch or a host memcpy/`cudaDeviceSynchronize` between the `cudaEventRecord` of eval A and the `cudaGraphLaunch` of eval B.

5. **Buffer the pool.** Overlap width = number of simultaneously checked-out non-blocking contexts. The greedy loop already refills the flight buffer up to pool capacity before polling (`evaluation_executor.cpp:254-330`), so with pool ≥2 the two evals launch back-to-back on two different streams and only then the poll pass drains. This is the overlap-ready invariant; preserve it.

**Exact Completes-to-convert-to-EventQuery+completeFromPins:**
- `RenderEngine::CompleteRenderPhase(EvaluationContext&)` (render_engine.cu:1723) — bypass when running async loop.
- `GPUMetrics::CompleteFastImplantDilationMetric(stream)` / `CompleteDistanceMapMetric(stream)` — bypass; read pinned scores directly in `completeFromPins` (they already do).
- `Render(Graph)` → `complete()` (graph_recipe:465) — keep for the serial helper; use `completeFromPins` on the admitted path.

---

## D. Verdict — what already has zero serialization vs what still serializes

### Overlap-ready already (zero host/device barriers between two eval launches)
- **The EvaluationContext graph path as captured and as admitted**: `EnqueueRenderPhase(EvaluationContext)`, the two metric `Enqueue(EvaluationContext)` overloads, all rendered on **per-context `cudaStreamNonBlocking`** with **pinned-dest async D2H tails**; completion via **non-blocking `cudaEventQuery` + pinned `completeFromPins`**, bounded pacing `sleep(10us)`; the greedy loop refills before poll-all. **Two contexts launched back-to-back on two streams can overlap today.**
- `EvaluationContextPool::Initialize` (per-context `cudaStreamCreateWithFlags(…, cudaStreamNonBlocking)`) — the two-stream-overlap precondition.
- Metric tails 15–24 and event-record are all on `ctx.stream` — nothing default.

### Still serializing (the walls that must stay out of the admitted loop)
- W1 `RenderPhase(BankState)` `cudaStreamSynchronize` :1397.
- W2 `CompleteRenderPhase(EvaluationContext)` :1733 `cudaStreamSynchronize`.
- W3/W4 the two `Complete…Metric(stream)` syncs.
- W5/W6/W7 the **default-stream** legacy kernels + **blocking `cudaMemcpy`** serial adapters (render legacy trunk, metric host-scalar overloads, `Compute…ResultSumWhite` at gpu_metrics:170).
- W8 the one-time `Compute…White` default-stream pump during `Prepare`.
- **Production-driver wiring** is the biggest residual risk: the handoff (`docs/handoff-2026-08-20-cuda-graph-executor-admission.md`) states the optimizer-manager `install` block still only wires the plain `launch` and the production graph path stays **default-deny `NotSubmitted`** until U5/U6. So the overlap-ready loop is exercised by the oracle test, but the *production* scheduler may still be calling the BankState serial shim (`EvaluateDirectDilationOnBank` → `RenderPrimaryCamera(bank)`, which includes the W1 sync) — that path cannot overlap until it delegates to the graph-enqueue + `completeFromPins` loop.

---

## Findings (cuda-concurrency lens)

| severity | confidence | title |
|---|---|---|
| P1 | 100 | The graph/async path is already overlap-ready by construction; the only way two evals serialize today is whichever path touches W1–W7. The fix is routing (use `cudaEventQuery` + `completeFromPins`; bypass the complete* syncs), not new kernels. |
| P1 | 100 | `RenderPhase(BankState)` ends in `cudaStreamSynchronize` (render_engine.cu:1397) and the `Enqueue/Complete` bank adapters internally call it per eval ⇒ single-host serial. |
| P1 | 100 | Metric `Complete` call sites do `cudaStreamSynchronize` (fast_implant:532, distance:179); convert-completion to pinned-pins+event, never on the hot path. |
| P2 | 100 | `CompleteRenderPhase(EvaluationContext)` (render_engine.cu:1731) still syncs; only reachable via the sync `Render(EvaluationContext)` wrapper — keep it out of the async loop. |
| P2 | 100 | Legacy default-stream host adapters (render_engine ~1048-1240, fast_implant:351/399/423/454+470, distance:80-127, gpu_metrics:152/165/170) are device-wide walls — legacy-only, fine if unused. |
| P2 | 50 | `Compute…WhiteSum` inside `createGraph` is a default-stream fence every `Prepare` — okay (one-time), but if `Prepare` is ever interleaved with in-flight evals it will fence ``. Advisory. |
| P2 | 50 | Production graph wiring appears default-deny/not-yet-wired (handoff doc) ⇒ runtime scheduler may still be on the serial bank path. Needs verification outside these files. Advisory. |

severity mapping: P0 not triggered (no correctness defect found in the overlap path). Max confidence 100 (all traced from source).

---

## Acceptance Report

<!-- criteria + evidence below -->