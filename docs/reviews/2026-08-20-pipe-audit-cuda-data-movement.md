# P13. Per-Eval DIRECT_DILATION Data Movement Audit

REPO: /repo/uf/JTML (actual root `/home/ajj/repo/uf/JTML`)
LENS: cuda-data-movement (pinned/zero-copy, async copies, stream-ordered, UVM, full-frame D2H)
SCOPE: per-eval DIRECT_DILATION execution path + serial compatibility path + init/frame wiring.
READ-ONLY.

Standard frame: **1024×1024×1 byte = 1,048,576 B (~1 MiB)** dimensions appear in
`Study2Grid/main.cpp:105-106`, `MlBridge.cpp:41-42`. The eval render output image is
`width × height` `unsigned char` on device (`evaluation_context.cpp:91`). A full-frame
D2H would be ~**1 MiB**. All per-pose copies below are 4–16 B unless marked FULL-FRAME.

Two coexisting execution paths matter:

- **GRAPH / per-eval (primary)** — `DirectDilationMonoplaneRecipe` in
  `graph_recipe_direct_dilation.cu`: `createGraph` captures `EnqueueRenderPhase(ctx) +
  EnqueueFastImplantDilationMetric(ctx) + EnqueueDistanceMapMetric(ctx)` on `cudaStream_t
  ctx.stream` into a `cudaGraph`. Per pose: `launch()` = `cudaGraphLaunch(exec, stream)`;
  `complete()` = `cudaStreamSynchronize(stream)` + read pinned `host_overflowFlag` +
  compose from pinned host scores.
- **A-serial (BankState / compat)** — `CostFunctionManager::EnqueueDirectDilationOnBank`
  (CostFunctionManager.cpp:382-408): `RenderPrimaryCamera` → `RenderPhase(BankState)` then
  `CompleteRenderPrimaryCamera` then both metrics. Synchronous in-frame.

The EvaluationContext Manager variants in CostFunctionManager.cpp:443-470 are **stubs**
(return quiet_NaN / cudaSuccess) — they are not the executing body; the graph recipe runs
through the executor.

---

## A. Copy inventory (dump item, per-pose stream)

### init-time (once per RenderEngine/GPUModel/EvaluationContext)

| # | site | dir | size | src→dst | stream |
|---|---|---:|---|---|---|
| I0 | `evaluation_context.cpp:217-220` | — | 4 B | `cudaHostAlloc(&ctx.host_overflowFlag)` (PIN) | n/a |
| I1 | `evaluation_context.cpp:226/229/232` + `:235` | device memset | 3×4 B | `dev_nextCandidate/nextChunk/dev_overflowFlag` = 0 | ctx.stream (then `cudaStreamSynchronize`) |
| I2 | `render_engine.cu:286-287` + `:363-364` | — | 2×4 B | `cudaHostAlloc(fragment_fill_)`, `(host_overflowFlag_)` PIN | n/a |
| I3 | `render_engine.cu:420-430` | H2D | `9·T`+`3·T` floats | dev_triangles_/dev_normals_ (once), T=triangle_count | **default stream, sync** |
| I4 | `evaluation_context.cpp:128-143` (AllocateMetrics) | — | 7×4 B | per-context host pins pixel/inters/union/white/distance/edge/curv(H·4) | n/a — ONCE per context (per-Initialize) |

### per-frame (baseline / serial) — NOT per-pose

| # | site:line | dir | size | src→dst | stream | sync? |
|---:|---|---|---:|---|---|
| F1 | `gpu_metrics.cu:170` | D2H | 4 B | `white_pix_count_` ← `dev_white_pix_count_` | **default stream** | **SYNC** (`ComputeSumWhitePixels`, called at graph create + once/frame cache) |
| F2 | `render_engine.cu:1383-1388` | D2H | 16 B | `bbox_host` ← `r.dev_bounding_box` | serial bank stream | **async, then** |
| F3 | `render_engine.cu:1390-1395` | D2H | 4 B | `fragment_host` ← `r.dev_fragment_fill` | serial bank stream | async |
| F4 | `render_engine.cu:1397` | — | — | `cudaStreamSynchronize(stream)` | serial bank stream | **BLOCKS host** (15 B combined read) |

Row F0–F4 are the **serial Once-per-frame** adapter; the per-eval graph path never runs
them. F0 is a **host-stall on the default stream per frame** (per graphics create at
`graph_recipe...:397`) — small (one kernel + 4 B copy + sync loop) but on the default
sampler, and it adds up if not batching.

### per-pose (Every pose — the graph body `cudaGraphLaunch`; the ONLY host-block is the final complete-sync)

| # | file:line | op | dir | size | src→dst | stream | host-block |
|---:|---|---|---|---|---|---|---|
| P0 | `render_engine.cu:1506-1507` | `cudaMemsetAsync` | device | **W·H ≈ 1 MiB** | `ctx.primary.output` = 0 | ctx.stream | NO (async device memset, full-frame GPU-side) |
| P1 | `render_engine.cu:1618/1623/1628` | `cudaMemsetAsync`×3 | device | 3×4 B | `dev_nextCandidate/Chunk/dev_overflowFlag` = 0 | ctx.stream | NO |
| P2 | `render_engine.cu:1701-1706` | `cudaMemcpyAsync` | D2H | 4 B overflow | `host_overflowFlag` ← `dev_overflowFlag` | ctx.stream | NO (async tail pinned)
| P3 | `gpu_metrics.cu:415-420` | `cudaMemcpyAsync` | D2H | 4 B pixel_score | `ctx.metrics.host_pixel_score` ← dev | ctx.stream | NO (async tail pinned) |
| P4 | `gpu_metrics.cu:498-506` | `cudaMemcpyAsync` | D2H | 4 B distance_map_score | `host_distance_score` ← dev | ctx.stream | NO (async tail pinned) |
| P5 | `gpu_metrics.cu:509-514` | `cudaMemcpyAsync` | D2H | 4 B edge_count | `host_edge_count` ← dev | ctx.stream | NO (async tail pinned) |
| P6 | `render_engine.cu:1731` | `cudaStreamSynchronize(stream)` | — | — | — | ctx.stream | **BLOCKS HOST until FULL pose graph done** (mandatory; complete()) |
| P7 | `graph_recipe_direct_dilation.cu:466` | `cudaStreamSynchronize` | — | — | (duplicate of P6 in real complete) | ctx.stream | **BLOCKS** (graph completion) |

The 4 small D2H metas (P2, P3, P4, P5) are the ENTIRE host→read traffic on the per-pose
graph. All are async, overlapping GPU work; the only real host stall per pose is the single
`cudaStreamSynchronize` in `complete()`.

### CPU endpoints — over the eval path (safe)

| # | site | dir | size | note |
|---:|---|---|---:|---|
| E1 | `render_engine.cu:1206-1210` `WriteImage` | D2H | **W·H ≈ 1 MiB** | full-frame, malloc + flip + imwrite (default stream, sync). |
| E2 | `render_engine.cu:1234-1238` `GetcvMatImage` | D2H | **≈ 1 MiB** | full-frame, CPU mat construction. |

**They do not appear on the per-eval evaluation line** (`getGraphNode`/`Enqueue…`/`Complete…`
— the graph body only runs P0–P5). Instrumented sites: `services/implant_estimator.cpp:117`
and `Study2Grid`+`MlBridge` only (verification), so `WriteImage`/**wait**d can be treated as
**CPU endpoints, safe**. If they ever need to run during search, move to a pinned host
frame + side-stream async D2H.

---

## Which copies are concurrent-blocking: verdict

- **Per-eval, all 4-host-tail**: P1–P5 are async + overlapped => **benign**. No per-pose
  `cudaMemcpy(Host)?` blocking the host. The ONLY host-blocking op is the completion sync P6
  (mandatory — score can't be composed before `dev_*` settle; async D2H has landed into
  pinned at-the-sync point).
- **SHARED-scratch**: none across poses. Each context/pose owns its own
  `ctx.primary.output` + `metrics.*` pins; the only buffers shared across poses are the
  once-at-init `dev_triangles_`/`dev_normals_` (read-only, H2D once, I3). No cross-pose
  race.
- **Serial adapter (F1/F3/F4) + F0** block the host — but they are not on the graph path.

So the answer to "which serialize badly" is:

1. **P6 (complete-stream-sync)** is the real per-pose host serialization — the host cannot
   start the next pose until the whole render + both metric chains + 4×4B + 2-bit flag
   D2H flush complete. If the evaluator does not overlap poses across its other context
   streams, this is the per-pose floor. The executor headless impl
   (`evaluation_executor.cpp:6-12`) stubs graph launch/polling, so the actual CUDA poller
   `evaluation_executor.cu` must issue launch() for pose N+1 BEFORE calling complete() on N
   on a DIFFERENT stream. That already empties P2–P5 behind it.
2. **F0/F4** are serial-path stances; F0 runs once per frame (not per pose).

---

## C. Pinned warmup (per-context host pins) — current gates / completion decision

Per-context output pins written by the async tail:

| pin | bytes | written by | read by (complete) |
|---|---:|---|---|
| overflow | 4 B | `render_engine.cpp:1701` memcpyAsync | reads `*ctx.overflowFlag` AFTER sync (`graph_recipe.cu:466`) |
| pixel | 4 B | `gpu_metrics.cu:415` | `host_pixel_score` (`completeFromPins` `:491`) |
| dist | 4 B | `gpu_metrics.cu:498` | `host_distance_score` (`:492`) |
| edge | 4 B | `gpu_metrics.cu:509` | `host_edge_count` (`:493`) |
| (white) | 4 B | `F0` sync | ctx.comparison_image_white_sum (frame cache) |

The completion decides by:

1. **`cudaStreamSynchronize(stream)`** (`gpu.cu graph_recipe:466`, and again
   `render_engine.cu:1733`) — waits for the whole graph **including every async D2H flush
   in P1–P5**.
2. **then reads** `*host_overflowFlag` (pinned) — if non-zero, set `fragment_overflow_`,
   fail fast (`graph_recipe.cu:466` + `render_engine.cu:1742`).

So the completion does NOT do a per-op memcpy - it reads the pinned twin after the
flag-sync and reuses all 4 small scoring pins already flushed async. This is the intended
"pinned warmup" behavior: `overflowFlag` is the completion gate, the 3 metric pins are the
result. Correct. But the single sync is a full-stream sync that also orders every
also orders **every** subsequent enqueue on that stream — so a later pose enqueue to the
same ctx after `complete()` waits again.

---

## Discrepancies / wants flagged

1. **Documented `EnqueueRenderPhase`/`CompleteRenderPhase(EvaluationContext&)` mismatch**
   - code comment at `render_engine.cu:1453-1463` claims CompleteRenderPhase syncs then
     "copies host_fragment_fill / [rv]fd bbox... so downstream metric kernels have host
     values" — **but `CompleteRenderPhase(ctx)` (lines 1723-1750) does NOT do those
     copies**; it only sync + overflow-check. Host bbox/fragment are never written on the
     graph path (metrics consume `dev_bounding_box` / `dev_metric_crop` on-device). The
     `host_bounding_box` per-context pin (16 B) is therefore **dead-allocated** on the
     graph path — harmless but unnecessary page-pin noise. (Doc is misleading; no
     correctness bug.)
   - corollary: serial BankState path ALSO honors using overflow async; `RenderPhase(bank)`
     does its own `cudaMemcpyAsync` D2H at 1383-1395 and sync at 1397 (F2-F4). Both serial
     (compat) path per-frame copies that both force a host sync.

2. **Advisable batching of small metas (D-part 'batch the small metas')**:
   P1-P5 five 4 B async ops on-stream could be folded into **one `struct {int32
   overflow, pixel, dist, edge; }= 16 B` pinned per context**, flushed with **one
   `cudaMemcpyAsync` D2H in the LAST enqueue (distance-map `Enqueue…Metric`)**, pointing
   to a pinned combined cell. But **overflow must be gate-visible first**: complete uses
   overflow to FAIL the pose BEFORE composing score. Since complete() synchronizes first,
   a single 16-B struct D2H carrying pixel+dist+edge (optionally + overflow) is
   acceptable. The dedicated 4B overflow memcpy could be dropped IF compose
   stays after the sync — which it is. Estimated saving is entirely host→driver call
   count (4 async enqueue calls vs 1), not meaningful bandwidth; not a correctness change.
   - If the evaluator adds more metrics (curvature edge), it**should** fold them to the same
     struct.

3. **F0 default-stream sync** (`gpu_metrics.cu:170`, ComputeSumWhitePixels, per frame) —
   a synchronous 4B host copy on the default stream. Runs once per graph-EXEC
   creation; a modest per-frame stall but not per-pose. Advice: make it async onto the
   ctx.stream using `dev_white_count` pinned (or batch into the same 16-B struct after the
   metric drains) and cache on ctx asynchrony — returns current latency back.

## D — What it would take to make D2H cheap / overlapped

- **Batch small metas → one struct copy:** combine P3/P4/P5 (+ optional P2) into a quadratic
  pinned `MetricPins` 16-24 B struct in `evaluation_context` and issue **one** async D2H port
  in the graph tail (distance-map metric already has the last device write; put the 16-B
  copy after `EdgeKernel`/`DistanceMapMetric_Kernel_Graph` last). Both metric kernels already
  write `dev_distance_map_score_`, `dev_edge_pixels_count_`; right now they hit 3 separate
  pinned flushes. One tail = one D2transfer block on the graph; cheaper to `complete()` and
  vendor-cleaner.
- **Optionally drop the render overflow tail into the same struct** - but only if none of
  the metrics need to fail-fast before the aggregation. Today overflow is read first; sync
  is still required. If failure-abort-fast is desired, keep it a separate pin (it must
  give up pose-2 GPU work). So a conditional: batch pixel+dist+edge (3×4B); leave overflow
  its own 4B pin if pose abort matters; otherwise bite it.
- **Keep full-frame copies off the eval path** — CONFIRMED already the case. WriteImage
  (render_engine 1206/1234) is NOT reached on the frame-graph path; 1 MiB D2H only occurs
  on snapshot/cv builder calls. Only recommendation: **if an eval frame must cross to
  host per-frame (e.g., segmentation)**, add a PinnedHostFrame + side stream with
  `cudaMemcpyAsync` and `cudaEventRecord`/`EventQuery` so the eval graph doesn't stall on a
  1 MiB synchronous memcpy.
- **The one sync that stays:** P6 (`cudaStreamSynchronize`) is the true per-pose gate. To
  make D2H "cheap/overlapped", the evaluator must(1) launch pose N+1 on ANOTHER context's
  stream before calling complete() on N, and (2) never reuse the same stream for
  back-to-back graphs — the code already signals this (launch() rejects a mismatched
  stream, `graph_recipe_direct_dilation.cu:448-452`). Batching only cuts the call count,
  sync.

---

## Findings applicable to the resident of this review

Below are the data-movement lens findings to feed back. Non-findings are in the body.

| # | lens | severity | conf | file:line | what | recommendation |
|---|---|---|---|---|---|
|  D1 | data-movement | P2 | 100 | render_engine.cu:1460-1462 vs 1723-1750 | Stale cmt claims CompleteRenderPhase copies bbox/fragment; code does not. Not a correctness bug but the pinned `host_bounding_box`/`host_fragment_fill` allocs (eval path) are dead. | fix the comment. Prefer to avoid allocating host pins for bbox/fragment if it never reads them on the graph path. |
|  D2 | data-movement | P2 | 75 | gpu_metrics.cu:415/498/509 | 3 separate 4-byte async D2H flushes per pose; possible to batch into one pinned 16-B struct | merge into 1 async struct D2H in last metric enqueue; keeps the stream tail atomic. |
|  D3 | data-movement | P2 | 100 | gpu_metrics.cu:170 | computeSumWhitePixels sync D2H on DEFAULT stream each frame. | make it async on ctx stream or fold into the frame-cached white struct; remove default-stream stall. |
|  D4 | data-movement | P3 | 100 | render_engine.cu:1397 | serial block is F4 (cudaStreamSync after bbox/fragment Copy) per-frame in serial-compat path. | no fix needed on graph path; just a note that serial adapter keeps it. |
|  D5 | data-movement | P2 | 75 | complete() stream sync `graph_recipe_direct_dilation.cu:466` | single sync per pose flushes full graph; the D2H "cheap" only with 2-context overlap. | Ensure the executor starts N+1 on another ctx before complete (as the launch() guard enforces). |
|  D6 | data-movement | P2 | 75 | render_engine.cu:1506 memsetAsync W·H per pose | full-frame device write is the largest GPU op; it's async so not a host stall, but it's the top of the per-pose D-width if using a small model. | benign — do NOT convert to sync; consider `cudaMemsetAsync` combined with the metric resets. |

**Residual risk:**

- The only per-pose host-block is the complete sync; if the Executor does not overlap contexts, per-eval latency = GPU full chain + 1 sync (no D2D overlap savings). This is not data-movement bug but a pipeline shaping issue.
- No full-frame D2H on eval today (good); if a snapshot crosses, it must go on a side stream.

---

### Accepted criteria trace

criterion-1: Findings with concrete file:line + severity, present (D1–D6) and covered by C in the body.