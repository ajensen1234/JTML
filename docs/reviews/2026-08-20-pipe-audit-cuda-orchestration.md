# P014 — Pipe Orchestration Audit: DIRECT → Cost → Render → Metric Pipeline

**Lens:** cuda-orchestration (greedy feeder, admission policy, launch-to-launch gap, host starvation)
**Mode:** READ-ONLY. No code was written or edited.
**Repo:** `/home/ajj/repo/uf/JTML` (jj workspace — read-only audit, no changes)
**Reference ground truth:** cuda-skill snapshots (CUDA 13.3 docs, manifest checked 2026-07-22):
`/home/ajj/.pi/agent/skills/cuda-skill/references/` → `performance-traps.md`, `best-practices-guide/11.5-concurrent-kernel-execution.md`.

---

## 0. TL;DR

The production serialization the task flagged is **verified and is the root barrier**, with two corrections to the exact attribution:

- **True:** `CostFunctionManager.cpp:389-390` (`EnqueueDirectDilationOnBank`) calls `EnqueueRenderPrimaryCamera(bank)` then **immediately** `CompleteRenderPrimaryCamera(bank)`, so the "enqueue" is host-blocking.
- **Correction:** the `cudaStreamSynchronize` is **inside `RenderEngine::RenderPhase(BankState&)` at `render_engine.cu:1397`**, reached via `GPUModel::EnqueueRenderPrimaryCamera` (`gpu_model.cu:310-317`). `CompleteRenderPrimaryCamera` (`gpu_model.cu:319-326`) itself is asynchronous (it reads the already-synced host pin `fragment_fill` at `render_engine.cu:1409` and launches StridePrefix+FillTriangle). There is **no symbol "TryRenderFrame"** in the current tree.
- **True:** `TrySetActiveBank` is object-global mutable state on `GPUModel`/`GPUMetrics` (`gpu_model.cu:290-308`, `gpu_metrics.cu:254-265`), bound/restored per eval (CostFunctionManager.cpp:386/391/405/414/423).
- **Expansion (new findings):** (a) each metric completer calls `cudaStreamSynchronize` (`fast_implant_dilation_metric.cu:532`, `distance_map_metric.cu:179`) — 3 host-blocking syncs per eval (render head + 2 metric completes); (b) the greedy feeder's `cudaEventSynchronize` (`cost_capacity_service.cu:383-384`) is a 4th blocking point per finished eval; (c) the serial fallback adapter uses the **default stream + blocking cudaMemcpy** (`render_engine.cu:1046-1125`, `fast_implant_dilation_metric.cu:470`, `distance_map_metric.cu:118-127`); (d) the only zero-sync design (EvaluationContext/CUDA-graph path, R13) is **unreachable in production** — admission is default-deny (`optimizer_manager.cpp:1354-1426`).

---

## 1. Full call pipeline (production, monoplane DIRECT_DILATION, bank pool admitted)

Host thread = Qt `optimizer_thread` (single). `SYNC` = host-blocking call; `enq` = enqueue, non-blocking.

```
OptimizerManager::Optimize()                                  optimizer_manager.cpp:866-1256
  └─ per-frame image loop; per-stage spec loop                :1019-1192 (Trunk/Branch/Leaf)
     └─ RunDirectStage(range, stage_manager)                  :1302-1502
        ├─ serial_cost = BuildGpuCostAdapter(...)             :1314-1315, body :1713-1752
        │    └─ pose set + stage_manager.callActiveCostFunction()
        ├─ DirectOptimizer opt(serial_cost, ...)              :1316-1317
        ├─ if pool>1 && monoplane && DIRECT_DILATION:         :1323-1352
        │    opt.SetBatchCost( [poses] →
        │        RunCostBatchGreedy(poses, serial_cost,
        │                            enqueue, complete))       :1349, feeder cost_capacity_service.cu:359-422
        │      enqueue  = SetCurrentPrimaryCameraPose(pose)
        │              + stage_manager.EnqueueDirectDilationOnBank(bank)   :1338-1339
        │      complete = stage_manager.CompleteDirectDilationOnBank(bank) :1345-1348
        └─ opt.Run()                                          direct_optimizer.cpp:93-131

DirectOptimizer::Run                                           direct_optimizer.cpp:93
  ├─ seed eval via EvaluateCostFunction(UnitCenter())          :102  (serial adapter → full serial path)
  └─ while (calls < budget): ConvexHull(); TrisectPotentiallyOptimal()  :115-128

TrisectPotentiallyOptimal (batch seam, plan 010 U11)           direct_optimizer.cpp:238-330
  ├─ per POH box: oc (no eval) + A(+shift) + B(-shift) centers :245-277
  ├─ results = batch_cost_(batch_centers)                      :280  ★ ITERATION BARRIER (by-value)
  └─ replay results in serial storage order                    :293-310 (optimum/callback order saved)

RunCostBatchGreedy (4-arg, enqueue+complete)                   cost_capacity_service.cu:359-422
  pool = 2 extra banks (1..2; bank 0 = compatibility)          :200, :206-224
  loop over poses:
    ├─ CheckoutBank()                                          :390-396  (on exhaustion: finish(front))
    ├─ enqueue(pose, bank)                                     :402  ★ blocks inside (S3)
    ├─ cudaEventRecord(completion_event, stream)               :408-409
    └─ leases.push(lease)
  drain: finish(lease) for all                                 :418-420
    └─ finish(): cudaEventSynchronize(event)  ★S4              :383-385
         → complete(*state)  ★S6 (2 stream syncs)              :386
         → RecycleBank                                         :387

enqueue → EnqueueDirectDilationOnBank                          CostFunctionManager.cpp:382-409
  ├─ TrySetActiveBank(&bank)                                   :386  (binds GPUModel + GPUMetrics)
  ├─ EnqueueRenderPrimaryCamera(bank)                          :389, gpu_model.cu:310-317
  │    ├─ TrySetActiveBank(&bank) (render engine aliases)      :311, render_engine.cu:480-524
  │    ├─ SetPose(current_pose_A_)                             :314
  │    └─ RenderPhase(BankState&)                              :315, render_engine.cu:1293-1398
  │         └─ memsetAsync / Reset / WorldToPixel / BBox / sizes / cub scan
  │            / LaunchPacket / 2× D2H async
  │            → cudaStreamSynchronize(stream)                 ★S3 ★★ :1397
  ├─ CompleteRenderPrimaryCamera(bank) IMMEDIATELY (★S2)       :390, gpu_model.cu:319-326
  │    └─ CompleteRenderPhase(BankState&)                      render_engine.cu:1400-1449
  │         ├─ host read r.host_fragment_fill  (drives S3)     :1409
  │         ├─ StridePrefixKernel<<<>>>                        :1424
  │         └─ FillTriangleKernel<<<>>>                        :1438 (enq only, no sync)
  │         # on success the bank STAYS bound (no unbind here)
  ├─ EnqueueFastImplantDilationMetric(rendered, cf, dil, stream)  fast_implant_dilation_metric.cu:477-528
  │         ├─ guard: active_bank_ must be set                 :482
  │         ├─ HOST READ host_bounding_box (grid math)         :485
  │         ├─ Reset/Edge/Dilate/Diff kernels                  :493-522
  │         └─ cudaMemcpyAsync(pixel_score D2H)                :523
  └─ EnqueueDistanceMapMetric(rendered, dm, dil, stream)       distance_map_metric.cu:137-174
       ├─ requires active_bank_                                :140
       ├─ HOST READ host_bounding_box                          :145-147
       ├─ reset×2 + DistanceMapKernel                          :151-166
       └─ 2× cudaMemcpyAsync D2H                               :167-172

complete → CompleteDirectDilationOnBank                        CostFunctionManager.cpp:411-425
  ├─ TrySetActiveBank(&bank)  (re-bind!)                       :414
  ├─ CompleteFastImplantDilationMetric(stream)                 :418
  │    └─ cudaStreamSynchronize(stream)  ★S6 :532 (drains the ENTIRE stream)
  │    └─ host read pixel_score_[0]                            :533
  ├─ CompleteDistanceMapMetric(stream)                         :419
  │    └─ cudaStreamSynchronize(stream)  ★S6 :179 (redundant 2nd drain)
  │    └─ host read distance_map_score_[0] / edge_pixels_count_[0] :180
  ├─ TrySetActiveBank(nullptr)  (unbind + restore)             :423
  └─ score = white_sum + fidm + distance

Fallback (pool ≤1 / biplane / batch ≤1): feeder serializes    cost_capacity_service.cu:310-312
  └─ serial_cost(pose) → BuildGpuCostAdapter                   optimizer_manager.cu:1713-1752
       └─ callActiveCostFunction() → costFunctionDIRECT_DILATION  DIRECT_DILATION.cpp:58-91
            ├─ RenderPrimaryCamera(Pose)  → RenderEngine::Render()  render_engine.cu:1046-1195
            │      ★ ALL KERNELS ON DEFAULT STREAM ★ + 2× blocking cudaMemcpy D2H :1116-1121
            ├─ FastImplantDilationMetric(3-arg, default stream)  fast_implant_dilation_metric.cu:341-473
            │        blocking cudaMemcpy D2H  :470-471
            └─ DistanceMapMetric(3-arg, default stream)          distance_map_metric.cu:66-135
                    2× blocking cudaMemcpy D2H  :118-127
```

**Graph path (exists, NOT reachable in production):**
`optimizer_manager.cpp:1353-1453` — admission is default-deny. `executorReady` requires `evaluation_executor_->poolSize() > 1` (`:1425-1426`) and the pool is lazily sized only **after** `decision.install` (`:1440-1441`); the comment at `:1354-1358` states: *"which no production path can reach yet"*. If it DID admit: `SetBatchCost` → `MaterializeOrderedScores(exec->RunBatch(poses, serial_cost))` (`:1448-1450`) → `EvaluationExecutor::RunBatchWithCost` (`evaluation_executor.cpp:211-240`) with enqueue/poll/pace/complete hooks (`evaluation_executor.cu:14-70`) → per context: `recipe->updateParams` (only the WorldToPixel kernel node is repacked — `graph_recipe_direct_dilation.cu:419-441`), `recipe->launch` = one `cudaGraphLaunch` (`:443-456`); the captured chain is `EnqueueRenderPhase(EvaluationContext&)` (**no sync, no D2H — device-side overflow check + persistent workers**, `render_engine.cu:1475-1717`) + metric enqueues with device-derived crop via `ComputeMetricCropKernel` (no host bbox read, `gpu_metrics.cu:319-425`); polling = **non-blocking** `cudaEventQuery` (`evaluation_executor.cu:40-43`) + 10 µs pacing (`:53-56`); completion = `completeFromPins` — host reads of pinned scores only, no sync (`graph_recipe_direct_dilation.cu:473-499`) — **zero sync on the admitted path (R13)**.

---

## 2. EVERY serialization / host-block point (file:line + why it blocks concurrency)

| # | Site (file:line) | What it does | Why it blocks N-way overlap |
|---|---|---|---|
| **S1** | `direct_optimizer.cpp:280` `results = batch_cost_(batch_centers)` | Whole-iteration batch call, returned **by value**; DIRECT cannot start ConvexHull(k+1) until all results are back | Iteration k+1 strictly depends on k (convex hull). Also caps pipelining at batch size (2·#POH — typically 2-6 centers). |
| **S2** | `CostFunctionManager.cpp:389-390` | enqueue+complete of the render chain fused into the enqueue step | Destroys the enqueue/deferred-complete seam; the "enqueue" is host-synchronous by construction. |
| **S3** | `render_engine.cu:1397` (inside `RenderPhase(BankState&)`) | `cudaStreamSynchronize(stream)` — waits for the whole eval's geometry chain (memset→WorldToPixel→bbox→sizes→CUB scan→packet→2×D2H) | THE root barrier: the next eval's enqueue cannot start until this eval's head is done. Only the post-sync tail (FillTriangle + metrics) can overlap the next head — at best a depth-1 sliding window across banks, not N-way eval overlap. The sync exists because tail launches (FillTriangle grid, metric grids) need host-side `fragment_fill` / `host_bounding_box` values (S5) — removing S3 requires moving that dependency on-device. |
| **S4** | `cost_capacity_service.cu:383-384` | `cudaEventSynchronize(event)` in the feeder's `finish()` | Blocks the host until the eval's metrics event fires. The intended completion point, but it is reached before `complete()` and made redundant by S6's internal syncs. |
| **S5** | `render_engine.cu:1409` (`fragment_fill`); `fast_implant_dilation_metric.cu:485` and `distance_map_metric.cu:145-147` (`host_bounding_box`) | Host-side scalar reads used for **host-side grid sizing** (FillTriangle grid, metric crops) | Each read requires its D2H copy to have completed → forces a stream sync before the dependent kernels can be launched. This is *why* S3 exists; it also makes the path graph-uncapturable and injects a host round-trip inside the eval. |
| **S6** | `fast_implant_dilation_metric.cu:532` + `distance_map_metric.cu:179` | two `cudaStreamSynchronize` per finished eval, both on the same stream | Each drains the full eval chain (render tail + metrics + D2H) before the host reads score pins. The second is a redundant drain — 2 host syncs where 1 event poll would do. |
| **S7** | `CostFunctionManager.cpp:386/414` → `gpu_model.cu:290-308` / `gpu_metrics.cu:254-265` → `render_engine.cu:480-524` / `gpu_metrics.cu:227-252` | Global mutable active-bank rebind (~27 pointer aliases) twice per eval + restore | Object-global "active bank" means only one bank can be bound at a time; enqueue(i+1) rebinds while eval i's tail is in flight, then `complete(i)` rebinds back. The host flip-flops the shared aliases inside the eval window; correctness is preserved only by launch-time argument capture. Serial critical section; blocks true multi-context roll-out. |
| **S8** | default-stream serial path: `render_engine.cu:1046-1195` (all kernels on stream 0), blocking `cudaMemcpy` `:1116-1121`; `fast_implant_dilation_metric.cu:470-471`; `distance_map_metric.cu:118-127` | Serial adapter for every fallback eval | Default stream 0 is the legacy stream: per BPG 11.5 concurrent-kernel-execution, kernels on the default stream begin only after **all** preceding work on **any** stream and block later work. Any default-stream eval serializes the GPU. Used by the fallback path, the seed eval (`direct_optimizer.cpp:102`), and sym-trap. |
| **S9** | `optimizer_manager.cpp:1316-1317` (single `DirectOptimizer`, one Qt thread) + callbacks (`:1462-1486`: 30 fps UI + cooperative stop) | All host structure on one thread | No host-side parallelism; interop/UI emitted on the same thread between batches. |
| **S10** | `optimizer_manager.cpp:1356-1458` (graph admission default-deny; `executorReady` gate `:1425-1426`; pool lazily sized only after `install`) | The only zero-sync design (R13) is **dead code in production** | Production stays on the S3-bound bank path; the N-way executor cannot be exercised by any current run. |

**Bonus observation (S7 territory):** `CompleteRenderPrimaryCamera` on success **leaves the bank bound** — unbind happens only on error in `EnqueueDirectDilationOnBank` (`:391-393, :405-406`) or at `CompleteDirectDilationOnBank:423`. Between enqueue(i) success and complete(i), enqueue(i+1) rebinds — and `complete(i)` re-binds back. Works but fragile; listed as a hazard, not a finding.

---

## 3. Read-only layer-by-layer recommendations for N-way eval overlap

**direct_optimizer** (`src/domain/direct_optimizer.cpp`)
- R1. Keep the batch seam (`:238-330`): its whole-iteration return is **inherent** (ConvexHull(k+1) needs batch_k's results). Do not pretend iteration-level pipelining is available without changing DIRECT's selection order.
- R2. Batch width: currently 2 centers/POH box (min batch = 2); the feeder can only overlap `min(|batch|, poolSize)` evals. No code change needed — flag that small early batches bound any speedup.
- R3. Move the seed eval (`:102`) off the full serial `Render()` path: give it one pooled bank context so even the seed is a stream-enqueue (avoids S8 for the first eval).

**optimizer_manager / coordinator**
- R1. **Change the batch wiring at `optimizer_manager.cpp:1323-1352`**: install an **enqueue-only** bank callback (render head async + metric enqueues async + event record + return) and a **sync-once** complete. The feeder API explicitly anticipates this: `cost_capacity_service.cu:344-346` — *"The current supported callback performs its terminal copies before returning; future enqueue-only callbacks may replace this with event polling without changing the input-indexed contract."*
- R2. **Admit the graph executor** (option-gated, monoplane DIRECT_DILATION): the EvaluationContext/graph path (`evaluation_executor.cu:14-70` + `graph_recipe_direct_dilation.cu:329-456`) is the zero-sync design, currently unreachable (`:1354-1358`). The manager must size the executor pool when the bank pool is admitted so `DecideGraphAdmission` installs.
- R3. Minimum-delta alternative: keep the bank path but drive it from the graph executor's `RunBatch` when admission succeeds; keep the bank path behind a flag.

**cost_function (`CostFunctionManager`)**
- R1. Split `EnqueueDirectDilationOnBank` into true enqueue-only (render head + metrics on bank stream, no sync); make `CompleteDirectDilationOnBank` the **only** syncing point, using a **single** event wait (drop the second metric sync `:418-419`).
- R2. Do not call `CompleteRenderPrimaryCamera` from the enqueue path (`:389-390`); reorder per eval as head(geometry, no sync) → FillTriangle tail → metric enqueues → record one event → completer.
- R3. Fix or delete the `...OnEvaluationContext` overloads — they currently return NaN without doing work (`CostFunctionManager.cpp:443-470`).
- R4. Binding: one bind per eval window, not two (`:386` enqueue-bind + `:414` complete-bind + `:423` unbind).

**render_engine**
- R1. Provide an async BankState render enqueue (the EvaluationContext variant `render_engine.cu:1475-1717` already proves the pattern): move the host packet barrier on-device — replace the host `fragment_fill` read (`:1409`) and overflow check with the **device-side OverflowCheck + persistent StridePrefix/FillTriangle workers** (allocate the 3 per-bank device counters + host twin like `evaluation_context.h:40-44`). Then the S3 `cudaStreamSynchronize` can be deleted.
- R2. `Render()` (`:1046-1195`, default stream) is needed only for the serial fallback/seed; keep for compatibility, route new paths through the pooled context.

**gpu_metrics**
- R1. Bank-path `Enqueue` (fast_implant `:477-528`, distance_map `:137-174`): remove the `host_bounding_box` read by switching to the **already-implemented device-side crop** (`ComputeMetricCropKernel` + `MetricCropParams`, `gpu_metrics.cu:319-425`) — enqueue then has no host-register dependency.
- R2. `Complete*` (`fast_implant_dilation_metric.cu:530-534`, `distance_map_metric.cu:177-180`): replace the full stream syncs with a poll of the bank's completion event (`cudaEventQuery`), or one sync at most.

**Layering audit outcome:** each of the five layers has one structural change: (1) direct_optimizer — pooled seed; (2) manager/feeder — enqueue-only callback + admissible graph pool; (3) cost_function — no fused complete + one-event completion; (4) render_engine — async head with device-side overflow + persistent fill; (5) gpu_metrics — device-cropped enqueues + event-based completion. The greedy feeder already targets the enqueue-only contract.

---

## Findings (lens JSON)

```json
{
  "findings": [
    {
      "lens": "cuda-orchestration",
      "title": "The 'enqueue' is host-blocking: RenderPhase(BankState) ends in cudaStreamSynchronize",
      "severity": "P1",
      "confidence": 100,
      "evidence": [
        "render_engine.cu:1397 — 'return cudaStreamSynchronize(stream);' terminates RenderPhase(BankState&), before StridePrefix/FillTriangle are enqueued",
        "gpu_model.cu:315 — EnqueueRenderPrimaryCamera calls primary_cam_render_engine_->RenderPhase(bank)",
        "CostFunctionManager.cpp:389-390 — 'EnqueueRenderPrimaryCamera(bank)' then 'CompleteRenderPrimaryCamera(bank)' unconditionally"
      ],
      "owner": "review-fixer",
      "suggestedFix": "Add an async BankState render enqueue mirroring EnqueueRenderPhase(EvaluationContext&) at render_engine.cu:1475 (no-sync, device-overflow, persistent workers) and defer completion to CompleteDirectDilationOnBank; remove the cudaStreamSynchronize at 1397 and the immediate Complete at CostFunctionManager.cpp:390."
    },
    {
      "lens": "cuda-orchestration",
      "title": "Per-eval completion drains the stream twice + a feeder event sync = 3-4 host blocks per evaluation",
      "severity": "P1",
      "confidence": 100,
      "evidence": [
        "fast_implant_dilation_metric.cu:530-534 — CompleteFastImplantDilationMetric → cudaStreamSynchronize(stream) then read pixel_score_[0]",
        "distance_map_metric.cu:177-180 — CompleteDistanceMapMetric → cudaStreamSynchronize(stream) then read distance_map_score_[0]",
        "CostFunctionManager.cpp:418-419 — CompleteDirectDilationOnBank calls both back-to-back",
        "cost_capacity_service.cu:383-384 — cudaEventSynchronize(event) in feeder finish() precedes complete()"
      ],
      "owner": "review-fixer",
      "suggestedFix": "Single completion point per eval: one event poll (cudaEventQuery) followed by pinned-score reads; delete the redundant second stream sync."
    },
    {
      "lens": "cuda-orchestration",
      "title": "Active-bank binding is object-global mutable state across the eval window",
      "severity": "P2",
      "confidence": 75,
      "evidence": [
        "gpu_model.cu:290-308 GPUModel::TrySetActiveBank → RenderEngine::SetActiveBank → BindBankPointers rebinds ~15 member aliases",
        "gpu_metrics.cu:254-265 → BindMetricBank rebinds ~12 aliases",
        "CostFunctionManager.cpp:386 binds, :423 unbinds; enqueue(i+1) can rebind between enqueue(i) and complete(i)'s re-bind (:414)"
      ],
      "owner": "maintainer",
      "suggestedFix": "Route bank evals through per-eval EvaluationContext state (already exists) so no shared mutable alias set exists; at minimum: one bind per eval, not two."
    },
    {
      "lens": "cuda-orchestration",
      "title": "Host-side scalar reads (fragment_fill, host_bounding_box) in the hot loop pin the S3 sync",
      "severity": "P2",
      "confidence": 100,
      "evidence": [
        "render_engine.cu:1409 — 'const int fragment_fill = *static_cast<int*>(r.host_fragment_fill);' (drives FillTriangle grid :1416-1423)",
        "fast_implant_dilation_metric.cu:485 — metric crop grid read of host_bounding_box",
        "distance_map_metric.cu:145-147 — same host read in the DistanceMap enqueue"
      ],
      "owner": "maintainer",
      "suggestedFix": "Move fragment/crop grid derivation to device (metric kernels already do this for the EvaluationContext path) and switch FillTriangle to the persistent-worker scheme, removing both host reads and the S3 barrier."
    },
    {
      "lens": "cuda-orchestration",
      "title": "The zero-sync graph path (R13) is unreachable: admission is default-deny in production",
      "severity": "P2",
      "confidence": 75,
      "evidence": [
        "optimizer_manager.cpp:1356-1358 — 'which no production path can reach yet' (executor admission comment)",
        "optimizer_manager.cpp:1425-1426 — executorReady requires poolSize() > 1, but the pool is only allocated after decision.install"
      ],
      "owner": "review-fixer",
      "suggestedFix": "Admit (option-gated) the EvaluationContext pool for monoplane DIRECT_DILATION and route batch_cost_ to MaterializeOrderedScores(executor->RunBatch(...)) — the no-sync path."
    },
    {
      "lens": "cuda-orchestration",
      "title": "Serial fallback/seed runs on the legacy default stream with blocking D2H copies",
      "severity": "P2",
      "confidence": 75,
      "evidence": [
        "render_engine.cu:1046-1195 Render() — no stream argument (default stream 0) + cudaMemcpy D2H at 1116 and 1121",
        "fast_implant_dilation_metric.cu:470-471 — blocking cudaMemcpy in the 3-arg metric",
        "distance_map_metric.cu:118-127 — blocking cudaMemcpy in the 3-arg metric"
      ],
      "owner": "maintainer",
      "suggestedFix": "Keep the legacy adapter for compatibility only; move the seed and the sym-trap eval to a pooled bank context (one stream enqueue/complete)."
    }
  ]
}
```

**Corrections to the claim attribution (verification result):** there is no `TryRenderFrame`; the sync lives in `RenderPhase` inside the *enqueue* step. The task's functional conclusion (per-hidden sync at the top of the production render chain) holds.

---

## Acceptance Report