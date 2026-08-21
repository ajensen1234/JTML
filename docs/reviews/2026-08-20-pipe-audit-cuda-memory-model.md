# Memory-Model Review: DIRECT_DILATION Render Path — shared-context data-race audit

Lens: **atomics / fences / ordering / data visibility** (cuda-memory-model)

Repo root resolved at `/home/ajj/repo/uf/JTML` (task path `/repo/uh/JTML` did not exist; the working dir is the authoritative repo).

Files read (all non-test):
- `src/compute/render_engine.cu` (1752 lines)
- `include/compute/render_engine.cuh` (334 lines)
- `include/compute/bank_state.cuh` (296 lines)
- `src/compute/gpu_metrics.cu` (521 lines)
- `include/compute/gpu_metrics.cuh` (190 lines)
- `src/compute/evaluation_context.cpp` (398 lines)
- `include/compute/evaluation_context.h` (103 lines)
- `src/compute/CostFunctionManager.cpp` (635 lines)
- `src/compute/gpu_model.cu` (grep: RenderPrimaryCamera/Enqueue/Complete/TrySetActiveBank)

---

## A. Per-eval-write DECLARE buffers: per-context vs shared

**The render-buffer allocations are per-context. They are NOT shared mutable globals.**

`EvaluationContext` owns full `RenderBuffers primary` + `MetricBuffers metrics` per context,
allocated in `AllocateRender` / `AllocateMetrics` (`src/compute/evaluation_context.cpp:87-123` and `:126-144`).
Every device pointer the DIRECT_DILATION render chain writes resolves to a **per-ctx** allocation:

| Render write target | Allocated | Owner |
|---|---|---|
| output image | `ctx.primary.output` (eval_context.cpp:91) | per-context |
| dev_bounding_box | `ctx.primary.dev_bounding_box` (:108) | per-context |
| dev_backface | `ctx.primary.dev_backface` (:93) | per-context |
| dev_projected_triangles | `ctx.primary.dev_projected_triangles` (:98) | per-context |
| dev_projected_triangles_snapped | `ctx.primary.dev_projected_triangles_snapped` (:100) | per-context |
| dev_bounding_box_triangles | `ctx.primary.dev_bounding_box_triangles` (:102) | per-context |
| dev_bounding_box_triangles_sizes | `ctx.primary.dev_bounding_box_triangles_sizes` (:104) | per-context |
| dev_bounding_box_triangles_sizes_prefix | `ctx.primary.dev_bounding_box_triangles_sizes_prefix` (:106) | per-context |
| dev_fragment_fill | `ctx.primary.dev_fragment_fill` (:109) | per-context |
| dev_stride_prefixes | `ctx.primary.dev_stride_prefixes` (:111) | per-context |
| dev_metric_crop | `ctx.primary.dev_metric_crop` (:113) | per-context |
| dev_cub_storage | `ctx.primary.dev_cub_storage` (:122) | per-context |
| dev_pixel_score | `ctx.metrics.dev_pixel_score` (:129) | per-context |
| dev_distance_score | `ctx.metrics.dev_distance_score` (:137) | per-context |
| dev_edge_count | `ctx.metrics.dev_edge_count` (:139) | per-context |
| dev_white_count / curvature | `ctx.metrics.*` | per-context |
| persistent counters | `ctx.dev_nextCandidate/dev_nextChunk/dev_overflowFlag` (evaluation_context.cpp:211-223) | per-context |

The **legacy** `RenderEngine` dev_* members are the opposite: they are the render engine's **bank-0
allocation** (`src/compute/render_engine.cu:290-298`), and they are **rebound** (not re-allocated)
when a bank/context binds:

- `BindBankPointers` (`render_engine.cu:480-524`) overwrites the members with the `BankState.primary`
  per-context pointers.
- `RestoreBank0Pointers` (`render_engine.cu:526-550`) snaps them back to the engine-owned bank-0 set.

So when an `EvaluationContext` is executed, the engine's `dev_*_` members become **non-owning aliases**.
They never allocate per-context memory; the pool does. The engine members are a **single persistent
writer slot**, not per-eval storage.

**Bottom line for A:** Every per-eval write **DECLARE buffer is owned per-context** by the pool. There
is no device allocation a second context would need to share. `BankState`/`EvaluationContext` naming:
- `RenderBuffers` (bank_state.cuh:43-62) and `MetricBuffers` (:66-82) are the per-context names;
  the engine `RenderPointerSet` (render_engine.cuh:304-322) is the bank-0 alias cache.
- `EvaluationContext.primary`/`.secondary`/`.metrics` are the per-context `dev_*_` equivalents
  (evaluation_context.h:32-33).

---

## B. Active-bank mutable global set — single writer today

Today the evaluation path is **single-threaded** by construction:
`CostFunctionManager::TrySetActiveBank` / `SetActiveBank` (CostFunctionManager.cpp:337-344),
`GPUModel::TrySetActiveBank` (gpu_model.cu:290-307), `RenderEngine::SetActiveBank` (render_engine.cu:1259-1265),
and `GPUMetrics::TrySetActiveBank` / `RestoreBank0Metrics` (gpu_metrics.cu:254-225).

Every one of these mutates **one shared object's member pointer set**:
- `RenderEngine`: `fragment_fill_`, `dev_*_` aliases, `active_bank_`, `execution_stream_`,
  `active_output_device_`, `active_bounding_box_host_` (render_engine.cuh:171, 328-331).
- `GPUMetrics`: `pixel_score_`, `dev_pixel_score_`, ..., `active_bank_`, `execution_stream_`
  (gpu_metrics.cuh:180-183).

These are **host-side** pointers, not device buffers — but they are the contract the launchers read at
kernel-launch time.

**If two streams enqueue interleaved** on one `RenderEngine` / `GPUMetrics` instance:
1. Thread A calls `BindBankPointers(&viewA)` → all shared aliases = context A buffers.
2. Thread B calls `BindBankPointers(&viewB)` → all shared aliases = context B.
3. Thread A continues to `EnqueueRenderPhase` reading shared `model_pose_`, `model_rotation_mat_`,
   `active_output_device_`, `dev_triangles_`, and the per-r `r` locals — silently mixing A's stack copy
   of `view.primary` with B's rebound members → cross-wired kernel parameters.
4. A thread pushes `RenderBank0Pointers()` (engine members → bank-0 net) while G is still launching.

The sequence is **not atomic**; there is no mutex/fence around bind→launch→restore.

**The `active` post-`RestoreBank0Pointers` returns from `EnqueueRenderPhase` — every launch in
`EnqueueRenderPhase` (renderers) and every `cudaMemcpyAsync` D2H (metrics tail) races with any second
writer of the same members.**

---

## C. Reduction targets — do two contexts share a reduction?

No reduction target is shared.

- DIRECT_DILATION score = `white_pix_sum (host)` + `FastImplant (dev_pixel_score_, per ctx: ctx.metrics.dev_pixel_score)` + `DistanceMap (ctx.metrics.dev_distance_map_score_, ctx.metrics.dev_edge_pixels_count_)`.
- All are per-context allocations (`AllocateMetrics`, ctx.metrics), and each `Enqueue*Metric` launches on
  `stream = ctx.stream` (graph-captured per ctx; gpu_metrics.cu:360-518). There is no single shared
  `dev_pixel_score` a second context writes.
- `dev_bounding_box` lives in `ctx.primary.dev_bounding_box` — again per-context.

So a second context does **not** race a bit-exact reduction target. The reductions (`atomicAdd` inside
`FastImplantDilationMetric_*`) are intra-context only.

---

## C. Verdict: the shared-buffer the second context would corrupt

Answer: **No one device buffer corrupts.** Every mutable device dirty-touch in the DIRECT_DILATION
render/metric chain is owned per-context. The barrier by which two contexts over an engine corrupt is a
**host-side singleton alias + pose race**, not a device-buffer race.

**Killer = the single mutable `RenderEngine` instance (`GPUModel::primary_cam_render_engine_`) + the
single `GPUMetrics` instance**, specifically:
1. `RenderEngine::model_pose_` / `model_rotation_mat_` — read at launch
   (`render_engine.cu:1531-1534`) after a second writer clobbers them.
2. The rebind in `BindBankPointers` at `render_engine.cu:486-523` and its
   `Restore` at `.cu:526-550`, plus the `fragment_fill_` host pointer and active *_host set.
3. `GPUMetrics::TrySetActiveBank` (gpu_metrics.cu:254-225) mutates the same aliases for the solver
   reductions.

The `N` written in `EnqueueRenderPhase`/`EnqueueFastImplantDilationMetric` is **only single-stream
within every launch slot**. Two host streams calling these on the same engine instance are a true
host-serialization gap: **no lock, no alias scoping to the stack, and a `RestoreOnExit` that walks a
shared tail.**

If the design is N-engines/N-GPUMetrics (one per stream), the render path is race-free on device. If
one engine serves two concurrent contexts — not safe.

---

### Severity / confidence

| Finding | Severity | Confidence | Basis |
|---|---|---|---|
| Per-eval device write buffers are per-context (no shared device reduct/write surface) | n/a (correct) | 100 | evaluation_context.cpp:87-144; all kernel launch reads/writes are per-r |
| Shared mutable host alias/pose state races on two concurrent contexts | **P1** | 75 | engine members + update stores (render_engine.cu:486-550, 1391-1417; species writes via two GPUModels) |
| `model_pose_` / `model_rotation_mat_` written pre-enqueue, re-resolution at launch | P1 | 75 | gpu_model.cu:204 TabText/purch; render_engine.cu:1531-1532 |
| No fences anywhere; two use-sites unbuffered | P2 | 75 | raw pointer member layout (render_engine.cuh:298-332) |
| Metric reductions (`pixel`, `distance`) independent per context | not a defect; confirms §C | 100 | gpu_metrics.cu:375-380, 488-495 / ctx.metrics |}}

---

**residual risk (not fully verified, out of the direct render scope):**
- The legacy `Render()` (bank-0) single rendered image `renderer_output_->dev_image_` is a true shared
  writable device buffer owned by the engine (render_engine.cpp:338). It is only touched on the
  synchronous `/legacy` path, but same-engine shared use invalidates the graph claim globally — if both
  a legacy call and a context run interleaved on the same engine, that buffer is the context-writable
  one that does corrupt (file:line `render_engine.cu:338` creating the buffer; `:1051-1054` clearing).

---