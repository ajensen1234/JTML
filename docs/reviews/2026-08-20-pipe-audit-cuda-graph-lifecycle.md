# P13 — Graph execution path vs graph-free multi-stream replay (DIRECT_DILATION monoplane)

Lens: `cuda-graph-lifecycle` (capture/instantiate/launch, exec-node updates, capture restrictions).
Scope: `src/compute/graph_recipe_direct_dilation.cu`, `test/oracle/graph_recipe_direct_dilation_test.cu`,
`src/compute/render_engine.cu`, `src/compute/gpu_metrics.cu`, `include/compute/graph_recipe.h`,
`include/compute/evaluation_context.h`, `include/compute/graph_preflight.cu`.
Grounding (local): `cuda-skill` references/cuda-guide/04-special-topics/cuda-graphs.md (hosted snapshots — see ¶reference-note).

---

## Key lifecycle facts established from the source

1. **Graph build = capture of the exact same three enqueues the replay runs.**
   `DirectDilationMonoplaneRecipe::createGraph` (graph_recipe_direct_dilation.cu:329–417) opens
   `cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal)` (line 351), then calls
   `inputs.render->EnqueueRenderPhase(*ctx)` (354), `EnqueueFastImplantDilationMetric` (356), and
   `EnqueueDistanceMapMetric` (363); then `cudaStreamEndCapture` → `cudaGraphInstantiate` (379).
   The graph-free replay path (`RunSerial` in the oracle, and the production `Enqueue→Complete` chain)
   calls the *identical three enqueues* stream-serialized. So the captured node set and the replay’s
   kernel sequence are the same kernel objects in the same stream order.

2. **The only pose-parameterized node is `WorldToPixel`.**
   `CaptureWorldNode` (155–202) walks the graph once at capture to find the kernel node whose
   `params.func == WorldToPixelKernel`, snapshots its 18 args, and returns. Per pose,
   `updateParams` (419–441) only ever issues `cudaGraphExecKernelNodeSetParams` on that one node,
   after `RebuildWorldArgs` recomputes x/y/z/rotation and re-points projected/snapped/backface.
   No other node receives a per-pose update.

3. **complete() vs completeFromPins().** complete() (458–471) = `cudaStreamSynchronize(stream)`
   then checks `host_overflowFlag`, then `completeFromPins(ctx)` (473–499). completeFromPins reads
   only `ctx.metrics.host_pixel_score / host_distance_score / host_edge_count` (pinned ints) plus
   the cached `ctx.comparison_image_white_sum`, and composes via the *shared*
   `ComposeDirectDilationScore` helper (graph_recipe.h:138–144). The D2H score pins are written by
   the async `cudaMemcpyAsync` tails captured inside the metric enqueues (gpu_metrics.cu:415, 498+).

4. **white-sum baseline is computed once at capture, not per pose.**
   createGraph computes `inputs.metrics->ComputeSumWhitePixels(inputs.comparison_frame->GetGPUImage())`
   once (graph.cu:397) and writes `ctx.comparison_image_white_sum`. Both the graph path and the serial
   path consume the cached value; the oracle asserts `comparison_image_white_sum > 0` (test:290, 414).

---

## A. Exact logical equivalence — graph vs replay produce identical Layer A bytes / Layer B ints

Yes, under three stated preconditions, and the oracle verifies it.

- **Layer A (rendered projection bytes):** both paths render into the same device buffer,
  `ctx.primary.output` (render_engine.cu EnqueueRenderPhase binds `active_output_device_` to
  `r.output`; gpu_metrics:339 fallback also resolves to `ctx.primary.output`). The kernels involved
  (WorldToPixel → BoundingBox → BoundingBoxSizes → CUB prefix → PrepareLaunch → persistent
  Fill/Stride workers) are the same functions in the same stream order. The oracle asserts
  `graph.image == serial.image` (test:306, ::346, ::357).
- **Layer B ints (pixel/distance/edge):** the metric kernels (`FastImplantDilationMetric_*_Graph`,
  `DistanceMapMetric_Kernel_Graph`) write the same device score scalars and the same pinned host
  cells the replay writes. Oracle asserts `graph.pixel_score == serial.pixel_score` etc.
  (test:307–309).
- **Layer C (score):** both grant through the *single shared* `ComposeDirectDilationScore`
  (graph_recipe.h:138) from the same three pins and same cached white_sum. Layer-C equality is
  asserted to abs 1e-12 / rel 1e-9 (test:244–248, 310).

**Preconditions (implicit but load-bearing):**
1. The single-stream capture is fully rejoined to the origin stream within `createGraph` before
   `cudaStreamEndCapture`. All enqueues (including the async D2H memcpy of score/overflow) are on
   `ctx.stream`, so the graph rejoins — matches the reference’s single-origin-stream capture shape.
2. Per-pose the graph launch is on the *same* stream that the graph was captured against
   (`launch()` actively rejects a mismatched stream — graph.cu:452) and `updateParams` rebinds the
   World node to the *current* ctx before the score pins are read.
3. `compare_white_sum` is constant within a frame (it is, by design: pose-independent, cached).

### What completeFromPins reads/promises
- Reads exactly four host values: `host_pixel_score`, `host_distance_score`, `host_edge_count`
  (all pinned, D2H-landed), and the process-local `comparison_image_white_sum`.
- It does **not** sync; it promises correct values **only if the caller has already finished**
  the captured/latent pipeline via a stream sync (complete() path) or an event query
  (graph_recipe.h:128–132 docstring: "Caller guarantees D2H landed via event query"). This
  contract is the same for graph path and replay.

**Result:** For a fixed input frame and fixed dilation, graph and replay are
pipeline-logically equivalent. The oracle (Kneel_1, two poses, two handles) verifies it at Layer A
(ints) and Layer C (score) tolerances.

---

## B. Per-context graph-exec handle lifecycle

- **Instantiated once per context:** `createGraph` builds and `cudaGraphInstantiate`-s **one**
  `cudaGraphExec_t` per `EvaluationContext` (graph_recipe_direct_dilation.cu:379). Each context gets
  its own private exec referencing its own `ctx.primary` buffers and its own stream. The oracle
  asserts `first_graph != second_graph` for two checked-out contexts (test:413) and drives two
  independent execs concurrently.
- **Reused per pose, not rebuilt:** for every subsequent pose, the caller calls
  `updateParams(graph, ctx)` → `cudaGraphExecKernelNodeSetParams` (World node only) → `launch` →
  `cudaGraphLaunch`. No re-capture, no `cudaGraphExecUpdate`, no `cudaGraphInstantiate` per pose.
- **When it re-creates:**
  - On every fresh `createGraph` call (new context / context re-checked-out), which is the pool
    lease boundary. `CaptureGeneration` (graph_recipe.h:57) exists to signal an input-identity change
    (frame rewrite epoch, dilation change); a live exec is not auto-recreated on these — the
    lifecycle is **external**: the executor must `destroyGraph` + recreate when input identity /
    dilation / frame changes.
  - `destroyGraph` (501–507) destroys the exec and graph and frees the wrapper. In the test the
    callers destroy before `pool.Recycle` (test:359→361, 438→440); the pool itself does not own or
    release graphs.

**Guardrails on reuse:** `updateParams` rejects unless `wrapper->captured_context == &ctx` and ctx
is in-flight/initialized (graph.cu:426–427); `launch` rejects if `ctx.stream` ≠ captured stream
(452). These prevent a graph bound to context A from being driven with context B.

---

## C. Is the graph-free replay behavior-identical termination, and what host work remains?

**Termination: yes, semantically identical** — both paths finish when the same pinned score cells are
D2H-valid and `host_overflowFlag == 0`, and both read the same three pins to compose the score. The
single substantive timing difference is **where the stream barrier sits**:

- **Graph path:** capture runs render + both metrics as one continuous stream; per pose the only
  barriers are `complete()`'s single `cudaStreamSynchronize` (line 465) plus the read. There is no
  intermediate render/metrics host sync inside the launch path.
- **Replay path (as written in the oracle `RunSerial`):** instruments an *extra* host ground
  `CompleteRenderPhase(ctx)` (line 272, syncs stream + D2H overflow read) that sits **between**
  EnqueueRenderPhase and the two metric enqueues (271/275/281). Because a single stream is
  serialized this extra ground does **not change Layer A/B bytes** (verified equal by the oracle),
  but it means the naive replay is **not launch-shape-identical** to the graph: it re-issues ~15
  kernel launches per pose *and* inserts a mid-pipeline host barrier the graph does not carry.

### Remaining host platform work for a graph-free replay per pose
1. Apply the pose: `SetPose(ctx, …)` / `SetEnginePose(engine, …)` (host rotation recompute).
2. `EnqueueRenderPhase(ctx)` — 12 kernel launches + memsets + persistent-worker zeroing, on
   `ctx.stream` (render_engine.cu:1475–1717).
3. `EnqueueFastImplantDilationMetric` (gpu_metrics:319–425) and `EnqueueDistanceMapMetric`
   (427–521) — 3 + 2 launches + async D2H score tails.
4. One completion gate: either `cudaStreamSynchronize(ctx.stream)` (like `complete()`) or an
   event-record query (if you want to avoid the blocking sync) before reading the three pinned
   ints.
5. Read `host_pixel_score / host_distance_score / host_edge_count` (+ the frame-level white_sum)
   and call the shared `ComposeDirectDilationScore`.

The saved-vs-cost tradeoff is the crux: the graph path pays a one-time capture+instantiate at
`createGraph` and trades each pose’s ~15 re-issued kernel launches for **one `cudaGraphLaunch`**
plus one `cudaGraphExecKernelNodeSetParams`, while the replay re-issues every kernel each pose with
no node-handle survival. Both consume the same pinned score pins, so termination values agree.

---

## Findings (this lens)

```json
{
  "findings": [
    {
      "lens": "cuda-graph-lifecycle",
      "title": "updateParams patches only the WorldToPixel node; all metric-node launch args (dilation, grid, intra-frame scalars) are frozen at capture",
      "severity": "P1",
      "confidence": 75,
      "evidence": [
        "graph_recipe_direct_dilation.cu:204-205 RebuildWorldArgs sets only projected/snapped/backface/x/y/z/rotation",
        "graph_recipe_direct_dilation.cu:439 cudaGraphExecKernelNodeSetParams only on wrapper->world.node",
        "unlike metric kernels `DilateKernel_Graph`/`DiffMetricKernel_Graph` receive `dilation` as a launch-value int (gpu_metrics.cu:384, 496); no per-pose SetParams path or graph exec for them",
        "updateParams guards only context identity + stream (426-428), NOT CaptureGeneration/dilation"
      ],
      "owner": "maintainer",
      "suggestedFix": "Document (and ideally enforce) that a graph exec is valid only for the exact dilation/dim/input identity present at createGraph. On any input-identity or dilation change, destroyGraph + re-create rather than reuse; or add a CaptureGeneration comparison in updateParams that forces recreation when inputs mismatched."
    },
    {
      "lens": "cuda-graph-lifecycle",
      "title": "Graph exec captures ctx.primary device pointers; recycle/rewrite of a context without a fresh createGraph leaves metric-node capture stale",
      "severity": "P2",
      "confidence": 75,
      "evidence": [
        "createGraph binds enqueues against ctx.primary buffers (graph.cu:346, rendered_image args) once",
        "metric enqueues pass cf->GetDeviceImagePointer()/dm pointers at capture (gpu_metrics:408,490) frozen into mem/node args",
        "updateParams re-binds only World projected/snapped/backface pointers (204-208), leaving metric-node pointers and dilation at capture epoch",
        "evaluation_context.h:55-58 CaptureGeneration comment: buffers can be rewritten in place (upload epoch changes)"
      ],
      "owner": "maintainer",
      "suggestedFix": "If the executor reuses a graph exec across a recycle/rewrite boundary, force destroyGraph+recreate (see finding 1). Otherwise the exec may silently produce result of the previous frame on a rewritten comparison/dilation buffer."
    },
    {
      "lens": "cuda-graph-lifecycle",
      "title": "completeFromPins is safe only after a same-stream completion gate; complete() provides it, completeFromPins alone does not",
      "severity": "P2",
      "confidence": 75,
      "evidence": [
        "completeFromPins (475-499) does no stream/event wait before dereferencing host pins",
        "graph.h:128-131 documents D2H-landed-via-query guarantee; launch() doc (graph.cu:448-452) warms that reading pins without a same-stream gate is a read-after-kernel launch race",
        "complete() provides the needed cudaStreamSynchronize before completeFromPins (465)"
      ],
      "owner": "maintainer",
      "suggestedFix": "No code change required if callers always go through complete() or an equivalent event gate with the same stream the graph launched on; document/prohibit a standalone completeFromPins against a just-launched Exec."
    },
    {
      "lens": "cuda-graph-lifecycle",
      "title": "Graph-free replay termination is byte-equal but NOT pipeline-shape-identical: it inserts a mid-stream sync that the graph path avoids",
      "severity": "P2",
      "confidence": 75,
      "evidence": [
        "test:271-272 EnqueueRenderPhase then CompleteRenderPhase (sync + D2H read) BEFORE metric enqueues; graph.cu:354-368 captures the same three enqueues with no intermediate sync",
        "reference cuda-graphs.md: workload: capture replaces enqueue with insertion into graph — no execution at capture",
        "Layer A/B equality holds (oracle asserts), but sync placement and kernel-reissue counts differ"
      ],
      "owner": "maintainer",
      "suggestedFix": "When claiming graph-free replay is 'behavior-identical', the replay should drop the intermediate CompleteRenderPhase and gate a single per-pose barrier + read. This makes the replay term the graph's exact kernel/stream shape."
    }
  ]
}
```

---

## Reference note

This repo does not bundle NVIDIA CUDA graph reference text (the oracle/robe is a stripped fork).
I grounded API semantics from the local `cuda-skills` reference snapshot
(`references/cuda-guide/04-special-topics/eda-frames.md`, which documents
`cudaStreamBeginCapture`/Global, `cudaStreamEndCapture`, single-origin-stream rejoin,
`cudaGraphInstantiate`, and the explicit node-update APIs — references available at
`/home/ajj/.pi/agent/skills/cuda-skill/references`). No claim in the analysis depends on a
local-absent fact; any such case is flagged as a caveat rather than assumed.

---

## Acceptance report