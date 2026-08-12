# Optimizer Path — Brainstorm Input (captured thoughts, 2026-08-12)

Status: input for a ce-brainstorm session (this is NOT a requirements doc and not
a plan). Purpose: capture the owner's five thoughts + the research run's
grounding so the brainstorm session starts warm.

## Source material a brainstorm must read first

- `.panoptes/optimizer-deep-dive/synthesis.org` — the master synthesis: 15+
  key findings, three Round-Updates sections, the dependency-ordered 8-step
  execution path (foundation → z-probe → multi-stage oracle → ablation harness
  → architecture cuts → variant switch → perf cuts → polish go/no-go).
- `.panoptes/optimizer-deep-dive/angles/01..06` — three rounds of deepening per
  angle (01 DIRECT variants, 02 cost-path foundation, 03 measurement apparatus,
  04 backend architecture, 05 compute perf, 06 differentiable polish).
- `papers/` + the Zotero corpus (`/home/ajj/zotero-paper-text/`, RAG-searchable
  via `zotero_rag_query`) — the primary texts: Flood 2018 (DIRECT-JTA, the
  tool's lineage), Mahfouz 2003, Gablonsky 2001, Jones 1993, Stripinis 2018/
  2021/2022, Yamazaki 2004, Jensen 2024, SoftRas/Modular Primitives/DiffDRR.
- `docs/plans/2026-08-11-006-refactor-shared-vm-layer-extraction-plan.md`
  (Follow-On Optimization Phase section) + the shared-VM-layer compound
  (`docs/solutions/conventions/jtml-shared-vm-layer-2026-08-11.md`).

## Owner thought 1 — modularizable optimizer: arbitrary "graphs"

The optimizer should be modularizable so arbitrary **graphs** can be loaded:
sequences of steps, each step = (algorithm variant, cost variant, its
parameters). The changes the run identified should be *configurations*, not
code edits.

**Grounding (what the run already established):**
- The seam work is mostly done: `OptimizerRunDriver` + `OptimizerRunLaunch`
  (by-value payload: calibration, frames, models, pose matrix, settings, 3×
  CostFunctionManager, directive) is the run boundary; `DirectOptimizer` takes
  an injected cost; angle 04's `StageScript`-as-data + `DeriveStageCostParams`
  pure functions + `DirectOptimizer::Options` (bit-identical defaults, R13) is
  the sketched shape — cuts A–E, additive-first, no new layer.
- DIRECT-JTA (Flood 2018) IS a graph: trunk (high dilation, full range) →
  3 branch restarts (lower dilation, fixed-size hyper-rectangles sized to
  escape symmetric traps) → z-leaf (asymmetric ranges, out-of-plane focus).
  JTML's current trunk → 2×branch → leaf is the same graph with different
  constants (branch ranges are byte-exact inheritance).
- The graph is the natural **unit of ablation**: the ablation runner (angle 03)
  should score graphs, not just variants.

**Open questions for the brainstorm:**
- What is the graph schema? Nodes = stages with {optimizer variant (classic
  DIRECT / GL / GL-gb / 1-DTDV / polish type), cost variant + weights, budget,
  ranges, dilation derivation, restart/seed semantics (start point = previous
  stage's optimum; hyper-rectangle size), termination}. Edges = stage ordering.
- What ships as v1 graphs? (Lineage config as the default; the run's evidence
  suggests the landing variant is 1-DTC-GL-gb.)
- Where does the graph live (JSON? C++ data? registry like CostFunctionManager)?
- What stays fixed: the driver seam, the GPU cost path, the IoU gates.

## Owner thought 2 — the best "graph" for this problem

Do we run certain forms of DIRECT at the beginning and change as we go?

**Grounding:**
- The run's variant evidence: current config (ε=0 classic DIRECT, all ties) is
  the literature's worst local-refinement regime; **1-DTC-GL-gb** wins at the
  ≥4k budget shapes (GL Pareto selection + 1-D trisection + globally-biased
  refinement restriction); 1-DTDV (diagonal vertex sampling) is dramatically
  better only when the optimum sits on the box boundary (joint limits?) — gated
  probe needed; GLh (hidden-constraint surrogate) only matters if the cost can
  emit NaN/Inf — verified: DIRECT_DILATION structurally cannot; MAHFOUZ/DD/
  sym_trap can (angle 02 round 2).
- The lineage graph already "changes as we go": high dilation smooths the cost
  for the trunk (explore), lower dilation sharpens it for the branches
  (exploit), the z-leaf is a dedicated out-of-plane stage. Flood gave the
  z-leaf **50,000 evals (equal to trunk)** — JTML cut it to 5,000; the leaf has
  never been exercised by any test.
- The polish question: Mahfouz's original method used **Nelder-Mead simplex**
  (negative weights) after its own coarse stage; angle 06's gradient polish is
  GO-gated on the z-profile probe's chamfer-surrogate valley depth, and the
  ablation should include a **simplex arm**.
- The DIRECT-JTA restart "sacrifices a notion of global convergence for
  improved asymptotic performance" — the graph design must state where the
  cover property is intentionally dropped.

**Open questions:**
- Default graph for the multi-stage oracle (Flood shape 50k/3×15k/50k vs JTML
  shape 20k/2×5k/5k vs a GL variant with the JTML budget)?
- Stage-to-stage algorithm switching: classic-DIRECT trunk → GL-gb branches →
  z-leaf with simplex/gradient polish?
- What the analytic battery (angle 01) scores before any GPU spend.

## Owner thought 3 — CUDA inefficiencies (deferrable)

Agreed deferrable. The known inventory (so it doesn't get lost): ~15 kernel
launches + ~5 synchronous D2H memcpys per eval; the ms/call meter has a 1000×
unit error on Linux (no CLOCKS_PER_SEC division) and excludes GPU time; nvtx is
declared but zero-instrumented; the stray `cudaGetLastError` at
render_engine.cu:713. All of this sits in the synthesis's step 7 (after
re-baseline, oracle, harness). The only interaction with the earlier steps: the
**instrumentation protocol (Cut 0) is cheap and should happen early** because
every perf decision needs measured numbers, and the meter fix (Cut 4) is a
correctness fix users can see.

## Owner thought 4 — parallel cost evals across the regions DIRECT wants to check

Per iteration, DIRECT's POH set can have several potentially-optimal boxes.
Today each is evaluated serially (one pose → render → metrics → scalar). If the
GPU has spare capacity (say 5 of N POH boxes fit concurrently), can we evaluate
the batch in parallel — **is the backend ready?**

**Grounding:**
- Current shape: `DirectOptimizer` calls `cost_(point)` (scalar, one pose per
  eval); the injected-cost lambda (`RunDirectStage`, optimizer_manager.cpp:1225+)
  sets the pose → renders (render_engine.cu, ~8 kernels) → metrics → scalar.
  Nothing is batched.
- Backend readiness: **mostly NO, additively reachable.** The render path is
  single-pose (pose transforms applied to the model); the metrics compose via
  in-place mutation on the rendered buffer (FastImplantDilationMetric writes
  back into the rendered image — batching needs per-pose scratch or
  stream-isolated eval state); zero `cudaStream`/`cudaEvent`/`cudaMemcpyAsync`
  in src/ (angle 05 round 2 grep-confirmed — streams would be green-field).
- Determinism: all active cost-path reductions are int atomics
  (order-independent) — a batched eval CAN be bit-identical if per-pose
  reductions are combined deterministically (the pose→score diff would prove
  it; angle 05's bit-identity protocol applies).
- The algorithm side: serial DIRECT picks one POH box per iteration; a
  batch-aware variant evaluates the whole POH set per iteration (parallel
  DIRECT exists in the literature — the 2022 study's variants are serial, but
  Flood 2018 line ~219 explicitly notes "the ability to make cost function
  calls in parallel" as an advantage of the similarity metric design).
- Angle 05's Cut 2 (iteration batching + double-buffered eval state) is the
  perf-side seed of this; angle 04's seam contract says "eval state is
  double-buffered and the cost layer can enqueue before it waits."

**Open questions:**
- Batch interface shape: cost becomes `batch(point[], costs[])` (optional
  additive API; single-point path unchanged)? A batch-aware DIRECT variant
  (evaluate the POH set per iteration)?
- GPU-side: N poses in one render pass (pose array in the kernel) vs N
  streamed single-pose renders vs a hybrid (batch the metric stage only)?
- What fills the batch: POH boxes? Multi-frame? Multi-variant at the same pose?
- Gates: bit-identity vs single-pose path; oracle IoU unchanged; speedup
  measured via the fixed meter.

## Owner thought 5 — anything else from the synthesis runs worth capturing

- **Measurement-first sequencing is the spine**: foundation (bug pins → fixes →
  one re-baseline) → z-profile probe → multi-stage oracle → ablation harness →
  architecture cuts → variant switch → perf → polish go/no-go. The brainstorm
  should confirm the ORDER and the gates, not re-derive them.
- **Decisions the run already resolved — do NOT re-litigate**: Mahfouz
  normalization semantics (paper-confirmed); the three-way dilation
  contradiction (Flood's constants + lineage configs go into the ablation);
  gb phase-switch numbers; GLh formula; the ms/call meter unit error; the
  DIRECT_DILATION NaN-impossibility.
- **The z-axis is an observability limit, not an algorithm problem**: the
  z-profile probe arbitrates; the real remedies are biplane or a known-geometry
  prior. Product question for the brainstorm: is biplane on the roadmap, and
  does that change what the z-leaf budget should be?
- **The re-baseline decision** (fix distance-map index → re-baseline exactly
  once): a decision to schedule, not a brainstorm topic.
- **The polish GO-gate** (z-profile valley depth on the chamfer surrogate) is a
  criterion; the simplex arm is now part of the ablation.
- **Oracle expansion** (multi-stage production shape, tibia, more frames) —
  what the measurement budget is worth spending on.
- **ML pathway** (segmentation → initializer-for-DIRECT) stays on the horizon;
  the graph design should not preclude an initializer node.
- **Process note**: this session's context is nearly exhausted; the brainstorm
  should run in a fresh session reading this doc + the synthesis + the angles
  (a fresh ce-brainstorm will have the full artifact base without the
  accumulated session weight).

## Recommended brainstorm agenda (for the session)

1. Agree the graph concept + schema (thought 1) — the container for everything
   else.
2. Decide the v1 graph(s) + the stage-switching policy (thought 2), grounded in
   the variant evidence + the lineage constants.
3. Scope the parallel-eval work (thought 4): algorithm variant + backend batch
   interface + GPU strategy + determinism gates. This is the one genuinely new
   idea the run didn't fully resolve — it deserves real brainstorm time.
4. Confirm the sequencing/gates (thought 5) and park CUDA inefficiencies
   (thought 3) as a deferred workstream with the known inventory attached.
