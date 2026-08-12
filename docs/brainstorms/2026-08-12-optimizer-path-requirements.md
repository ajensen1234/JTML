---
date: 2026-08-12
topic: optimizer-path-graph-container-and-parallel-evals
---

# Optimizer Path: Graph Container + Parallel Evals

## Problem Frame

JTML's optimizer run shape is hard-coded: trunk 20k → 2×branch 5k → leaf 5k
(cumulative 20/25/30/35k), with dilation and constants fixed in
`include/domain/settings_constants.h` — a verbatim transcription of Flood
2018's tibia configuration (the DIRECT-JTA lineage). Changing any of this —
the per-stage cost parameters, the optimizer variant, the stage ordering, the
budgets — means editing code. The deep-dive research run
(`.panoptes/optimizer-deep-dive/synthesis.org`, six angles, three rounds,
grounded in the Zotero corpus) resolved the *algorithm* and *measurement*
questions; this brainstorm decided the *container*: runs become **graphs**
(configurations, not code edits), and the parallel-eval idea (owner thought
4) is scoped as execution-level parallelism of the per-iteration POH batch.

The agreed shape, at a glance — the v1 graph is the current run shape as
data, unchanged behavior:

```
jtml-production (v1 graph — the current run shape expressed as stages)
┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ stage 1: TRUNK  │ → │ stage 2: BRANCH  │ → │ stage 3: LEAF   │
│ classic DIRECT  │   │ classic DIRECT   │   │ classic DIRECT  │
│ budget 20000    │   │ budget 5000 ×2   │   │ budget 5000     │
│ range (35)⁶     │   │ (15,15,25,25,25) │   │ (3,3,15,3,3,3)  │
│ dilation 6      │   │ dilation 4       │   │ dilation 1      │
└─────────────────┘   └──────────────────┘   └─────────────────┘
each stage = {cost variant + params, optimizer-variant slot (defaults to
classic), budget, ranges, seed semantics, repeat}
```

Two workstreams carry this forward:

1. **The graph container** — stages-as-data over the *existing* compute
   engine (driver seam, GPU cost path, IoU gates untouched): a feasibility
   PoC that must reproduce today's behavior bit-identically before any
   optimization improvement is attempted.
2. **Parallel evals** — today each DIRECT iteration evaluates its POH batch
   (2×|POH| new centers, typically 4–10 poses) fully serially, one
   render→metrics→scalar round trip at a time. The target is streamed N-way
   execution of that batch (N GPU streams, one pose each, kernels untouched),
   bit-identical by construction.

Sequencing is confirmed as the research run's dependency-ordered path
(measurement-first, one re-baseline, every step gated) — the brainstorm
confirmed the order, it did not re-derive it:

```
Phase 0 Foundation → Phase 1 Measurement → Phase 2 Graph container →
Phase 3 Algorithm (battery-scored) → Phase 4 Performance (parallel w/ 1–3)
→ Phase 5 Polish horizon (GO-gated)
```

---

## Actors

- A1. **Owner/developer**: registers graphs in the C++ registry; the only
  actor who changes graph structure (a code change, by design).
- A2. **Measurement harness** (multi-stage oracle + z-profile probe +
  ablation runner): the only actor that *scores* graphs and variants; no
  variant or perf claim is accepted without its gates.
- A3. **Production app** (widgets + QML, via the OptimizerRunDriver seam):
  consumes the default graph unchanged; must observe no behavioral
  difference in v1.

---

## Key Flows

- F1. **Graph-driven optimizer run**
  - **Trigger:** A run launch reaches the driver seam (production UI or the
    multi-stage oracle driving `OptimizerRunController` directly).
  - **Actors:** A1 (defines the graph), A3 (launches), the driver seam.
  - **Steps:** (1) launch names a registered graph; (2) the stage script is
    built from the graph (pure builder, no recompilation for parameter
    changes within declared axes); (3) each stage runs through the existing
    `RunDirectStage` runner + injected-cost lambda; (4) stage N+1 seeds from
    stage N's recovered optimum; (5) stage bookkeeping (cumulative costCalls,
    stage text, dilation) emits through the existing observation channel;
    (6) the IoU ≥ 0.85 appearance gate arbitrates the result.
  - **Outcome:** A run executes the graph's stages in order with unchanged
    semantics; the manifest records exactly which graph + parameter overrides
    were measured.
  - **Covered by:** R1, R2, R4, R6, R7
- F2. **Harness scoring of a graph/variant**
  - **Trigger:** An ablation run or analytic-battery run for a named graph or
    a variant in a stage slot.
  - **Actors:** A2, A1 (interprets results).
  - **Steps:** (1) manifest references a graph by name + declared-axis
    overrides; (2) the harness runs the probe/oracle/battery under the graph;
    (3) results land in `ablation.json` (per-run variant dimensions: gb
    semantics, split rule, ε, dilation schedule + semantics, Canny config,
    z-leaf budget, render resolution — recorded, never assumed); (4) gates
    evaluate (oracle green, bit-identity where claimed, measured improvement
    within pre-registered bands); (5) a pass may promote the graph or variant
    to the default — a code change by the owner.
  - **Outcome:** Every decision about variant switching or perf work is a
    data point in the manifest, scored before it touches production.
  - **Covered by:** R2, R3, R8, R9, R10

---

## Requirements

**[Graph container and schema]**

- R1. The optimizer run shape is a **graph**: an ordered sequence of stages
   (a path; variable length 1..N), each stage carrying {cost variant +
   parameters, optimizer variant, budget, ranges, seed semantics, repeat}.
   Edges are implicit: stage N+1 seeds from stage N's optimum.
- R2. Graphs are **C++-typed data in a named registry** (the
   `CostFunctionManager` registration pattern). The ablation manifest
   (`ablation.json`) references graphs by name and parametrizes only the
   declared ablation axes (dilation schedule + semantics, z-leaf budget, ε,
   Canny config, backface, render resolution). JSON never invents graph
   structure.
- R3. Each stage has an **optimizer-variant slot** (the
   `DirectOptimizer::Options` surface: selection, ε, split rule, gb phase
   switch, GLh, size measure) defaulting to today's classic DIRECT behavior
   bit-identically. In v1 the slot is empty/default everywhere — stage
   switching (e.g., GL-gb trunk → classic branches → polish leaf) is policy
   filled only after the harness scores it. The slot must accept any future
   stage kind: polish, an ML initializer prefix node, a no-search leaf.
- R4. Graph execution goes through the **existing seams**: the
   `OptimizerRunDriver` interface (`include/coordinator/optimizer_run_driver.h`),
   the `RunDirectStage` injected-cost lambda, the GPU cost path, and the IoU
   gates. No new layer, no RunBackend enum, no executor extraction — Cut E
   stays deferred behind its falsifiable gate (wrong-stage-dilation
   perturbation × three gates).
- R5. The schema must not preclude: biplane (the same costs over a second
   projection — a single 6-DoF pose transformed to the second image; z falls
   out of the DIRECT cost landscape naturally), a tiered-dilation cost
   variant, and a polish stage consuming the existing budget accounting
   (parity counting).

**[v1 graph set and feasibility PoC]**

- R6. The v1 registry ships **one graph: `jtml-production`** — the current
   production shape (trunk 20k → 2×branch 5k → leaf 5k, classic DIRECT
   everywhere, dilation 6/4/1). Other stage kinds may exist as stubs, but no
   alternate optimization scheme ships in v1. This is the feasibility PoC:
   stages-as-data over the existing engine.
- R7. The PoC must be **behavior-preserving**: same cumulative caps
   (20/25/30/35k), same stage bookkeeping, same recovered poses, oracle green
   (IoU ≥ 0.85), qml parity unchanged. Preservation is asserted via the
   lineage's four invariants with assertion homes in the multi-stage oracle:
   group-once dilation across branch repeats; per-repeat re-seed (assert the
   recovered-pose *sequence*, not just the final pose); asymmetric z-leaf;
   frame-to-frame seed chaining.

**[Measurement-first spine and gates]**

- R8. Sequencing is the research run's dependency-ordered path, confirmed
   as-is: **Phase 0** foundation (bug pins → behavior-neutral fixes → the
   live distance-map index fix → exactly one single-variable re-baseline,
   Canny pinned 3/0/150, recovered-pose delta recorded; plus the CUDA-free
   finite-check at `DirectOptimizer::EvaluateCostFunction` and the meter fix
   — see R10) → **Phase 1** measurement (z-profile probe → multi-stage oracle
   on the driver seam → ablation runner + `ablation.json`) → **Phase 2** graph
   container (Cut A pure builders in parallel with Phase 1; Cut B script-driven
   loop + adapter after the oracle exists to pin it; Cut C `Options` with
   bit-identical defaults; Cut F torch include/link hygiene) → **Phase 3**
   algorithm (analytic battery, then the variant switch) → **Phase 4**
   performance (parallel to Phases 1–3) → **Phase 5** polish horizon.
- R9. **No algorithm delta before the apparatus**: the variant switch is
   scored by the analytic battery (22 n=5 DIRECTGOLib v1.2 + Hartman6 + GKLS,
   the ε-arm {0, 1e-7, 1e-4}, gb-semantics arms (canonical + code-as-shipped),
   DIRECT-l one-per-group arm, staged basin-capture metric) and the oracle
   before any production change. 1-DTC-GL-gb is a candidate, not a
   commitment. Decisions the run resolved are not re-litigated: Mahfouz
   normalization (kernel-as-spec), the dilation three-way (ablation axis),
   gb numbers, the GLh formula, DIRECT_DILATION NaN-impossibility.
- R10. **Cut 0 instrumentation + Cut 4 meter fix land early** (foundation,
   parallelizable): the ms/call meter's 1000× unit error on Linux is a
   correctness fix users can see (IPS/ETA wrong ~1000×); the fixed meter must
   agree with an independent wall-clock measurement (|meter − wall| ≤ 5% or
   1 ms over a 10 s window). Measured numbers (cudaEvent GPU-active, kernel
   breakdown, evals/s bands) replace the derived 20–60 evals/s estimate
   before any perf or variant claim — the falsification rule stands:
   GPU-active > 1 ms ⇒ kernel-level work first (ncu), cut order inverts.

**[Parallel-eval workstream (streamed N-way)]**

- R11. The parallel-eval target is **execution-level parallelism of the
   per-iteration POH batch**: streamed N-way — N `cudaStream`s, one pose per
   stream, existing kernels untouched, N-bank eval state (N render buffers +
   N score banks; the double-buffer situs from the run's Cut 2 extended from
   2 to N). The GPU is under-occupied per kernel (simple 12k-triangle meshes,
   small crops), not just idle — N streams keep more work in flight.
- R12. **The batch seam**: `DirectOptimizer`'s boundary stays
   `std::function<double(const Point6D&)>` and gains one optional sibling
   `std::function<std::vector<double>(const std::vector<Point6D>&)>`;
   the single-point path is unchanged. Replay-ordered bookkeeping keeps
   calls, optimum, and callback order identical whether the cost layer
   batches or not. A Tier-0 headless replay test asserts the batch callback
   receives exactly the sequential pose sequence. Determinism: per-pose int
   atomics, fixed combine order — a pose→score diff proves bit-identity vs
   the serial path. **Batchability is a variant property, not a cost-layer
   one**: the seam's contract states that the optimizer MAY batch any set of
   pairwise-independent evals (DIRECT's POH batch qualifies; a future
   Nelder-Mead polish's decision-dependent steps do not, beyond its initial
   simplex) and that the cost layer guarantees per-point results in input
   order with identical observable behavior either way. Whether a variant
   batches is its internal decision — no capability advertisement is needed
   for correctness in v1; a `batchable` flag on the variant slot is a
   one-line addition when a second optimizer kind lands, so the harness can
   record which variants exercised the parallel path.
- R13. **Gates** for the workstream: headless green, oracle green,
   bit-identity diff empty, measured improvement within the pre-registered
   bands (async copies → 60–300 evals/s; N-way → 1,000–5,000; Flood's 3,000
   evals/s on a 2018 GTX 970 as the cross-era parity line). A cut that
   doesn't move the needle is reverted in the same jj change.
- R14. **Caching/memoization is out of scope**: DIRECT never re-evaluates a
   pose within a stage (each trisection evaluates only new centers), and
   cross-stage config changes (dilation) invalidate cached costs anyway.
   Same-pose-same-config repeats are intentional measurements only (probe
   noise-floor 5× repeats, start-pose IoU checks).

**[Parked workstreams]**

- R15. Parked with the run's inventory attached, each gated on Cut 0's
   numbers: fragment-fill removal (Cut 3, conditional trigger), metric fusion
   (Cut 5 — distance-map + overlap staying on GPU without host round-trips),
   render-resolution arm {1024², 512², 256²}, block-size sweep, the
   pose-array batched render (design B — pose axis through the render
   kernels, SoA pose data, shared read-only model data, bindless
   re-examination) as the follow-up if occupancy measurements justify it, and
   same-pose multi-config batching (rejected as a batch mode; a
   tiered-dilation *cost variant* instead).
- R16. Parked on the horizon: the four-arm polish study (A/B/C/D incl. the
   plain Nelder–Mead simplex arm — ~150–200 host lines, zero new kernels)
   gated on the G1–G4 precondition via the term-decomposed z-profile probe,
   with the Yamazaki-style decoupled z-polish as the z-arm; biplane (deferred
   — mechanism understood, architecture keeps it cheap); the ML initializer
   (graph prefix stub only); the NO-GO list stands (photometric gradient,
   torch bridge, NDC depth channel).

---

## Acceptance Examples

- AE1. **Covers R6, R7.** Given only the `jtml-production` graph registered,
  when the multi-stage oracle drives it through the driver seam, costCalls()
  lands on the 20/25/30/35k caps, the four lineage invariants hold, and
  per-frame IoU ≥ 0.85 against the re-baselined values — no code edits to the
  optimizer, the cost path, or the gates.
- AE2. **Covers R11, R12.** Given a stage whose iteration produces a POH
  batch of 6 poses, when the streamed N-way path runs, the recorded
  pose→score sequence is bit-identical to the serial path (empty diff), the
  replay test passes headlessly, and the fixed meter shows the pre-registered
  band — not a display artifact.
- AE3. **Covers R3.** Given a graph with a non-default optimizer variant in a
  stage slot, when the harness scores it, the manifest records the variant
  dimensions (gb semantics, split rule, ε, dilation schedule + semantics)
  — the harness always knows exactly which variant it measured.
- AE4. **Covers R10.** Given the fixed meter running over a 10 s window,
  |meter − wall| ≤ 5% of wall (or ≤ 1 ms) against an independent
  steady_clock + cudaEvent measurement, with the wall−GPU gap reported.

---

## Success Criteria

- The owner can express any optimizer run shape (1..N stages, per-stage
  cost/variant/budget/ranges) as a registry entry and have it run, measured,
  and scored — without editing the optimizer, the GPU cost path, or the
  gates. Stage-switching policy is data, arbitrated by the harness.
- The parallel-eval workstream lands as streamed N-way with proven
  bit-identity and a measured evals/s figure inside the pre-registered band —
  and the "is the GPU really the bottleneck?" question is answered by Cut 0
  data, not by the derived estimate.
- A downstream planner (`/ce-plan`) can produce the implementation plan from
  this doc + `.panoptes/optimizer-deep-dive/synthesis.org` without inventing
  product behavior, scope boundaries, or success criteria — and without
  re-litigating the run's resolved decisions.

---

## Scope Boundaries

- No variant switch ships in v1 (scored by the battery + oracle, then
  landed). 1-DTC-GL-gb stays a candidate.
- No optimization-scheme work while the stage machinery is being built —
  v1 ships one graph with stubs for other stage kinds.
- No N-way GPU-strategy work beyond streamed launches: pose-array rendering,
  metric fusion, bindless, and the block sweep are parked (R15).
- No caching/memoization of cost results (R14).
- No same-pose multi-config batching (a tiered-dilation cost variant
  instead, also parked).
- No biplane implementation (mechanism recorded; z falls out of the DIRECT
  landscape over two projections).
- No ML initializer implementation (graph prefix stub only).
- No fragment-fill removal or metric fusion (parked, gated).
- No new layer / RunBackend enum / headless executor extraction (Cut E
  deferred behind its falsifiable gate).
- No torch in the cost path (Cut F include/link hygiene only).
- No re-litigation of the run's resolved decisions (R9).

---

## Key Decisions

- **C++-typed graphs + named registry, not JSON**: matches the run's
  sketch; zero parsing; fully pinnable and headless-testable; the manifest
  references and parametrizes, never defines.
- **Path graph (ordered stages), not explicit edges**: every known scenario
  (lineage, flood shape, polish, initializer) is a path extension; conditionals
  (G1–G4) live in runner branches, not the graph.
- **v1 = `jtml-production` only, stubs allowed**: feasibility PoC over the
  existing engine; other optimization schemes are explicitly out of scope
  while stage creation is being figured out.
- **Per-stage optimizer-variant slot, empty by default**: the schema holds
  stage switching; the policy is harness-arbitrated data.
- **Parallel evals = streamed N-way (design A)**: kernels untouched, bug
  surface minimized, launch-level only; the run's depth-2 double-buffer
  machinery is the seed, extended from 2 banks to N.
- **Sequencing/gates confirmed as-is**: the run's dependency-ordered path
  and pre-registered gates; measurement-first; one re-baseline.
- **Biplane deferred**: mechanism understood (same costs, two projections,
  single pose transformed to the second image; z falls out naturally); the
  architecture must keep it cheap, no explicit work now.
- **Caching = no-op, recorded**: DIRECT never re-evaluates within a stage;
  cross-stage config changes invalidate.
- **Cut 0 + Cut 4 absorbed from the deferred-CUDA inventory into Phase 0**:
  instrumentation and the meter fix are cheap, early, and every later
  decision needs measured numbers.

---

## Dependencies / Assumptions

- The grounding is `.panoptes/optimizer-deep-dive/synthesis.org` (final,
  three rounds) + the six angle files + this session's input doc
  (`docs/brainstorms/2026-08-12-optimizer-path-brainstorm-input.md`).
  Planning must read the synthesis; it is the normative source for every
  gate, pin, and pre-registered band referenced here.
- The driver seam (`include/coordinator/optimizer_run_driver.h`), the
  `RunDirectStage` injected-cost lambda, and `DirectOptimizer`'s injected
  cost boundary are the shared vocabulary; nothing re-owns them.
- `include/domain/settings_constants.h` is a verbatim Flood tibia
  transcription — the lineage is the spec for the v1 graph's constants.
- The Phase-2 re-baseline stays single-variable (distance-map index fix
  only; Canny pinned 3/0/150) or the recovered-pose delta is uninterpretable.
- Gates: IoU ≥ 0.85 is the appearance arbiter, never raw pose; z-gap is
  banded-informational; bit-identity is required wherever claimed; one jj
  change per logical step.
- The batch seam's `std::function` sibling introduces no CUDA types into the
  domain header (layer-purity contract of `include/domain/direct_optimizer.h`).
- nsight-systems needs `pixi add` (nsight-compute is already in the lock).

---

## Outstanding Questions

### Resolve Before Planning

- (none)

### Deferred to Planning

- [Affects R2][Technical] Exact registry shape and TU placement for named
  graphs (mirroring the `CostFunctionManager` registration pattern vs a
  simple factory map).
- [Affects R1][Technical] Final StageSpec field set — the run's sketch is
  {kind, range, budget, repeat, cfm_index}; confirm against
  `DirectOptimizer::Options` and the stub kinds.
- [Affects R11][Technical] The N-bank count (batch capacity) and whether the
  eval-state bank array is a compile-time constant or a run-time size; the
  run's pinned situs (second renderer_output_b_ in RenderEngine + second
  score-bank in GPUMetrics) extends to N.
- [Affects R10][Technical] Whether `pixi add nsight-systems` is accepted into
  the lockfile for the Cut 0 recipe.
- [Affects R9][Needs research] GPU-active verdict at Cut 0 (≤ 1 ms sync-bound
  confirmed vs > 1 ms falsified — kernel work first): the load-bearing
  measurement of the whole perf program; unmeasured until the instrument
  lands.

---

## Next Steps

-> /ce-plan for structured implementation planning
