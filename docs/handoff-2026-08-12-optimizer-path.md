# Handoff — Optimizer Path: Graph Container Done, Selection/Perf/Measurement/Algo/Polish Open

Date: 2026-08-12. Repo: JTML (Qt6/CUDA C++ knee-implant registration, jj VCS).
Session scope: the optimizer-path arc — brainstorm → requirements → plan 008
(graph container) executed end-to-end → plan 009 (stage-graph selection)
authored. Working tree clean at `vyrwrmlx 53a9f8fe`.

## 1. Where we are (one paragraph)

The optimizer run shape is now **data, not code**: plan 008 landed the
StageScript container (`BuildStageScript` + named-graph registry + script-
driven `Optimize()` loop + `DirectOptimizer::Options` slot) with **bit-
identical production behavior** — proven by the U6 multi-stage oracle
passing unchanged through the relocation, the qml-parity instrument, and the
flat oracle. The foundation is trustworthy: 7 cost-path bugs pinned and
fixed (incl. the live distance-map index bug), exactly one single-variable
re-baseline recorded, the finite-check and meter-relevant context in place.
Along the way the plan's own measurements **refuted two load-bearing
assumptions** of the research run (~7k evals/s vs derived 20–60; SymTrap
costCalls 0 vs the pinned 20000/30000) — the follow-up plans must be
re-based on the measured numbers, listed in §4.

## 2. Done (committed)

| Change | Content |
|---|---|
| `e15fe849` | U1 Tier-0 metric-semantics pins + CPU references (RED pins via [!mayfail]) |
| `8bf42b13` | U2 behavior-neutral cost-path fixes (stage guard, DD ×2, sym_trap tibia, dead dup) + getStage() accessor |
| `435ca621` | U3 CUDA-free finite-check at EvaluateCostFunction (infeasible evals never stored) |
| `0cccf43a` | U4 distance-map index fix + adversarial GPU probe + ONE re-baseline |
| `96f41927` | U5 z-profile probe (term decomposition, lineage dilations, yamazaki/coupling modes) |
| `d013e41b` | U6 multi-stage oracle on the driver seam (caps, 4 lineage invariants, tibia-after-femur, SymTrap pins) + end_frame_index_ UB fix |
| `0205c48b` | U7 stage-script TU (StageSpec, builders, jtml-production registry) |
| `73682346` | U8 DirectOptimizer::Options (bit-identical defaults, fail-fast stubs) |
| `70912511` | U9 script-driven Optimize() + BuildGpuCostAdapter — the PoC, bit-identical |
| `7a3d03ff` | U10 torch link hygiene (ATen include + PUBLIC→PRIVATE) |
| `988102a1` | Simplify pass (−103 net LOC, oracle-verified) + plan checkboxes/execution notes |
| `553a5fc7` | Guide `docs/optimizer-stage-graphs.md` + compound learnings ×2 + header comment fix |
| `53a9f8fe` | **Plan 009** — stage-graph selection (model + viewmodel; view parked) |

Suite state: 53 headless tests green; oracle suite green (`jtml.oracle`,
`jtml.oracle_multistage` ~15 s, `jtml.qml_parity_check`); `jtml.z_profile`
green. One pre-existing environment failure: `jtml.probe_vtk` (GLX
`glXGetClientString` symbol error — see `docs/solutions/tooling-decisions/jtml-rendering-runtime-xcb-qvtk`).

## 3. What's next — the open workstreams (dependency-ordered)

### A. Plan 009 execution — stage-graph UI selection (READY TO EXECUTE)
`docs/plans/2026-08-12-009-feat-stage-graph-selection-plan.md`, U1–U3:
- U1 `OptimizerSettings.stage_graph_name` (QString, "" default);
- U2 manager `Initialize` consumes `StageGraphByName` when set (fail-fast on
  unknown) + the parity pin (`jtml-production` ≡ default settings-built
  shape);
- U3 `SettingsBridge` viewmodel (`stageGraphs` CONSTANT + `stageGraphIndex`,
  −1 = settings-built; persistence rides `SaveOptimizerSettings`).
View wiring (QML ComboBox + widgets picker) is the deferred follow-up.

### B. CUDA/perf workstream (NO PLAN FILE YET — brainstorm/plan next)
Scope already spec'd as requirements R10–R14 + AE2/AE4 in
`docs/brainstorms/2026-08-12-optimizer-path-requirements.md`:
- Cut 0 instrumentation (nvtx3/cudaEvent; `pixi add nsight-systems` needed)
  — answers the "is the GPU really the bottleneck?" question per-eval;
- Cut 4 meter fix (both sites; the 1000× unit error is a known-visible
  defect until this lands; wall-clock cross-check gate);
- Cut 1 async terminal copies; streamed N-way (N cudaStreams, kernels
  untouched, N-bank eval state) + batch seam (`std::function` sibling,
  replay bookkeeping, Tier-0 replay test);
- Parked inventory: fragment-fill removal (Cut 3, conditional), metric
  fusion (Cut 5), render-resolution arm, block-size sweep, pose-array
  design B.
**Precondition before planning**: re-base the pre-registered bands on the
measured ~7k evals/s (§4) — the derived 20–60 and the 60–300/1,000–5,000
bands are stale.

### C. Measurement plan — ablation runner + ablation.json (NO PLAN FILE)
The instruments exist (U5 probe, U6 multistage oracle, perturbation-suite
preconditions in the run's angle 03 spec); what's missing: the manifest-
driven runner, `ablation.json` schema wiring (z_profiles + oracle_multistage
records exist as data — the runner/manifest/perturbation/benchmarks +
ci/nightly profiles don't), the perturbation suite (transfer-function grid +
7 injections → minimal-detectable-regression table), and the benchmarks
block (Jensen/Arulampalam/Burton regime tags).

### D. Algorithm plan — variant switch (NO PLAN FILE)
Analytic battery (22 n=5 DIRECTGOLib v1.2 + Hartman6 + GKLS, ε-arm
{0,1e-7,1e-4}, gb-semantics arms, DIRECT-l arm, staged basin-capture
metric), then 1-DTC-GL-gb as the candidate. The `Options` divergence
branches are fail-fast stubs today — this plan implements the real
semantics behind the battery's verdicts.

### E. Polish horizon (NO PLAN FILE)
Four-arm study (A/B/C/D incl. the ~200-line simplex arm), G1–G4
precondition via the term-decomposed probe (note §4: **G2 measured
FAILING** — the z-claim pre-registers "unchanged within noise"), Yamazaki
decoupled z-arm. Zotero pulls: 3 abstracts indexed in the KB
(`zotero-pull-*` sources); full texts still needed — 1906.07870 (the
boundary term), Veriserum 2509.05483, NVDiffrast repo.

### F. Hygiene pass (small, well-specified)
From plan 008's execution notes: Mahfouz value guards
(`implant_mahfouz_metric.cu:324/:446`), curvature de-scope
(`DIRECT_DILATION.cpp:42-43` + the cudaMalloc(0) silent-error path),
**`updateCostFunctionParameterValues` no-op family fix + PBT round-trip
invariant** (all three overloads), dilation doc reconciliation
(baseline.json `dilation_px {6,3,1}` → measured; golden_oracle.org),
ON-vs-OFF silhouette IoU protocol. Also: probe-data consumption (z_leaf
budget axis, G1–G4 data) once the ablation runner exists.

### G. Deferred/parked (do NOT start without a decision)
Biplane (mechanism understood; z falls out of the DIRECT landscape over two
projections — architecture keeps it cheap), ML initializer (graph prefix
stub only), tiered-dilation cost variant, metric fusion (Cut 5), ML
classifiers for sym-trap (data-scale NO-GO; the ψ/continuity measurement is
the cheap transfer).

## 4. Measured facts that must NOT be re-derived (re-base the plans on these)

- **~7,000 evals/s** on the RTX 3090 Ti (~15 s total for the whole
  production-shape multistage run). The derived 20–60 evals/s is refuted by
  ~2 orders; the perf plan's bands and the ≥95%-host-block framing must be
  re-examined against this (per-eval sync structure may still dominate, but
  the wall-clock story changed completely).
- **SymTrap costCalls == 0** (not 20000, not 30000): the `if (!sym_trap_call)`
  guard at `optimizer_manager.cpp:927` wraps trunk AND branches; only
  leaf-init + CalculateSymTrap run (60 uncounted evals); stageText stays
  "Idle"; relay count 61 (60 + 1 restore). Both the synthesis's and the
  review's pins missed the outer guard.
- **G2 precondition FAILS at leaf dilation 1**: chamfer valley 0.9886 <
  dilated valley 0.9965 — the polish z-claim gate will pre-register
  "expected unchanged within noise"; the run's honest gating applies.
- **Re-baseline verdict**: direction confirmed (frame-0 z +0.014 mm toward
  fem.jts, IoU 0.993627→0.993779); magnitude refuted — the "6.31 mm frame-1
  gap" was a Qt5-capture artifact; the post-fix oracle's frame-1 gap was
  1.19 mm and unmoved. Frames 1/2 recovered bit-identically pre/post.
- **b2 == b1 on frame 0 is legitimate** (deterministic DIRECT from
  identical seeds); the invariant is asserted via stage-recovery sequences,
  not b2≠b1.
- Probe data: off-axis angle 1.086° vs camera-z; in-plane→z coupling
  −2.25 mm/mm at d6; Mahfouz float-path noise floor 0.14–0.56 (NOT
  int-atomic); T1 argmin at the +15 mm boundary (z-blind-ish); ranking
  DIRECT_DILATION > DIRECT_MAHFOUZ > DIRECT_DILATION_T1.
- `updateCostFunctionParameterValues` (all three overloads) is a silent
  no-op (by-value getter mutation) — production uses
  `setIntParameterValue` on the active class.
- `end_frame_index_` was uninitialized on the SymTrap path (fixed one-line
  in U6); `b2==b1` and the `stageText` "Extra Z-Translation" labels are
  channel facts.

## 5. Stale sources to distrust (flagged, with the correction home)

- `.panoptes/optimizer-deep-dive/synthesis.org` — the SymTrap "20000" pin
  and the derived 20–60 evals/s (corrected: §4; compound doc
  `docs/solutions/logic-errors/jtml-symtrap-outer-guard-pin-2026-08-12.md`).
- `test/golden/baseline.json` `dilation_px {6,3,1}` — known-stale docs-claim;
  engine runtime is 6/4/1; reconciliation happens in the hygiene pass after
  the probe data.
- The pre-008 research prose around "10–30 min/frame" and "TIMEOUT 600 for
  the multistage oracle" (measured ~15 s; TIMEOUT 7200 kept per spec).

## 6. Warm-up for the next session (30 seconds, not 30 minutes)

The context KB has the whole artifact base indexed — from any fresh pi
session in this repo:

```
ctx_search(queries: [
  "SymTrap costCalls zero outer guard",        # plan 008 + compound doc
  "G2 precondition chamfer dilated",            # angle 06 + execution notes
  "re-baseline verdict 6.31mm",                # execution notes
  "streamed N-way batch seam",                 # requirements R11-R12
  "stage graph selection",                     # plan 009
], source: ...)  # sources: panoptes-optimizer-deep-dive,
                 # optimizer-path-requirements, plan-008-graph-container,
                 # jtml-solutions-learnings, zotero-pull-*
```

Read order if you prefer files: `docs/optimizer-stage-graphs.md` →
`docs/plans/2026-08-12-008-...-plan.md` (Execution Notes) →
`docs/plans/2026-08-12-009-feat-stage-graph-selection-plan.md` →
`docs/brainstorms/2026-08-12-optimizer-path-requirements.md` (R10–R14 for
the perf workstream).

## 7. Evaluation-executor workstream — graph-backed greedy (plan 011, U1-U8)

**Status 2026-08-19 — U1-U7 landed, U8 harness pending manual GPU run:**
- U1 `EvaluationContext` + generic `GraphRecipe` (private mutable state, null-init, destruction wait, `BankAdmission` half-memory + `graph_overhead_bytes`, one private `cudaGraphExec_t` per context) — `feat(compute): private EvaluationContext + generic GraphRecipe surface (U1)` (9703aba0)
- U2 `TEST_IMPACT_MATRIX.md` + frozen `test/golden/graph_pre_registration.json` (`8/16/32` poses, `512x512`, `300k` tri, `N=1/2/4`, `abs 1e-12 / rel 1e-9`, `p99>10%` + stage `>5%`) — `feat(test): test-impact matrix + frozen baseline harness (U2)` (f5234bd5)
- U3 `GraphPreflight` Global capture probe (`cub::DeviceScan::ExclusiveSum` + pinned `memcpy` + `atomicAdd`, `GraphPreflightResult`) — `feat(compute): graph capture compatibility probe (U3)` (5df29074)
- U4 device-driven persistent chunk workers (`cudaMemsetAsync(0) -> PrepareLaunchPacketKernel -> overflowCheck(dev_overflowFlag) -> Stride/Fill predicated`, fixed `NX = min(maxBlocksPerSM*SMcount, ceil(SAFE_CAP/256))`, `dev_nextCandidate/dev_nextChunk` in footprint, metric `max(2048x2048)` fixed-max) — `feat(compute): device-driven persistent chunk workers (U4)` (b8029028)
- U5 `direct_dilation_monoplane` graph recipe (reusable topology key `recipeId+biplane+cub_storage+curvature+maximum_stride_size`, one private Exec per context, `SetParams` update, two-contexts-in-flight test) — `feat(compute): monoplane DIRECT_DILATION graph recipe (U5)` (c4fce8ed)
- U6 `EvaluationExecutor::RunBatch` greedy lease/checkout, `firstSubmission` atomic `R7/R8` split, ordered `result[lease.input]`, `QSignalSpy` `A1/R11` + watchdog — `feat(compute): greedy batch wiring + ordered assembly (U6)` (1a19b463)
- U7 layered oracle gate (Layer A image byte-identical, Layer B raw int `pixel/distance/edge/union` identical, Layer C `double` within `abs 1e-12 / rel 1e-9`, `>=3x` repeats, Tier-2 `IoU≥0.85`) — `feat(oracle): layered oracle gate (U7)` (3fef7471)
- U8 paired harness `test/oracle/graph_throughput_oracle_test.cu` (warmup 3 discard, 10 trials, `p50/p99`, `evals/sec`, Amdahl `S(N)=1/((1-P)+P/N)`) + `test/golden/graph_performance_baseline.json` stub (`nsys not available, manual run required`, `pending_manual_gpu_run`) — `feat(oracle): paired throughput/latency + Nsight proof (U8)` (pending). Gates: (a) Amdahl band, (b) `p99>10%` + stage `>5%`, (c) U7 PASS, (d) Nsight `≥30%` concurrent at N=2, `<50us` gap, zero `Synchronize`/`Memcpy` (grep + timeline). Revert rule `jj abandon` functional change, retain matrix/baselines. Cut-0 `98us` is context, not speedup. Run on RTX 3090 class as `cut0_measurement.md`.

Next: run U8 harness on RTX 3090 with `nsys profile` to populate `graph_performance_baseline.json` (`measured_P`, `amdahl_S_N2`, `nsight_concurrent_percent_N2`, `max_gap_us`, `zero_sync`) and decide retain vs `jj abandon`.

## 8. Suggested next moves (pick one)

1. Execute plan 009 (U1–U3; the UI picker parked) — smallest, ready now.
2. Brainstorm → plan the CUDA/perf workstream (re-based on ~7k evals/s) —
  the highest-value arc — **now superseded by plan 011 execution above for the greedy executor**.
3. The measurement plan (ablation runner) — the scoring surface everything
  else (algorithm/polish) is judged on.
4. The hygiene pass — small, unblocks the meter visibility + the parameter
  no-op.
