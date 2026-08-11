The Round 2 section is appended and verified in `angles/04-backend-architecture.org`. The write is the deliverable and it succeeded; the round-1 file content is untouched (append-only).

## Round 2 completion summary

All six dispatch items were tightened with source-verified answers, plus one new finding (the leaf-destruct error-gating asymmetry):

1. **Cut E gate made falsifiable** — one named perturbation (wrong-stage dilation emit at the branch transition), three named gates (IoU ≥ 0.85, costCalls on 20/25/30/35k, parity IoU), owned by angle 03's perturbation suite; structural reason the seam is blind to the emission class: the controller's stage channel is counter-derived (`stageText()` = `StageLabel(budgets, calls)`, optimizer_run_controller_core.h:159-165). Non-triggers explicitly excluded (z-profile probe consumes the adapter, not the seam). New evidence: `OptimizeCoordinator` (src/coordinator/optimize_coordinator.cpp:36) is already a headless single-stage executor — Cut E is specifically stage dispatch over the GPU loop.
2. **StageScript schema** — exact fields {kind, range, budget, repeat, cfm_index}, `repeat=0` semantics for the Sym_Trap no-search leaf, term-for-term emit-order preservation argument, and a dedicated direct-compile-testable TU for `DeriveStageCostParams`/`BuildStageScript` (optimizer_stage_script.{h,cpp}, one line in src/coordinator/CMakeLists.txt).
3. **Cut B pin strategy** — golden's end-to-end assertions stay verbatim; only the twin body (oracle_test.cpp:286-291) moves to `BuildGpuCostAdapter`; adapter's honest pin surface enumerated; calibration parameterization flagged.
4. **Double-buffered eval-state contract** — compute owns buffer state, manager owns the graph, domain sees only two `std::function` shapes with replay-order semantics.
5. **Torch boundary** — exact situs (CMakeLists.txt:77, src/compute/CMakeLists.txt:45,67, CostFunctionManager.h:28) and a gated two-move Cut F.
6. **Options cut** — defaults mapped line-by-line to today's ε=0/Jarvis/one-side/L2 code with guarded-divergence proof; plumbing traced through OptimizerRunLaunch → adapter → Initialize → RunDirectStage; all 18 DirectOptimizer call sites verified as 4-arg.