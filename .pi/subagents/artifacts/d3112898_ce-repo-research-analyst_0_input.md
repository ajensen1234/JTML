# Task for ce-repo-research-analyst

Planning context (JTML repo: Qt6 + VTK 9.3 + CUDA 12.4 + OpenCV C++20, pixi; jj VCS):
- Plan: 'graph container arc' — make the optimizer run shape (hard-coded trunk 20k -> 2x branch 5k -> leaf 5k in OptimizerManager::Optimize) a data-driven sequence of stages (StageScript: vector of {kind, range, budget, repeat, cfm_index}) over the EXISTING compute engine. Feasibility PoC: bit-identical behavior, no GPU kernel changes, no new layer.
- Origin doc: docs/brainstorms/2026-08-12-optimizer-path-requirements.md (R1-R16). Normative research base: .panoptes/optimizer-deep-dive/synthesis.org (angle 04 = backend architecture: StageScript-as-data, DeriveStageCostParams/BuildStageScript pure functions in a dedicated TU include/coordinator/optimizer_stage_script.h, Cut B = script-driven Optimize loop + BuildGpuCostAdapter, four lineage invariants, Cut E deferral gate, DirectOptimizer::Options bit-identical defaults).
- Key areas to examine and report on (exact file paths, class/function names, current structure):
  1. src/coordinator/optimizer_manager.cpp — the Optimize() stage loop (trunk/branch/leaf), RunDirectStage, stage bookkeeping (costCalls/stageText), the three CostFunctionManager members, the sym_trap path, the two ms/call meter sites.
  2. include/coordinator/optimizer_run_driver.h + src/coordinator/optimizer_run_driver.cpp — the by-value OptimizerRunLaunch seam, adapter forwarding.
  3. include/domain/direct_optimizer.h/.cpp — extracted DIRECT, injected cost boundary, SetCallOffset, callbacks, the split rule (TrisectPotentiallyOptimal).
  4. include/coordinator/optimizer_run_controller_core.h — typed Directive enum (incl. SymTrap), relay observation channel.
  5. The CostFunctionManager registration pattern (jta_cost_function) — the 'named registry' pattern the graph registry should mirror.
  6. test/ layout: unit/ (Catch2 + hegel PBT), lifecycle/ (QtTest), golden/, oracle/ (GPU oracle_test.cpp:286-291 the hand-rolled cost twin), qml/. The jtml_test_metric_semantics + _props targets don't exist yet (planned).
  7. CMake conventions: jtml_coordinator's explicit .cpp source list (new .cpp files must be added manually), AUTOMOC gotcha for Q_OBJECT headers in test targets.
  8. The 7 known cost-path bugs with their file:line homes (esp. distance_map_metric.cu:27 index formula; stage guard CostFunctionManager.cpp:46; DD min_dist :134; Y_dist :117-120; sym_trap tibia x->z sym_trap_function.cpp:106; Mahfouz pointer guard implant_mahfouz_metric.cu:324/446; curvature stub DIRECT_DILATION.cpp:42-43).
Report: concrete file paths, current-code shape, patterns to mirror, and anything that contradicts the synthesis's claims (esp. about the stage loop and the driver seam). Do NOT modify files.

## Acceptance Contract
Acceptance level: attested
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return concrete findings with file paths and severity when applicable

Required evidence: review-findings, residual-risks

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```