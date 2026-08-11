# Task for panoptes-librarian-source

ROUND 2 (deepening) of the optimizer-deep-dive panoptes run (runId i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive).
Read your EXISTING angle file + the workspace synthesis.org FIRST, then APPEND a '## Round 2' section via the panoptes_write_artifact tool: call it with runId=i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, path=angles/NN-<slug>.org, content=<your Round 2 section, with its own '## Round 2' header>, append=true. The write is the deliverable; if the write is rejected, RETURN the full Round 2 content in your output instead.
Your round-1 file stays untouched. NEVER spawn subagents. You are read-only except for that sanctioned artifact append.
ANGLE: cost-path-foundation (append to angles/02-cost-path-foundation.org)
Execution specifics round 1 left open: (1) NaN/Inf emission audit: does the production GPU silhouette cost ever emit NaN/Inf (the DIRECT-GLh trigger)? Design the finite-check (where in the pipeline it wraps, how to probe headlessly vs on GPU, expected failure modes: degenerate projection, zero denominator, empty silhouette); (2) distance-map index fix impact analysis: which cost terms read the corrupted index (distance_map_metric.cu:27), how the corruption biases the z-sensitive complement of the chamfer, expected DIRECTION of the re-baseline move (not magnitude); (3) the two backface evidence checks as executable specs: KR_right_7_fem.stl manifoldness/watertightness (tool + criterion) and the ON-vs-OFF silhouette IoU measurement protocol; (4) the Mahfouz normalization question: the 2.55 scale + host x255 + -2.67 weighting needs the original 2003 paper - flag as a Zotero pull + specify the kernel-as-spec fallback decision rule; (5) the curvature de-scope cut: exact removal surface (the leaky AllocateCurvatureHausdorfScore from initializeDIRECT_DILATION) + de-scoped banner wording + what happens to the heatmap upload path in OptimizerManager::Initialize; (6) the Tier-0 metric-semantics test inventory: per metric (chamfer overlap/normalization, Mahfouz ratios, distance-map, IoU/L1) name the CPU reference property test + hegel PBT twin targets. LOCAL CONTEXT: the 7 bugs verified (DD_NEW_POLE_CONSTRAINT.cpp:134 + :120-122; sym_trap_function.cpp:104-108; CostFunctionManager.cpp:46; distance_map_metric.cu:27; implant_mahfouz_metric.cu:324/446; curvature_hausdorf_metric.cu stub); oracle test/oracle/oracle_test.cpp (DIRECT_DILATION, backface OFF, budget 3000); baseline.json gates; test_cost_function.cpp covers only the parameter registry today; hegel PBT conventions in docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md.

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