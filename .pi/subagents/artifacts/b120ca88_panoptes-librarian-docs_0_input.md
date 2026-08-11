# Task for panoptes-librarian-docs

ROUND 2 (deepening) of the optimizer-deep-dive panoptes run (runId i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive).
Read your EXISTING angle file + the workspace synthesis.org FIRST, then APPEND a '## Round 2' section via the panoptes_write_artifact tool: call it with runId=i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, path=angles/NN-<slug>.org, content=<your Round 2 section, with its own '## Round 2' header>, append=true. The write is the deliverable; if the write is rejected, RETURN the full Round 2 content in your output instead.
Your round-1 file stays untouched. NEVER spawn subagents. You are read-only except for that sanctioned artifact append.
ANGLE: direct-variants (append to angles/01-direct-variants.org)
Resolve what round 1 left dangling: (1) gb phase-switch parameters: the implementation reads 5 non-improving iterations + security every 5th; the 2022 paper cites published Gb-BIRECT/DISIMPL recommendations [10,28] - verify which numbers are canonical and the switch semantics; (2) the split tie-break contradiction (repo: largest denormalized side vs 2022 study: least-split-so-far) - decide with evidence which rule 1-DTC-GL-gb uses and what the analytic battery must probe to distinguish them; (3) concretize the analytic battery: exact DIRECTGOLib n=5/6 function set + GKLS classes + crossover budgets (100/500/1000/4000) + termination (percent-error pe<=1e-2, f*=0 subset reported separately) with sources; (4) the GLh integration point: where the finite-check wraps the injected cost, what NaN/Inf does to the hull today (silent corruption analysis), the surrogate's exact normalized-distance formula vs the repo's DenormalizeFromCenter (direct_optimizer.cpp:238-247); (5) the 1-DTDV boundary probe: battery configuration that triggers a range-edge optimum + the go/no-go criterion; (6) contradiction 2 head-on: does the variant choice interact with the z-sensitivity ranking (chamfer-heavy >= Mahfouz >= pure overlap), or orthogonal? LOCAL CONTEXT: DirectOptimizer include/domain/direct_optimizer.h + src/domain/direct_optimizer.cpp (ConvexHull L100-148 epsilon=0, TrisectPotentiallyOptimal L151-200); budgets baseline.json trunk 20000 / branch 2x5000 / leaf 5000 cumulative 20/25/30/35k; papers at papers/ (2022 study arXiv:2109.14912; DIRECT-GLh via thesis ch.5 + DIRECTGO arXiv:2107.02205 section 2.3). Conventions: R13 pin-first, headless default.

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope
- criterion-2: Return evidence sufficient for an independent acceptance review

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

Review gate: required by reviewer.

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
    },
    {
      "id": "criterion-2",
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