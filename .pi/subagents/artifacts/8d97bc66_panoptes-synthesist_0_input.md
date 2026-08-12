# Task for panoptes-synthesist

Panoptes synthesis FINAL pass (ROUND 3 wave 2). RUN: i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive.
ALL SIX angles now carry '## Round 3' sections grounded in Zotero primary texts. Wave 2 added: 05-compute-perf (Flood's ~3,000 evals/s GTX 970 + 145k evals/frame as the cross-era sanity baseline; Abdellah's 25-67ms volumetric DRRs vs silhouette evals ~2 orders cheaper; the fast-DRR family's techniques assessed against render_engine.cu: bindless textures, block projection, splatting — what Cut 3+ licenses; a target evals/s range for the RTX 3090 with gates) and 06-differentiable-polish (the full differentiable-rendering corpus surveyed: SoftRas coverage, edge-sampling, SDF reparameterization, DiffDRR forward-projection — cheapest gradient strategy for the hand-written CUDA stack; the Mahfouz Nelder-Mead simplex precedent re-evaluating the polish MVP; Yamazaki 2004's depth-position remedies vs the z-depth-channel NO-GO; the Zotero pull list re-ranked with in-library items marked).
Read ALL six angle files (with Round 3 sections) + the existing synthesis.org, then UPDATE synthesis.org: add a 'Round 3 Updates (wave 2)' section, fold any verdict-changing findings into Key Findings (e.g. the perf sanity baseline, the simplex-polish arm, the pull-list re-rank), finalize the dependency-ordered execution path, and mark the run's research phase complete (the run is at its final round).
Write via the sanctioned artifact path to /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive/synthesis.org (overwrite the whole file). Report the file path + a 10-line final summary covering the whole round-3 delta.

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

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