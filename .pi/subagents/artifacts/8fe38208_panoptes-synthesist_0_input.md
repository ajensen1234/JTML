# Task for panoptes-synthesist

Panoptes synthesis pass (ROUND 3 wave 1). RUN: i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive.
Four angles now carry '## Round 3' sections grounded in Zotero primary texts: 01-direct-variants (GLh formula verified from the paper Eqs 3-6 with selection-time d/sqrt(n) sharpening; gb numbers vs Gablonsky; Jones 1993 epsilon re-check; DGO battery specifics; DIRECT-JTA convergence tradeoff), 02-cost-path-foundation (Mahfouz normalization RESOLVED from the paper; three-way dilation contradiction RESOLVED with Flood constants + recommendation; Canny divergence flagged; Jensen sym-trap findings), 03-measurement-apparatus (z-leaf budget evidence from Flood; Yamazaki depth-position remedies; SOTA benchmarks from Burton/Jensen/Arulampalam; perturbation-suite mirroring), 04-backend-architecture (StageScript fidelity vs Flood's 3-branch staging; convergence-tradeoff note; Cut B pin strategy; torch boundary vs differentiable forward projector).
Read ALL six angle files (the four with Round 3 sections + 05/06 from rounds 1-2) + the existing synthesis.org, then UPDATE synthesis.org: add a 'Round 3 Updates (wave 1)' section recording the deltas (esp. the RESOLVED contradictions: Mahfouz normalization, dilation three-way, GLh formula verification), fold any verdict-changing findings into Key Findings, and update the dependency-ordered execution path (e.g. the dilation-schedule ablation now tests lineage configs; the multi-stage oracle's z-stage budget is first-class). Keep the existing structure; record deltas.
Write via the sanctioned artifact path to /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive/synthesis.org (overwrite the whole file). Report the file path + a 8-line delta summary.

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