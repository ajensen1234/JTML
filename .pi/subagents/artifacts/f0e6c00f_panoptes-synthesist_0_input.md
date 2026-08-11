# Task for panoptes-synthesist

Panoptes synthesis re-run (ROUND 2 wave 1). RUN: i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive.
Four angles now carry '## Round 2' append sections: 01-direct-variants (gb phase-switch resolved, split tie-break, analytic battery concretized, GLh integration, 1-DTDV probe, z-interaction), 02-cost-path-foundation (NaN/Inf audit — key finding: DIRECT_DILATION structurally cannot emit NaN/Inf but MAHFOUZ/DD/sym_trap can; distance-map impact, backface specs, Mahfouz Zotero pull, curvature de-scope, Tier-0 inventory), 03-measurement-apparatus (perturbation suite, z-profile spec, Directive::SymTrap seam addition, ablation.json schema, noise floor), 04-backend-architecture (Cut E gate, StageScript schema, Cut B pin strategy, double-buffer seam, torch boundary, Options defaults).
Read ALL six angle files (the four with Round 2 sections + 05-compute-perf + 06-differentiable-polish from round 1) + the existing synthesis.org, then UPDATE synthesis.org: fold the Round 2 findings into the key findings and the dependency-ordered execution path, resolve or re-state the contradictions the Round 2 sections resolved (e.g. gb numbers, NaN/Inf trigger now conditional per-variant), and note what wave-2 (compute-perf + differentiable-polish round 2) will add. Keep the existing structure; add a 'Round 2 updates' section that records deltas rather than rewriting the whole document.
Write via the sanctioned artifact path to /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive/synthesis.org (overwrite the whole file with the updated content). Report the file path + a 8-line delta summary.

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