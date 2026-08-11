# Task for panoptes-synthesist

Panoptes synthesis final pass (ROUND 2 wave 2). RUN: i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive.
ALL SIX angles now carry '## Round 2' sections. Wave 2 added: 05-compute-perf (instrumentation protocol with nvtx3/cudaEvent placements + nsys recipe; Cut 1 full spec with a fresh finding — the ms/call meter has a 1000x unit error on Linux (no CLOCKS_PER_SEC division, glibc clock() returns microseconds, labeled ms/call); Cut 2 double-buffered eval-state design; Cut 3 decision procedure; Cut 4 event-based meter; the 4-gate measurement protocol) and 06-differentiable-polish (executable MVP spec: precondition gate metrics, forward-difference gradient with 6 renders/step, analytic projection Jacobian + gradient-of-distance-map kernel, the ablation arm arbitrating the 12x-vs-1.5-3x contradiction, Zotero pull list with per-paper value).
Read ALL six angle files (with their Round 2 sections) + the existing synthesis.org (which already has the wave-1 Round 2 Updates section), then UPDATE synthesis.org: extend the 'Round 2 Updates' section with wave-2 deltas (the meter unit error is a new high-value finding — fold it into Key Findings; the polish MVP preconditions; any contradiction changes), and finalize the dependency-ordered execution path to reflect the round-2 state. Keep the existing structure; record deltas.
Write via the sanctioned artifact path to /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive/synthesis.org (overwrite the whole file with the updated content). Report the file path + a 8-line final delta summary.

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