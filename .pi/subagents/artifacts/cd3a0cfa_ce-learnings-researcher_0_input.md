# Task for ce-learnings-researcher

Planning context: implementing the 'graph container' in JTML (Qt6/CUDA C++ knee-implant registration app) — data-driven optimizer stage sequences (StageScript) over the existing Optimize() loop in src/coordinator/optimizer_manager.cpp, preserving behavior bit-identically (R13 pin-first doctrine); plus foundation work: Tier-0 characterization pins for 7 cost-path bugs, one oracle re-baseline, the z-profile probe and multi-stage oracle test instruments. Search docs/solutions/ (categories: build-errors, conventions, logic-errors, test-failures, tooling-decisions, ui-bugs) for applicable learnings: headless Qt/CUDA testing, CMake AUTOMOC + explicit source-list gotchas, oracle/golden testing, GPU metric or cost-function work, pin-first characterization, stage-loop/manager patterns, layer purity, re-baseline or baseline.json conventions. Report: the specific solution docs (paths + YAML frontmatter) with the one-line lesson each, prioritized by relevance to stage-loop refactoring and headless test seams. Do NOT modify files.

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