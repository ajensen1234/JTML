# Task for ce-learnings-researcher

Planning context: we are planning an improvement pass over the experimental QML front-end (src/app/experimental/, jtml_experimental) of a Qt6+VTK9.3+CUDA desktop app: structured QML review + best-practice fixes, UI/UX audit, Qt Quick Test coverage, and qmlprofiler profiling. Search docs/solutions/ for institutional learnings that apply: QML-specific conventions and bugs (e.g. jtml-qml-experimental-frontend-2026-08-11.md, jtml-qml-model-pose-sync-queued-functor-never-delivered-2026-08-11.md, jtml-moc-signals-section-placement-duplicate-definition-2026-08-11.md), CMake/AUTOMOC/test conventions (jtml-testability-and-cmake-conventions-2026-08-07.md), rendering/xcb learnings (jtml-rendering-runtime-xcb-qvtk-2026-08-10.md), layered-lib split conventions, and anything about testing QML, qmllint, Qt Quick Test, or UI styling. Report each relevant learning with its doc path and a one-line summary of how it constrains or informs the plan.

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