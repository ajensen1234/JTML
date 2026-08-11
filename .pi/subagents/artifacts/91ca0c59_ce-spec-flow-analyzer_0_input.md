# Task for ce-spec-flow-analyzer

Analyze user-flow completeness and edge cases for a planned improvement pass over the experimental QML front-end of a Qt6.7 desktop app (src/app/experimental/).
The app: main.qml (~700 lines) = toolbar (calibration/images/models buttons + interaction-mode toggle + dialog openers),
left panel (frame ListView + model ListView with delegate selection + ML strip with .pt pickers/segment/estimate buttons),
center QmlVtkRenderer viewport, bottom run bar (Run/Stop + progress).
Dialogs: FileDialogs (calibration/images/models/.pt/pose/kinematics), replace-dataset confirm, message dialog,
settings Dialog (SettingsPanel.qml form), poses Dialog (PoseCell.qml table, Repeater over all frames, 6 editable cells/row,
dirty badge, copy-prev/next, save/load). Bridges (C++ QObjects exposed as context properties) hold all logic; QML is thin glue.
Planned changes: extract main.qml panels into component files; property injection for testability;
Theme token/typography compliance; keyboard nav + accessibility; pose-table virtualization (Repeater -> ListView);
PoseCell binding-fix; Qt Quick Test coverage; qmlprofiler pass.
Identify: (1) state-transition gaps in the view layer (e.g. dialogs open during optimizer run, dialog interplay,
run-state + dialog openers, empty states), (2) edge cases the new component extraction could break
(id/scope references, anchors, Qt.callLater timing, Connections targets),
(3) user-flow completeness gaps worth covering in tests (load flow ordering, ML degradation, pose edit validation, replace-dataset confirm),
(4) handoff gaps between QML view and bridge signals.
Report concrete findings with file/line references where possible; keep it tight and actionable.

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