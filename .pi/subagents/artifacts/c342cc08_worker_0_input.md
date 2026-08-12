# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
You are Agent 3: Component Loading & Lifecycle. CHECK: Loader patterns; Connections with dynamic targets; context-property usage in C++ integration (main.cpp) — ownership risks; the Qt.callLater datasetChanged deferral (main.qml ~421-424) vs the list-model swap; dialog lifetime (FileDialogs as Window children, Dialogs with contentItem components); image loading. READ the 5 QML files in src/app/experimental/ (main.qml ~1020 lines, SettingsPanel.qml, PoseCell.qml, Theme.qml, renderer.qml) plus the bridge headers (StudyBridge.h, SettingsBridge.h, OptimizerBridge.h, MlBridge.h, PoseBridge.h, AppBridge.h) as needed to trace symbols.

CONTEXT: this is a Qt 6.7.2 QML app; bridges are QObjects exposed via rootContext()->setContextProperty in main.cpp (that is why qmllint reports Unqualified access everywhere — known, deliberate, and scheduled for property-injection). The app is experimental; findings feed an improvement plan. Report findings at confidence >80 only; 60-79 goes to investigation targets (max 10). Do NOT report findings already covered by this lint context.

LINT SCRIPT FINDINGS (qt_qml_lint.py, deterministic):
PoseCell.qml:37 BND-2 imperative '=' on text kills the storedValue binding (known C1).
SettingsPanel.qml: PRF-1 transparent Rectangle (line 24), PRF-3 clip:true (59), BND-1 property var commit (71/105/133/135), JS-1 var (89), ~20 ORD-1 ordering nits.
Theme.qml:7 STY-1 no id: root.
main.qml: IMP-3 plain QtQuick.Controls import (3), IMP-1 QtQuick.Window redundant (5), IMP-2 versioned import (8), BND-1 var (123), BND-2 imperative '=' on messageDialog.title (152), JS-1 var (161), LAY-2 width in RowLayout children (275,282,289,296,303,310,317), PRF-3 clip (347,652,689), PRF-1 transparent Rectangles (570,655,692), DEL-1 model.roleName without required property (~20 sites in the pose-table delegate), JS-2 loose equality (695), STY-3 anchors dot notation (940).
renderer.qml: IMP-1 (2), IMP-2 (3).
QML LINT (Qt6 qmllint 6.7.2): main.qml 8x 'Detected width on an item that is managed by a layout' (275-366); ~150 'Unqualified access' warnings from rootContext property globals (studyBridge/settingsBridge/optimizerBridge/mlBridge/poseBridge/appBridge/fileDialogBridge); 'QmlVtkRenderer was not found' at 909/916 (C++-registered type, import-path gap, not a bug); SettingsPanel 3x 'Property commit is a variant property' + unqualified access; renderer.qml anchors unresolved in smoke scene; Theme 1 unqualified. Env noise: Qt5+Qt6 module ambiguity on QtQuick.Dialogs import.

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