# Task for ce-repo-research-analyst

Scope: technology, architecture, patterns. Planning context: we are planning an improvement pass over the experimental QML front-end of this repo (JTML, a Qt6 + VTK9.3 + CUDA knee-implant registration app). The app lives in src/app/experimental/ (jtml_experimental target): QML shell main.qml (~700 lines: toolbar, 8 dialogs, 2 ListViews, ML strip, progress), SettingsPanel.qml, PoseCell.qml, Theme.qml (pragma Singleton + qmldir), renderer.qml (smoke scene); C++ side: AppBridge hub owning StudyBridge/SettingsBridge/OptimizerBridge/MlBridge/PoseBridge + FileDialogBridge + QmlVtkRenderer (QQuickVTKItem subclass, render-thread contract) + ExperimentalScene/ExperimentalSession; FrameListModel/ModelListModel direct-compiled from src/view. Bridges are exposed to QML via rootContext()->setContextProperty in main.cpp. CMake uses add_executable + resources.qrc (no qt_add_qml_module). Planned work tracks: (1) structured QML review + best-practice fixes (file structure, bindings, layouts, Theme token compliance), (2) UI/UX audit+polish (typography scale, keyboard nav, accessibility), (3) Qt Quick Test coverage (tst_*.qml + CMake/ctest wiring; repo rule: headless tests must not touch GPU/VTK/widgets), (4) qmlprofiler performance profiling of the 2D chrome. Research: repo structure and conventions, how QML files are built/shipped, existing test registration patterns in test/CMakeLists.txt (labels, WORKING_DIRECTORY, AUTOMOC gotchas), whether Qt6::QuickTest is available in the pixi env, existing docs conventions (docs/plans, docs/solutions, docs/handoff), AGENTS.md guidance that materially affects the plan, the layered-lib rules (domain/services/coordinator/compute/view), and the QML decision record at src/view/CMakeLists.txt:6. Report concrete file paths and patterns.

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