# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement plan 007 U6 for the JTML repo (cwd: /home/ajj/repo/uf/JTML). READ first: docs/plans/2026-08-12-007-refactor-qml-experimental-improvement-pass-plan.md (U6 + decision D1 property injection + the Open Questions), and the current src/app/experimental/ components (U2-U5 landed: main.qml bootstrap, StudyPanel, MlStrip, RunBar, ViewportPanel, PosesDialog, PosesTable (virtualized ListView), PoseCell, SettingsPanel, Theme).

U6 GOAL: the view layer gets automated Qt Quick Test coverage — headless (offscreen, no VTK), via a test/qml/ harness — enabled by the D1 property-injection refactor.

PART A — PROPERTY INJECTION (D1, testability refactor):
Currently every component reads bridges as rootContext properties (poseBridge, settingsBridge, studyBridge, optimizerBridge, mlBridge, appBridge) — qmllint reports them as Unqualified access (the lint gate accepts that category with 'removed by U3's property injection' as the removal path; U6 is where it actually lands).
Refactor: each component declares `required property var <bridge>` (same names as today) for the bridges it reads, and main.qml passes the real bridges at the use sites. This makes qmllint clean AND lets tests inject fakes.
Component -> required properties (read each component and confirm the exact set):
- StudyPanel: appBridge, studyBridge, optimizerBridge
- MlStrip: mlBridge, studyBridge, optimizerBridge
- RunBar: optimizerBridge
- ViewportPanel: appBridge, studyBridge (it owns the scene glue; the QmlVtkRenderer stays; NOTE: ViewportPanel is instantiated in main.qml AND is NOT part of the headless test set — it must still load in the app only)
- PosesDialog: poseBridge, studyBridge, optimizerBridge
- PosesTable: poseBridge, studyBridge, optimizerBridge
- PoseCell: poseBridge, studyBridge
- SettingsPanel: settingsBridge
main.qml passes every bridge into every component use site explicitly. Where a component ALSO uses the bridges in its inner inline components (SettingsPanel's RangeField commit closures use settingsBridge via the component scope), the injected property must be in scope for the inline components (declare on the root item; inline components inherit the file scope).
IMPORTANT correctness check: after injection, grep for any remaining bare bridge-name references that would now be undefined (e.g. in Connections targets inside components — Connections { target: studyBridge } must become target: root.studyBridge or the injected property). The app MUST still launch and behave identically.

PART B — THE test/qml/ HARNESS:
Create test/qml/ with:
- main.cpp — quick_test_main (Qt6::QuickTest) loading the tst files from the qrc.
- tests.qrc — embeds the tst files + the components under test + a test qmldir. The components are executable-backed (no qt_add_qml_module), so tst files import components via RELATIVE DIRECTORY import (Form 2 per the qt-qml-test source-import rules) resolved inside the qrc; singleton Theme needs the qmldir entry.
- FakeBridges.qml — test-only QObject types (or plain QtObject instances) exposing the exact property/signal surface each component reads: poseBridge (rowCount, tableModel — needs a real QAbstractListModel for the table... if PoseTableModel is C++-only, either embed a tiny C++ fake model in main.cpp or use a QML ListModel with the same role names; decide by reading PoseBridge.h role names), settingsBridge (the full field surface), studyBridge (frameCount, currentFrame, primaryModelIndex, selectedModels, frameListModel, modelListModel...), optimizerBridge (running, canRun, progress, stageText, costCalls, currentMinimum), mlBridge (segmentFemPt, hasSegmentModel, ...), appBridge (frameCount, modelCount).

tst files (one per component, per the qt-qml-test skill rules — TestCase, SignalSpy only for source-declared signals, tryCompare after mouse events, no-op test functions forbidden, singleton tests restore state):
- tst_Theme.qml — tokens exist, TypeScale monotonic + pinned values (12/14/16/18), contrast pairs >= 4.5:1 for the used text/bg pairs (hardcode the ratios from Theme's comment), singleton access.
- tst_PoseCell.qml — THE data-integrity pins (C1/C2 from U3): commit lands on the captured (frame, model, axis) even if selection changes mid-edit; failed commit reverts display AND keeps the binding alive (type again after failed commit works); storedValue change re-renders text; validation message display.
- tst_SettingsPanel.qml — field bindings mirror the fake bridge; SpinBox x100 scale commits 2-decimal doubles; dirty badge states; Save/Reset; dilation-field enablement per stage.
- tst_PosesTable.qml — U5's pins: with a fake 500-row model only visible rows instantiated (count via objectName/children on the ListView); edit row 3 -> scroll away mid-edit (commit-on-pool) -> scroll back -> typed value committed to row 3, no cross-row write; failed commit on a recycled cell reverts + stays editable; Loader gate (table not built while dialog closed); model rows count.
- tst_StudyFlows.qml — panel-level with fakes: load-ordering guards (calibration-first messages for Images and Models via the message dialog), models-first-then-images merge rule, replace-confirm Yes/No, ML degradation enabled-binding matrix (incl. Estimate-requires-segment), run-lock matrix (Black-sil./Fem/Tib/mode toggles/rows disabled during run), keyboard contract (frame list onCurrentIndexChanged -> setCurrentFrame; Space/Enter model toggle; dataset-swap no -1 write), dirty-close guard.

PART C — CMake registration (additive, in test/CMakeLists.txt):
- block-scoped find_package(Qt6 COMPONENTS QuickTest) (precedent: the render-smoke block);
- add_executable(jtml_test_qml_view ...) with main.cpp + tests.qrc (+ a C++ fake model TU if needed);
- link Qt6::QuickTest + Qt6::Quick + Qt6::Qml + Qt6::QuickControls2 (NOT VTK, NOT the bridges — the fakes replace them; do NOT link jtml_coordinator or jtml_compute — this must stay headless and VTK-free);
- add_test jtml.qml_view LABELS headless TIMEOUT 300 WORKING_DIRECTORY repo root, ENVIRONMENT QT_QPA_PLATFORM=offscreen QT_QUICK_CONTROLS_STYLE=Material (the app pins Material Dark — the tests must exercise the same style);
- rpath recipe per the existing test targets.

CONSTRAINTS:
- Scope: src/app/experimental/** QML (injection) + test/qml/** (new) + test/CMakeLists.txt (additive). NO changes to bridge .h/.cpp, backend, oracle, or existing tests.
- The test target MUST NOT instantiate QmlVtkRenderer/VTK. If a tst file imports ViewportPanel, that is forbidden — viewport stays untested (it is the oracle/render-smoke territory).
- Do not run git/jj. Do not stage/commit.
- Verification: pixi run build; pixi run test (must stay 47/47 + the new jtml.qml_view green); run the lint gate (injection should REMOVE most unqualified warnings — the gate's accepted set stays but the count should drop dramatically; report the before/after warning count); report the tst pass/fail counts from ctest -R jtml.qml_view --output-on-failure.
- If a test cannot be written as specified because the fake surface is incomplete, make the minimal fake addition and document it.
- Report: injection refactor summary (per component), the tst files + scenario coverage table, CMake registration, test results (counts), lint before/after, residual risks.

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