# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement plan 007 U2 for the JTML repo (cwd: /home/ajj/repo/uf/JTML). Plan: docs/plans/2026-08-12-007-refactor-qml-experimental-improvement-pass-plan.md — READ the plan's U2 section, the High-Level Technical Design (target file structure + component wiring contract), and the Key Technical Decisions D11/D1 first. Also read docs/reviews/2026-08-12-qml-experimental-review.md findings D-01, D-03, I-03, I-04 (they are part of this unit's scope).

U2 GOAL: Theme tokens + component extraction. main.qml (~1020 lines) shrinks to a bootstrap; every panel/dialog is a file with single responsibility; all colors and type sizes come from Theme.qml; Layout.* sizing and import rules clean.

SCOPE HARD LIMITS (owner constraint, absolute): modify ONLY files under src/app/experimental/** (QML, qrc, qmldir, CMakeLists) and add a qmllint ctest registration to test/CMakeLists.txt. Do NOT touch src/view, backend seams (domain/services/coordinator/compute), the widgets app, oracle/golden fixtures, or any bridge .h/.cpp files. This unit is QML-only: no C++ changes.

WORK (in this order):
1. Theme.qml: keep existing palette tokens; ADD a TypeScale block with pinned roles: caption 12, label 14, body 16, h2 18 (major second 1.125 from base 16). Add id: root at top (STY-1).
2. Token re-point BEFORE extraction (D11): replace every hardcoded color in SettingsPanel.qml and main.qml with Theme tokens (add the theme import to SettingsPanel.qml — it currently imports only QtQuick/Layouts/Controls and hardcodes #cfd3da/#8b929c where Theme.fg/Theme.fgMuted exist). Replace the duplicated dirty-badge hexes (#e5b567/#3a4a3d/#2a2118/#8fbf96) with Theme tokens. ALSO fix the invisible dirty-badge pill (review D-01): the badge Rectangle collapses to 0 width in its RowLayout (implicitWidth 0) — give it Layout.preferredWidth derived from its label (e.g. Math.max(label.implicitWidth + 12, 28)).
3. Extract components (create new files in src/app/experimental/):
   - StudyPanel.qml — left column: frame ListView + model ListView + selection label (the two lists + their delegate logic). Convert delegate root-id references (frameList.width, modelList.width, frameList.currentIndex) to ListView.view + required properties (review I-04).
   - MlStrip.qml — the ML controls strip (pick buttons + path labels + Segment/Estimate + Black sil./implant/view rows + estimate/status labels).
   - RunBar.qml — bottom bar: Run/Stop + ProgressBar + stage/calls/min labels.
   - ViewportPanel.qml — center region: the QmlVtkRenderer + placeholder + debug readout + the scene-glue Connections (studyBridge scene relays + selection + viewport-target block). Exposes property alias viewport. Add a Component.onCompleted guard that warns (console.warn) if the renderer failed to instantiate (review I-03).
   - PosesTable.qml — the pose table (ScrollView + column header RowLayout + the Repeater body + empty-state labels). NOTE: do NOT virtualize in this unit (that is U5) — move the existing Repeater-based table as-is.
   - PosesDialog.qml — the Poses dialog shell: header row (model context + dirty badge), action buttons row (copy/save/load), the PosesTable, validation message label. The dialog stays root-owned OR self-contained per the wiring contract — it must expose open()/close() for the Run button.
4. main.qml shrinks to bootstrap: Window + Material settings + the 8 FileDialogs + replaceDialog + messageDialog + the remaining glue Connections (studyBridge datasetChanged/message, optimizerBridge, mlBridge, poseBridge) + toolbar RowLayout + the region composition (StudyPanel | ViewportPanel) + RunBar + settingsDialog. Run button closes both dialogs (keep that behavior). Target main.qml <= ~300 lines.
5. Wiring contract (from the plan HLTD — follow exactly): single-owner Connections per bridge signal surface (studyBridge's two blocks split: datasetChanged+message stays root/StudyPanel; scene relays+selection+viewport-target move to ViewportPanel; optimizer/ml/pose relays stay root-owned glue). The Qt.callLater deferral in onDatasetChanged MUST survive extraction (it outlasts the list-model swap in clearDataset). Every new component file is registered in qmldir (engine-root documents cannot use inline components) and listed in resources.qrc.
6. Layout + import fixes: items directly inside RowLayout/GridLayout size via Layout.* only (the dialog column-header labels at old main.qml:275-317, the pose-row labels, PoseCell width:78/height:26 -> implicitWidth/implicitHeight — note PoseCell.qml is a separate file, fix it too). Transparent spacer Rectangles -> Item (main.qml:570,655,692 + SettingsPanel:24). SettingsPanel content width: bind to the ScrollView's availableWidth instead of root.width - 18 (review D-03). Drop redundant QtQuick.Window imports (main.qml, renderer.qml) IF Window resolves from QtQuick on Qt 6.7 (verify — if the build breaks without it, keep it and comment). Keep the deliberate Material style import with a comment. Versioned import 'jtml.experimental 1.0' — keep (it is the C++ module registration, versioned by design).
7. test/CMakeLists.txt: add a headless qmllint gate ctest (name jtml.qml_lint, LABELS headless, WORKING_DIRECTORY repo root) running $CONDA_PREFIX/lib/qt6/bin/qmllint (the Qt6 binary — NEVER the bare qmllint name which is Qt 5.15.8 on PATH) --json over the app's QML files; fail on non-empty real warnings EXCEPT known-accepted categories (document them in a comment: Unqualified access from context-property bridges is expected until U3's property injection; QmlVtkRenderer-not-found is the C++ registration gap; QtQuick.Dialogs ambiguous-module noise is environmental). Simplest robust form: the test script may filter accepted categories via grep.
8. Do NOT touch renderer.qml except the import cleanup in step 6 (it is the smoke scene — keep it working).

CONSTRAINTS:
- No C++ changes at all in this unit.
- Do not run git/jj commands. Do not stage or commit. The orchestrator handles jj.
- Do not run the full test suite. You may run: pixi run build (to compile-check) and the qmllint command manually to verify your gate works.
- Behavior must be identical: this is a relocation pass (no logic changes). The app must launch and behave as before.
- After finishing, run: pixi run build and verify the target compiles. Fix any build errors.
- Report: files created/modified, main.qml final line count, qmllint gate result, any deviations from this spec with reasons, and the manual-visual checklist items that need the owner's xcb display (you cannot run the GUI).

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