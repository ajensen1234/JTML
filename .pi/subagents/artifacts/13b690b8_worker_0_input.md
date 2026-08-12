# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
YOU ARE AGENT B — R3: componentize the toolbar + load flows so the harness can pin them.

CONTEXT: main.qml (~517 lines) is the composition root; the toolbar Rectangle (with 4 clusters: load actions Calibration/Images/Models, calibration status label, Interact label + Camera/Model toggle buttons, Optimizer Settings…/Poses… openers + the shell dirty pill) lives inline at roughly lines 250-470, and the pickImages()/pickModels() functions at the top use fileDialogBridge + studyBridge + the root showMessage() + replaceDialog.

YOUR WORK:
a. Create src/app/experimental/Toolbar.qml:
   - The toolbar Rectangle (Theme.panel, radius 4, Layout.fillWidth, preferredHeight from the row's implicitHeight + 2*Theme.spacingSm, clip: true) with the four clusters — move the EXACT current cluster content from main.qml (buttons, labels, ButtonGroup with Camera/Model, the shell dirty pill).
   - Injected required properties: studyBridge, optimizerBridge, fileDialogBridge (var).
   - pickImages()/pickModels() move in (with the calibration-first guard + the frameCount>0 replace check) — they call fileDialogBridge.getOpenFileNames(title, filter, "", "images"/"models") and, instead of calling root functions directly, emit:
       signal showMessageRequested(string title, string message)
       signal replaceRequested(var paths)   // pendingPaths for the replace dialog
       signal calibrationRequested()        // open the calibration FileDialog
       signal settingsRequested()
       signal posesRequested()
     (the load buttons keep their current enabled bindings incl. the run lock).
   - Register in qmldir + resources.qrc.
b. main.qml: replace the inline toolbar with Toolbar { studyBridge: root.studyBridgeRef; optimizerBridge: root.optimizerBridgeRef; fileDialogBridge: fileDialogBridge; onShowMessageRequested: (t, m) => showMessage(t, m); onReplaceRequested: (p) => { replaceDialog.pendingPaths = p; replaceDialog.open() }; onCalibrationRequested: calibrationFileDialog.open(); onSettingsRequested: settingsDialog.open(); onPosesRequested: poseDialog.open() } — keep the dialog ids root-owned. This should shrink main.qml well below the plan's ~300-line criterion.
c. ALSO in main.qml (same file, your ownership):
   - The discard-dialog Escape loop fix (review finding 12): the discardDialog (the 'Discard unsaved changes?' confirm) must NOT be dismissible by Escape — remove CloseOnEscape from its closePolicy (buttons only) so Escape cannot re-arm the dirty-guard cycle.
   - Wire the discard-Yes path to call poseDialog.prepareDiscard() BEFORE closing (Agent A is adding that API to PosesDialog — call it defensively: if (poseDialog.prepareDiscard) poseDialog.prepareDiscard()).
d. tst_StudyFlows.qml additions (the harness imports components via qrc — add Toolbar.qml to tests.qrc + components-qmldir):
   - test_toolbarCalibrationFirstMessage: Toolbar with fakes (studyBridge.hasCalibration=false, frameCount=0), click the Images button (findChild objectName — give the buttons objectNames if missing), assert showMessageRequested fired with the 'Load Calibration First!' message (SignalSpy).
   - test_toolbarReplaceRequested: hasCalibration=true, frameCount=5, click Images with picker paths — the picker is fileDialogBridge.getOpenFileNames... the fake bridge: pass a fake object with getOpenFileNames returning a path list (a QtObject with a function). Assert replaceRequested fired with the paths.
   - test_toolbarModelsAfterCalibration: hasCalibration=true → Models button → studyBridge.loadModels called (fake log) — no message.
   - test_toolbarRunLock: optimizerBridge.running=true → Camera/Model toggle buttons + load buttons disabled (the review's I4 gap: Camera/Model run-lock).
   - test_shellDirtyPill: settingsBridge.dirty or poseBridge.dirty → the pill shows 'unsaved' (check via text or a property).
e. Follow the qt-qml skill (load /home/ajj/.pi/agent/skills/qt-qml/SKILL.md): Layout.* sizing, Theme tokens only, no anchors+Layout mixing, single-owner Connections (the toolbar emits signals — no Connections to bridges).
f. Report: files changed, main.qml final line count, pins added, qmllint results, deviations.
Repo: /home/ajj/repo/uf/JTML (Qt 6.7.2 QML app; experimental front-end under src/app/experimental/). This is a fix round after a 10-persona code review of plan 007.
YOU MAY LOAD THE QT SKILLS for guidance: /home/ajj/.pi/agent/skills/qt-qml/SKILL.md, /home/ajj/.pi/agent/skills/qt-qml-test/SKILL.md (+ its references/ for the 47 testing rules), /home/ajj/.pi/agent/skills/qt-qml-review/SKILL.md. Apply them.
CONSTRAINTS: EDIT-ONLY — do NOT run pixi run build / pixi run test / cmake / ninja (the orchestrator builds and tests after the parallel wave lands; concurrent builds on the shared .build dir race). You MAY run $CONDA_PREFIX/lib/qt6/bin/qmllint (or .pixi/envs/default/lib/qt6/bin/qmllint) --json on your changed QML for syntax checking (the bare qmllint on PATH is Qt 5.15.8 — never use it). No git/jj commands, no staging, no commits.
Test conventions (read existing files first): test/qml/ harness — quick_test_main, tests.qrc aliases REAL component sources, fakes (FakePoseBridge/FakeStudyBridge...) injected via required properties, QT_QPA_PLATFORM=offscreen + QT_QUICK_CONTROLS_STYLE=Material. Run pattern the orchestrator uses: QT_QPA_PLATFORM=offscreen QT_QUICK_CONTROLS_STYLE=Material .build/bin/jtml_test_qml_view tst_<Name> (the existing binary — but do not rebuild it).

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