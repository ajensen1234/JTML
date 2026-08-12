# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
YOU ARE AGENT A — R1+R2 completion: pins for the pose-table keyboard contract and the mid-edit flush mechanisms.

DESIGN ALREADY LANDED (read these files to see it):
1. src/app/experimental/PoseCell.qml: added `signal navRequested(int direction)` (encoding: -2 up, -1 left, +1 right, +2 down) + Keys.onPressed handling Left/Right/Up/Down (emits navRequested) and Escape (edited=false; resetDisplay() — reverts without committing); added `property bool suppressDestructionCommit` + Component.onDestruction { if (edited && !suppressDestructionCommit) doCommit() }.
2. src/app/experimental/PosesTable.qml delegate: each PoseCell wires onNavRequested — Left/Right calls the delegate's focusCell(axis), Up/Down calls moveToRow(frame±1, axis); the delegate has focusCell(axis) (forces focus on the sibling cell) and moveToRow(frame, axis) (positionViewAtIndex + Qt.callLater focus); added onFrameIndexChanged: flush all six cells (commitIfEditing) then resetDisplay (the in-place re-bind / instant-jump path); cells receive suppressDestructionCommit: root.suppressDestructionCommit.
3. PosesTable root: add `property bool suppressDestructionCommit: false` if not already present (check — the delegate references root.suppressDestructionCommit).

YOUR WORK:
a. PosesDialog.qml: add `function prepareDiscard() { posesTable.suppressDestructionCommit = true }` (find the PosesTable instance id; the discard-close path in main.qml will call it — the main.qml wiring is another agent's job, just expose the API).
b. tst_PoseCell.qml pins:
   - test_escapeRevertsCell: type '12.5' into a focused cell, keyClick(Qt.Key_Escape), assert the text reverts to storedValue and NO commit landed (commitLog length 0), and typing again works (binding alive).
   - test_destructionCommitsLiveEdit: type, then cell.destroy() (or set the cell's parent null? use destroy() on the temporary object), wait, assert commitLog has the entry with the captured tuple.
   - test_destructionSuppressedDoesNotCommit: type, set suppressDestructionCommit = true, destroy, assert commitLog unchanged.
c. tst_PosesTable.qml pins (the table already exists in the harness — read it):
   - test_keyboardLeftRightMovesCell: focus cell X of a visible row, keyClick(Qt.Key_Right), assert the Y cell of the same row has activeFocus (via children lookup); Key_Left returns to X.
   - test_keyboardUpDownMovesRow: focus a cell in row 1, keyClick(Qt.Key_Up), wait(100) (the Qt.callLater + position), assert a cell in row 0 with the same axis has activeFocus.
   - test_instantJumpFlushesEdit: the in-place re-bind path — edit a cell in row 3, then list.positionViewAtIndex(400, ListView.Center) (an INSTANT jump, not gradual), wait(100), assert the typed value committed to row 3 (the onFrameIndexChanged flush) — check the existing recycle test's comment for the exact list/fake setup.
d. Existing pins may break from the new mechanisms (the onFrameIndexChanged flush + destruction commit are behavior changes): read tst_PoseCell.qml + tst_PosesTable.qml fully, adjust any pin that now behaves differently, and make sure the comments match the landed mechanisms. If a pin's intent conflicts with the new behavior, prefer the new behavior (it is the reviewed design) and update the test.
e. Report: files changed, pins added/changed, qmllint results, any design deviations with reasons, anything the orchestrator must verify at build time.
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