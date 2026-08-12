# Task for worker

You are reviving a previous subagent conversation.

Original run: 895e4355
Original agent: worker
Original session file: /home/ajj/.pi/agent/sessions/--home-ajj-repo-uf-JTML--/2026-08-12T14-48-28-894Z_019ff672-39de-7cf4-8437-a02bdf255d86.jsonl

Use the stored session context as background. Answer the orchestrator's follow-up below. Do not assume the original child process is still alive.

Follow-up:
Resume your U6 work. Good progress: the harness builds and 31/35 tests pass. Four failures remain — fix them efficiently and move fast:
1. qml_view::SettingsPanel::test_scaleCommitsTwoDecimals — trunkRangeX compare fails (you were mid-debug).
2. qml_view::SettingsPanel::test_saveAndResetReachBridge — saveCalls property compare fails.
3. qml_view::PosesTable::test_commitOnPoolMidEdit — commit-on-pool pin: typed value did not land on row 3.
4. qml_view::PosesTable::test_rejectedCommitSurvivesRecycle — verify() false: binding-alive pin.

Speed directives:
- Run the failing tests in isolation first: QT_QPA_PLATFORM=offscreen QT_QUICK_CONTROLS_STYLE=Material .build/bin/jtml_test_qml_view <TestCase>::<testFunction> — iterate on ONE failure at a time.
- Distinguish test-harness bugs from product bugs: if the component behavior is correct and the TEST is wrong (wrong fake surface, wrong event simulation), fix the test. If the component is wrong, fix the component. Either is fine — the pins must pass.
- SettingsPanel SpinBox scale test hint: the x100 SpinBox commits value/100 on onValueModified; if the fake's property is a JS property on a QtObject, binding updates do not notify — the fake may need signal-based properties or the assertion needs tryCompare.
- PosesTable recycle tests hint: verify the fake model emits proper modelReset/dataChanged; ListView pooling requires the model to signal row removal. If simulating scroll+pool is flaky, drive the pool directly via positionViewAtIndex to force eviction.
- Do not rewrite passing tests. Do not gold-plate. Get 35/35 green, run the full headless suite (pixi run test) + the lint gate, then write your acceptance report.
Constraints unchanged: no git/jj, no staging/committing; scope = src/app/experimental/** QML + test/qml/** + test/CMakeLists.txt; no bridge/C++ changes.

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope
- criterion-2: Return evidence sufficient for an independent acceptance review

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

Review gate: required by reviewer.

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
    },
    {
      "id": "criterion-2",
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