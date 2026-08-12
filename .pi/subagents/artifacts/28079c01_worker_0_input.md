# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
YOU ARE AGENT C — R4 batch (C++/CMake/docs; NO main.qml, NO PosesDialog.qml, NO test/qml tst files — other agents own those).

YOUR WORK:
a. src/app/experimental/PoseBridge.cpp: role-filtered dataChanged (review finding 14): notifyCellChanged currently emits dataChanged(index(frame,0), index(frame,0)) with NO roles — all 6 cells re-read per single-axis edit. Qt 6 supports role-filtered dataChanged: emit with the single role for the edited axis. Check the table model's roleNames (frameIndex/x/y/z/xa/ya/za) and map axis→role. One-line, behavior-preserving; keep the existing tests green (test/unit/experimental_pose_bridge_test.cpp must not need changes — check!).
b. src/app/experimental/AppBridge.cpp: ML-estimate table refresh (finding 15): MlBridge's successful estimate writes storage directly (SavePose) but the D3 refresh triggers (run terminal state, viewerPoseApplied) don't include it — an open pose table shows the pre-estimate value. Wire: when mlBridge reports a successful estimate (find the signal — onPoseEstimated or similar), ALSO refresh the pose table — WITHOUT clearing the seed (refreshTable only emits modelReset per the api-contract review — verify it doesn't clear the seed; if it does, use a separate refresh path). Read MlBridge.h/AppBridge.cpp wiring first.
c. test/CMakeLists.txt: the jtml.qml_lint gate silently unregisters when no Qt6 qmllint binary is found (finding 9): add message(WARNING ...) at configure when QML_LINT_QT6 ends up empty, so the gate's disappearance is visible.
d. src/app/experimental/ViewportPanel.qml: remove the dead I-03 guard (Component.onCompleted with `if (!viewportItem)` — unreachable: the id always exists) — finding 17. Keep a comment noting the failure mode is silently unobservable (the renderer reports no failure).
e. FileDialogBridge headless coverage (finding 10): make rememberDir/lastDir/mruUrls public (non-invokable — just move them to public in the header; they're trivial accessors) and add test/unit/experimental_file_dialog_test.cpp (Catch2, QtCore-only — FileDialogBridge is a QObject with QSettings; the test redirects XDG_CONFIG_HOME to a temp dir via qputenv before constructing, then pins: rememberDir→lastDir round-trip per purpose, purpose-bucket isolation (images vs models), MRU order + dedupe + kMaxMruDirs cap (5)). Register the target in test/CMakeLists.txt following the jtml.experimental_settings test recipe (block-scoped, rpath, LABELS headless) — NOTE: FileDialogBridge.h has Q_OBJECT (AUTOMOC: list the header in the add_executable sources). Link Qt6::Core + Qt6::Widgets (QFileDialog is in the .cpp — linking Widgets pulls it; the test only exercises the QSettings helpers, which don't exec dialogs).
f. docs: handoff-2026-08-12-qml-experimental-improvement.md — fix the stale line-count claim ('main.qml 1020 → ~230 lines' → the actual post-U4/final figure; the toolbar extraction this round shrinks it further — use the orchestrator's final number if unknown, else write '~300 (post-toolbar-extraction)' and note the review) — actually: correct it to the true final value after the toolbar extraction lands; if you can't know it, qualify it. And AGENTS.md: add test/qml/ to the Test suite layout enumeration (Qt Quick Test harness + jtml.qml_lint gate) beside the existing QtTest/Catch2 bullet — one or two lines, matching the file's style.
g. Report: files changed, test target summary, deviations.
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