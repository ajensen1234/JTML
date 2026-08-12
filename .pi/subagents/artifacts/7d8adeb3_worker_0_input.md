# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement plan 007 U5 for the JTML repo (cwd: /home/ajj/repo/uf/JTML). READ first: docs/plans/2026-08-12-007-refactor-qml-experimental-improvement-pass-plan.md (U5 + decisions D2/D8 + the Open Questions), docs/reviews/2026-08-12-qml-experimental-review.md (D-02, D-04, D-05, D-06, I-02, I-06, I-08, I-09, and delegate findings F1-F3), and the current src/app/experimental/PosesTable.qml + PoseCell.qml + PosesDialog.qml (U3 already landed the capture-at-edit-start commit contract in PoseCell).

U5 GOAL: virtualize the pose table. Currently PosesTable.qml has Repeater-in-Column-in-ScrollView — every frame row (1 Label + 6 PoseCell TextFields) is instantiated. Replace with a ListView + reuseItems, with the commit-on-pool contract.

SCOPE: only src/app/experimental/** QML (PosesTable.qml, PoseCell.qml if needed, PosesDialog.qml if the loader gate needs it). NO C++ changes. NO test files in this unit (the test pins land in U6's test/qml/ harness per the plan — the orchestrator will pin them next). Do not run git/jj. Do not stage/commit. Run pixi run build + pixi run test (must stay 47/47 green incl. jtml.qml_lint).

WORK:
1. Replace the Repeater/Column/ScrollView with a ListView:
   - fixed row height (24px rows — U4 set list rows to 24; match the table's existing cell height 26 via implicitHeight; keep column widths aligned with the header: 44 + 6x78);
   - row delegates size from ListView.view.width (review D-02/I-08: the old Layout.fillWidth inside a plain Column was a no-op; the new delegate must actually stretch);
   - header/table width alignment with the scrollbar (review I-06: header RowLayout spans full width while the table is availableWidth — decide one width authority and align).
2. reuseItems: true with the commit-on-pool contract (D8, owner-confirmed):
   - capture-at-edit-start already exists (U3/D2) — the captured tuple (frame, model, axis) makes recycling safe;
   - onPooled must run AFTER the commit handler (Qt ordering: focus loss fires editingFinished before pooling; verify the order in your implementation and pin it with a comment — never reset text before the commit reads it);
   - onReused must imperatively re-sync the cell text (the storedValue binding is dead after editing — a plain re-bind or explicit text assignment from the new row's storedValue);
   - no stale text from the pooled instance (explicit reset in onReused).
3. Loader gate (review D-04/D-05): wrap the table content so it is NOT built while the dialog is closed — bind a Loader's active to the dialog's open state (or create the PosesTable on demand). Verify the PosesDialog still opens focused on the first cell (U4 added focusFirstCell — check it still works through the Loader; adjust the focus path if the Loader defers creation).
4. Empty/disabled states must survive: the 'No frames loaded' / 'Select a model' labels, the disabled-during-run behavior, and the validation message label keep working.
5. Scroll behavior: boundsBehavior default; smooth scrolling with 100+ frames; the scrollbar-gutter jitter concern (I-09) — if availableWidth re-evaluates on scrollbar toggle, mitigate by sizing the header from the same authority as the table.
6. Perf expectations to verify statically and document in comments: O(visible rows) delegates instead of O(frames); note the refresh-granularity re-check (I-02: PoseTableModel::refresh() full-resets on copy/load — acceptable for now, U7 may address notify granularity).
7. Keep the qmllint gate green (run .pixi/envs/default/lib/qt6/bin/qmllint --json; no NEW non-accepted warnings).

CONSTRAINTS:
- Behavior identical for the user flows except the perf/robustness improvements: editing, commit-on-edit-end, validation revert, dirty badge, copy/load/save buttons all keep working.
- Do not touch the C++ bridges (PoseBridge etc.). If you find a C++ change would be needed, STOP and report it — do not make it.
- Report: implementation summary per work item, the pool-ordering verification (how you confirmed editingFinished-before-onPooled), Loader/focus interplay result, test results, residual risks for the U6 harness to pin.

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