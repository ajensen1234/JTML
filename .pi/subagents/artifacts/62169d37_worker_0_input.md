# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement plan 007 U3 for the JTML repo (cwd: /home/ajj/repo/uf/JTML). READ first: docs/plans/2026-08-12-007-refactor-qml-experimental-improvement-pass-plan.md (U3 section + Key Technical Decisions D1-D8 + the Open Questions), docs/reviews/2026-08-12-qml-experimental-review.md (findings D-07 run-lock, D-08 ButtonGroup, I-05 keyboard), and the current state of src/app/experimental/ (U2 just extracted: main.qml bootstrap + StudyPanel/MlStrip/RunBar/ViewportPanel/PosesDialog/PosesTable/PoseCell).

U3 GOAL: fix the view-layer state gaps with data-integrity consequences. This unit TOUCHES C++ BRIDGES — the first unit to do so. Owner constraint: only src/app/experimental/** + additive test/unit suite extensions + additive test/CMakeLists.txt registrations. NEVER touch backend seams (domain/services/coordinator/compute), src/view, or anything else.

WORK ITEMS (each maps to a plan decision):

1. POSE-CELL COMMIT CONTRACT (D2, fixes C1/C2): PoseCell.qml currently commits on onEditingFinished reading frameRow/axisIndex props + studyBridge.primaryModelIndex AT COMMIT TIME. Rework: capture (frameRow, axisIndex, primaryModelIndex) when editing starts (onActiveFocusChanged entering focus, or editingStarted); commit against the CAPTURED values; a failed commit must restore the display WITHOUT killing the binding (use a Binding re-sync or Qt.binding() — never a one-shot text = assignment). The capture tuple must survive delegate recycling (U5 prerequisite).
2. POSE-TABLE REFRESH OWNER (D3, fixes I1/I2): PoseBridge.h/.cpp — connect optimizerBridge.runStateChanged (completed AND error states) + the viewerPoseApplied / scenePoseChanged paths → refresh the table model (single refresh owner; relay plumbing only, no policy). First verify the stale-on-reopen premise empirically per the plan: write the U6-style reasoning in a comment, but the fix is unconditional (the review confirmed QQC2 Dialog never destroys contentItem).
3. ML-SEED INVALIDATION (D4, fixes I3): any manual pose write on the seeded frame+model drops the pending seed — viewer drags (applyViewerPose), pose-table edits (setPoseValue), copy-prev/next (copyPrevious/copyNext), pose + kinematics file loads (loadPoseFile/loadKinematics). The pending seed lives in OptimizerBridge (setSeedPose/clearSeedPose — verify by reading OptimizerBridge.h/.cpp and MlBridge.h/.cpp; the seed may currently be cleared only on selectionChanged). Route invalidation through OptimizerBridge::clearSeedPose (or whatever the actual seed-clear seam is). Additive signals/relays only.
4. RUN-LOCK MATRIX (D5 + review D-07, fixes I4): extend !optimizerBridge.running locks to: Black-sil. checkbox, Fem/Tib buttons, AND the Camera/Model interaction-mode toggles (review D-07 — they escaped the original inventory). Introduce a single readonly runLocked property in the QML shell root (or per-component) and bind every locked control to it. The viewport lock gets a VISIBLE state: when optimizerBridge.running, ViewportPanel shows a 'Running — interaction locked' overlay/dim over the renderer (and the renderer's mouse events are blocked — enabled: false on the QmlVtkRenderer item).
5. ESTIMATE ENABLEMENT (D6, fixes I5): MlStrip's Estimate button binding gains mlBridge.hasSegmentModel (check MlBridge exposes it or add the enablement state additively); add/keep the hint label for the disabled case.
6. BUTTONGROUP SINGLE-SOURCE (review D-08): the three ButtonGroups (interactGroup in main.qml toolbar, implantGroup + viewGroup in MlStrip) pair checked: bridge.x === N with onClicked: bridge.x = N — QQC2 ButtonGroup writes checked imperatively on click, killing the binding on the clicked button. Fix: keep checked bound to the bridge value and make onClicked the only writer (verify the group exclusivity still works; if the imperative check-kill persists, use the group's checkedButton instead of per-button checked bindings).
7. KEYBOARD WIRING (D7, fixes I6): StudyPanel frame list — wire onCurrentIndexChanged to studyBridge.setCurrentFrame (single source of truth) with a DATASET-SWAP GUARD: suppress the sync while a dataset replace is in flight (clearDataset model swap) so a transient -1/0 index never reaches the bridge. Model list — Space/Enter toggles the row (Keys handler on the delegate or ListView). Rows get activeFocusOnTab + a visible focus indicator (bound to activeFocus). Note: this is the wiring — the full keyboard audit is U4.
8. EXTEND THE BRIDGE SUITES (additive): test/unit/experimental_pose_bridge_test.cpp + experimental_ml_bridge_test.cpp — add cases pinning: (a) commit contract paths (the pose table refresh fires on runStateChanged completed/error + viewerPoseApplied), (b) seed invalidation on copy/load/edit/drag paths, (c) hasSegmentModel-aware enablement state if it landed in the bridge. Follow the existing test style exactly (Catch2, repo-root fixtures).

CONSTRAINTS:
- Bridge changes are ADDITIVE RELAYS ONLY: new signals/relays/enablement state; no behavior changes to existing paths or APIs. Read the existing bridge tests first and keep them green.
- Follow the plan's thinness rule: no logic in bridges beyond relay plumbing.
- Do not run git/jj commands; do not stage or commit. The orchestrator handles jj.
- Run pixi run build and the headless test suite (pixi run test) — the experimental bridge tests (jtml.experimental_pose_bridge, jtml.experimental_ml_bridge, jtml.experimental_optimizer_gate, jtml.experimental_selection, jtml.experimental_settings) must pass. Do NOT run oracle-labeled tests (jtml.qml_parity_check is ~1h; the orchestrator runs it after this unit).
- If a design decision is genuinely ambiguous (e.g., where the dataset-swap guard lives), make the minimal choice consistent with the plan and document it in your report.
- Report: files changed, each work item's implementation summary, test results, deviations with reasons, and anything needing owner manual-visual verification.

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