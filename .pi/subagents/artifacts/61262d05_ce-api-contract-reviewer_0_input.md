# Task for ce-api-contract-reviewer

You are ce-api-contract-reviewer. The QML-callable surfaces: OptimizerBridge.h (hasSeedPose/clearSeedPose additions), PoseBridge.h (refreshTable), MlBridge (hasSegmentModel), FileDialogBridge signature change (purpose param — QML call sites updated in main.qml?), AppBridge.cpp wiring, the qmldir entries (new component registrations), resources.qrc completeness (every new .qml listed?), and whether any pinned bridge test or the parity path depends on a changed signature.

INTENT: Plan 007 — full improvement pass over the experimental QML front-end of a Qt 6.7.2 + VTK 9.3 + CUDA knee-implant registration app. Units: structured review; Theme tokens + component extraction (main.qml 1020->230 lines); view-state gap fixes (PoseCell commit contract, pose-table refresh owner, ML-seed invalidation via OptimizerBridge::clearSeedPose, run-lock matrix incl. Camera/Model toggles, Estimate enablement, keyboard wiring + dataset-swap guard); UX audit + visual composition pass; pose-table virtualization (ListView reuse + commit-on-pool); Qt Quick Test harness (test/qml/, 38 test functions, fake bridges, property injection); profiling build (pixi configure-profiling task). Plus two owner-feedback rounds: toolbar overflow fix, drag-sync scene-index fix (QmlVtkRenderer), heatmap-less optimizer runs (GPUHeatmap 0-keypoint guard + AllocateCurvatureHausdorfScore guard + OptimizerManager skip-if-empty), Black-sil. checkbox label, frame-picker ListView.view-null-in-nested-MouseArea fix, FileDialogBridge per-purpose dir memory + sidebar MRU.
CONSTRAINTS: oracle parity band must stay intact (jtml.qml_parity_check green, IoU 0.993627); widgets app (src/view) and backend seams untouched except the owner-requested heatmap guard (src/compute/gpu_heatmaps.cu, gpu_metrics.cu, src/coordinator/optimizer_manager.cpp); headless tests must not touch GPU/VTK.
REVIEW SCOPE: diff base ywpkloqo -> working copy @. CHANGED FILES (read them; do NOT rely on this list alone — check the diff too):
  src/app/experimental/: main.qml (bootstrap), StudyPanel.qml, MlStrip.qml, RunBar.qml, ViewportPanel.qml, PosesDialog.qml, PosesTable.qml, PoseCell.qml, SettingsPanel.qml, Theme.qml, QmlVtkRenderer.cpp, AppBridge.cpp, OptimizerBridge.h/.cpp, PoseBridge.h/.cpp, StudyBridge.cpp, FileDialogBridge.h/.cpp, qmldir, resources.qrc, renderer.qml
  src/compute/gpu_heatmaps.cu, gpu_metrics.cu; src/coordinator/optimizer_manager.cpp
  test/qml/ (harness: main.cpp, tests.qrc, fakes, tst_*), test/qml_lint.cmake, test/CMakeLists.txt, test/unit/experimental_pose_bridge_test.cpp, experimental_ml_bridge_test.cpp
  pixi.toml (configure-profiling/build-profiling tasks), docs/plans/2026-08-12-007-*.md (the PLAN — read for requirements verification), docs/reviews/2026-08-12-qml-experimental-review.md, docs/solutions/* (2 new entries), docs/handoff-2026-08-12-*.md, .gitignore
RUN `jj diff --from ywpkloqo --to @` for the full diff (this repo uses jj, not git).

OUTPUT CONTRACT:
1. Write your FULL analysis to .context/compound-engineering/ce-code-review/20260812-123833-8f0aeec8/<your-name>.json — JSON with: { reviewer, findings: [ { title, severity (P0|P1|P2|P3), file, line, confidence (0|25|50|75|100), autofix_class (safe_auto|gated_auto|manual|advisory), owner (review-fixer|downstream-resolver|human|release), requires_verification (bool), pre_existing (bool), suggested_fix (optional), why_it_matters, evidence[] } ], residual_risks: [], testing_gaps: [] }.
2. Return COMPACT JSON: { reviewer, findings: [ {title, severity, file, line, confidence, autofix_class, owner, requires_verification, pre_existing, suggested_fix?} ], residual_risks: [], testing_gaps: [] }.
Confidence anchors: 50 = verified real but advisory; 75 = verified, implementers will concretely hit it; 100 = airtight. Suppress below 50. Every finding needs file+line and a direct evidence quote in the artifact.
YOU ARE READ-ONLY: no file edits (except your artifact write), no git/jj mutations, no commits.

Your persona rubric is built into your agent configuration — apply it, with the suppression rules for findings that belong to other personas.

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