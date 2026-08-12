```json
{
  "reviewer": "testing",
  "findings": [
    {
      "title": "tst_PoseCell::test_commitUsesCapturedTuple mutates state AFTER the synchronous commit — the capture contract it claims to pin cannot fail (false confidence)",
      "severity": "P2",
      "file": "test/qml/tst_PoseCell.qml",
      "line": 62,
      "confidence": 90,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Mutate primaryModelIndex/frameRow while the cell still has focus (before the focus-out that fires editingFinished), then move focus out; a live-read implementation would then commit [99,1,1,12.5] and fail."
    },
    {
      "title": "test_datasetSwapNeverWritesNegativeOne never creates the transient it guards — removing the D7 suppress guard leaves the test green",
      "severity": "P2",
      "file": "test/qml/tst_StudyFlows.qml",
      "line": 194,
      "confidence": 85,
      "autofix_class": "gated_auto",
      "owner": "review-fixer",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Simulate the real swap: clear + repopulate frameListModel (FakeStudyBridge::clearDataset is a no-op) before emitting datasetChanged(), so a transient currentIndex reset actually occurs and a guard removal writes -1."
    },
    {
      "title": "Plan U6 composition-root scenarios untested: load ordering, replace-confirm, run-closes-dialogs, discard-reopen, toolbar run-lock (Camera/Model toggles), shell dirty badge — main.qml has no harness path",
      "severity": "P2",
      "file": "test/qml/tests.qrc",
      "line": 1,
      "confidence": 85,
      "autofix_class": "manual",
      "owner": "downstream-resolver",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Host main.qml in the harness (fake FileDialogBridge) or extract the toolbar/dialog flows into testable components; pin calibration-first, replace Yes/No/Esc, run-close with discardConfirmed, discard reopen, Camera/Model run-lock, shellDirty."
    },
    {
      "title": "Heatmap 0-keypoint guard (GPUHeatmap ctor, AllocateCurvatureHausdorfScore, OptimizerManager skip-if-empty) has zero test coverage",
      "severity": "P3",
      "file": "src/compute/gpu_heatmaps.cu",
      "line": 25,
      "confidence": 75,
      "autofix_class": "manual",
      "owner": "downstream-resolver",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Add a GPU oracle case: GPUHeatmap(0 keypoints, null host buffer) -> IsInitializedCorrectly() true, heatmap_on_gpu_ false; AllocateCurvatureHausdorfScore(0) no-op; run under the oracle label."
    },
    {
      "title": "qmllint gate accepted set: 'unqualified' rationale stale post-U3 (25 warnings remain in PosesTable) and the set silently absorbs delegate role typos; 'unused-imports' accepted with zero current warnings",
      "severity": "P3",
      "file": "test/qml_lint.cmake",
      "line": 18,
      "confidence": 90,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Fix the accepted-set doc (unqualified is now delegate model.*/ListView.view.* accesses, not U3-removed context properties); consider dropping unused-imports from the accepted set and re-run the gate."
    },
    {
      "title": "MlStrip D-08 toggle re-asserts (view Orig/Seg, implant Fem/Tib) and the view-toggle run-lock are unpinned",
      "severity": "P3",
      "file": "src/app/experimental/MlStrip.qml",
      "line": 285,
      "confidence": 75,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Add clicks + programmatic bridge mutations asserting the Binding re-sync contract, and extend test_mlRunLock to include the view toggles."
    },
    {
      "title": "SettingsPanel branch/leaf enable checkboxes unpinned despite dedicated testability objectNames",
      "severity": "P3",
      "file": "test/qml/tst_SettingsPanel.qml",
      "line": 1,
      "confidence": 75,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Toggle settingsBranchEnable/settingsLeafEnable and assert enableBranch/enableLeaf reach the fake."
    },
    {
      "title": "FakePoseBridge accepts values the real bridge rejects (inf/Infinity via parseFloat, no axis/frame range checks) — fake fidelity gap",
      "severity": "P3",
      "file": "test/qml/FakePoseBridge.qml",
      "line": 27,
      "confidence": 80,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Mirror real validation: reject !isFinite(v) and add the axis/frame/model bounds checks from PoseBridge::setPoseValue."
    },
    {
      "title": "Plan U6 scenario (c) dialog-level omissions: save-failure-keeps-dirty, copy-prev/next boundaries, estimate-label-cleared-on-failure untested",
      "severity": "P3",
      "file": "test/qml/tst_StudyFlows.qml",
      "line": 258,
      "confidence": 70,
      "autofix_class": "manual",
      "owner": "downstream-resolver",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Give FakePoseBridge a save-failure switch; add pins for failed-save keeps dirty, copy boundaries at frame 0/last, failed estimate keeps the label hidden."
    }
  ],
  "residual_risks": [
    "Harness runs exclusively offscreen; pooling/focus semantics verified only under offscreen — the xcb app path may differ.",
    "QmlVtkRenderer drag-sync scene-index fix has no automated coverage (VTK excluded from headless; render-smoke doesn't drive EndInteraction) — relies on owner's queued manual re-verify.",
    "Fake bridges can drift from the C++ bridge Q_PROPERTY surfaces — no compile-time cross-check.",
    "main.qml glue (context properties, viewportPanel.viewport.* relays, Connections) is outside both the harness and the gate's teeth — all its warnings fall in accepted categories.",
    "FileDialogBridge per-purpose dir memory + MRU untestable headless as written (modal exec, real QSettings writes) — no test exists.",
    "Root-level discardDialog reopen semantics (Esc/press-outside -> confirm -> reopen) unpinned."
  ],
  "testing_gaps": [
    "0-keypoint heatmap guard — no test at any level; needs a GPU oracle case.",
    "main.qml composition-root flows: calibration-first, replace Yes/No/Esc, partial-load, run-closes-dialogs, discard reopen, re-run-after-Completed, toolbar run-lock incl. Camera/Model, shellDirty badge.",
    "Run-lock matrix: MlStrip view toggles (Orig/Seg) missing; ViewportPanel lock untested by design (VTK).",
    "MlStrip D-08 Binding re-asserts — no click/programmatic pins.",
    "SettingsPanel branch/leaf checkboxes (objectNames declared, unused).",
    "Dialog error paths: save-failure keeps dirty, copy-prev/next boundaries, estimate-failure clears label.",
    "Fake fidelity: FakePoseBridge accepts non-finite input + lacks range checks; FakeStudyBridge omits selectionChanged/sceneBackgroundChanged on setCurrentFrame and clearDataset() is a no-op.",
    "FileDialogBridge per-purpose dir memory + MRU — no test path."
  ]
}
```