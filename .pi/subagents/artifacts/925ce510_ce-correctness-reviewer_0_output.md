Artifact written and validated. Final compact report:

```json
{
  "reviewer": "correctness",
  "findings": [
    {
      "title": "PoseCell: stale commit tuple (-1,-1,-1) after delegate recycle with retained focus — typing into the recycled focused cell without clicking commits to (frame -1, model -1), is rejected, and the typed value is silently lost",
      "severity": "P2",
      "file": "src/app/experimental/PoseCell.qml",
      "line": 101,
      "confidence": 75,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Re-capture the commit tuple in resetDisplay() when the cell still holds focus (`if (activeFocus) { commitFrame = frameRow; commitModel = studyBridge.primaryModelIndex; commitAxis = axisIndex }`); pin with a type-without-clicking-after-recycle test"
    },
    {
      "title": "D7 pose-table cell keyboard contract never landed: no Left/Right cell navigation, no Up/Down row moves, no Esc-revert, no single-tab-stop — recycled rows remain unreachable by keyboard",
      "severity": "P2",
      "file": "src/app/experimental/PosesTable.qml",
      "line": 107,
      "confidence": 100,
      "autofix_class": "manual",
      "owner": "downstream-resolver",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Implement the plan D7/U4 contract (arrows between cells/rows, Esc reverts, single tab stop) + the U6 (e) pins, or formally record the deferral"
    },
    {
      "title": "GPUMetrics: curvature_hausdorf_score_/dev_curvature_hausdorf_score_ never initialized; the new 0-keypoint no-op leaves them indeterminate and the destructor cudaFree's them — newly reachable in the no-segmentation run the heatmap guard now enables",
      "severity": "P2",
      "file": "src/compute/gpu_metrics.cu",
      "line": 97,
      "confidence": 75,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "Null-initialize both members in the GPUMetrics ctor (or in the AllocateCurvatureHausdorfScore early return)"
    },
    {
      "title": "SettingsPanel: inline value:/currentIndex:/checked: bindings die on first user interaction, so Reset leaves the form display desynced from the bridge (same class the D-08 Binding-object fix addressed elsewhere)",
      "severity": "P3",
      "file": "src/app/experimental/SettingsPanel.qml",
      "line": 119,
      "confidence": 75,
      "autofix_class": "gated_auto",
      "owner": "downstream-resolver",
      "requires_verification": true,
      "pre_existing": true,
      "suggested_fix": "Convert SpinBox value / ComboBox currentIndex / enable-CheckBox checked bindings to persistent Binding objects (D-08 pattern)"
    },
    {
      "title": "tst_PosesTable commit-on-pool comment misdocuments the mechanism ('focus loss fires editingFinished BEFORE reuse') — pooling retains focus (probe-verified); the explicit onPooled flush is the real mechanism; the 'instant jump... no commit' claim is also wrong",
      "severity": "P3",
      "file": "test/qml/tst_PosesTable.qml",
      "line": 94,
      "confidence": 100,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "Rewrite the comment to match the landed edited-flag + ListView.onPooled commitIfEditing mechanism"
    },
    {
      "title": "FakePoseBridge.setPoseValue over-accepts: parseFloat accepts partial-numeric strings ('12.5abc') and 'Infinity' that the real bridge rejects — QML pins cannot catch over-acceptance regressions",
      "severity": "P3",
      "file": "test/qml/FakePoseBridge.qml",
      "line": 27,
      "confidence": 75,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "Mirror the real bridge's full-parse + finite validation in the fake"
    },
    {
      "title": "Pose table goes stale after an ML estimate: MlBridge's estimate writes storage directly (SavePose) but the D3 refresh triggers (run terminal state, viewerPoseApplied) do not include it — an open table shows the pre-estimate value, inviting a stale-data 'correction'",
      "severity": "P3",
      "file": "src/app/experimental/MlBridge.cpp",
      "line": 386,
      "confidence": 75,
      "autofix_class": "gated_auto",
      "owner": "review-fixer",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Add the successful-estimate path to the D3 refresh-owner relay (refreshTable) without routing through clearSeedPose"
    }
  ],
  "residual_risks": [
    "GPUHeatmap memcpy-failure double-free (pre-existing, gpu_heatmaps.cu:64-65)",
    "AllocateCurvatureHausdorfScore per-stage re-alloc leak for runs with keypoints (pre-existing)",
    "MlStrip blackSilCheck inline checked binding dies on first toggle (no programmatic writer today)",
    "FileDialogBridge: MRU dirs not existence-validated; multi-dir multi-select remembers only first file's dir; exec() blocks GUI thread",
    "Dataset-swap deferral window: a click inside the single-tick callLater window is dropped (negligible)",
    "Mid-edit Esc-close commits via focus-out->editingFinished (probe-verified); the discard-confirm does not undo the committed in-memory edit",
    "Stop-path refreshTable: stops land in a terminal state, table re-reads (verified)",
    "D4 seed invalidation complete across drags/edits/copy/load/selection/dataset-clear (verified); estimate's own SavePose correctly keeps the seed"
  ],
  "testing_gaps": [
    "No pin for recycled-with-retained-focus typing (F1) — tst_PosesTable always re-focuses before typing",
    "No pose-table cell-keyboard pin (F2) — unimplemented",
    "FakePoseBridge validation divergence (F6)",
    "No test for ML-estimate -> pose-table refresh (F7)",
    "GPUMetrics 0-keypoint teardown is GPU territory; member init is the headless-verifiable half",
    "FileDialogBridge MRU helpers (rememberDir/lastDir/mruUrls) have no headless test",
    "Frame-list Up/Down keyboard verified empirically here, not pinned in the harness"
  ]
}
```