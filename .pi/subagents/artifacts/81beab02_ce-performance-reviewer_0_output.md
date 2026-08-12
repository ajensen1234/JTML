Artifact written and validated. Final compact report:

```json
{
  "reviewer": "performance",
  "findings": [
    {
      "title": "FileDialogBridge constructs a fresh QFileDialog per open — 3.2 s modal UI freeze (profile hotspot #1)",
      "severity": "P2",
      "file": "src/app/experimental/FileDialogBridge.cpp",
      "line": 35,
      "confidence": 75,
      "autofix_class": "advisory",
      "owner": "downstream-resolver",
      "requires_verification": true,
      "pre_existing": true,
      "suggested_fix": "Interim: one member QFileDialog reused across opens (amortize construction + file-model cache); final: plan's queued QML PathPickerDialog (deferred item B)"
    },
    {
      "title": "RunBar progress bindings: 18,736 evals/label per 8 runs — coarse shared-NOTIFY granularity; verified acceptable today",
      "severity": "P3",
      "file": "src/app/experimental/OptimizerBridge.h",
      "line": 60,
      "confidence": 75,
      "autofix_class": "advisory",
      "owner": "downstream-resolver",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "No change now (measured trivial: ~0.7 ms/frame vs 7 ms budget, zero jank). Lever if U7 re-profile shows jank: per-property NOTIFY or change-guard in onControllerProgressChanged"
    },
    {
      "title": "PoseTableModel::notifyCellChanged emits whole-row dataChanged without a role list — single-cell commit re-evaluates all 6 cells + frame label",
      "severity": "P3",
      "file": "src/app/experimental/PoseBridge.cpp",
      "line": 122,
      "confidence": 75,
      "autofix_class": "gated_auto",
      "owner": "review-fixer",
      "requires_verification": false,
      "pre_existing": true,
      "suggested_fix": "Role-filter the emit (axis→role switch, emit dataChanged(cell, cell, {role})) — one-liner, behavior-preserving; the code already Q_UNUSEDs axis"
    },
    {
      "title": "Model-list delegate selection color: selectedModels.indexOf(index) — O(visible × selected) per selection change",
      "severity": "P3",
      "file": "src/app/experimental/StudyPanel.qml",
      "line": 135,
      "confidence": 75,
      "autofix_class": "advisory",
      "owner": "downstream-resolver",
      "requires_verification": false,
      "pre_existing": true,
      "suggested_fix": "Q_INVOKABLE bool isModelSelected(int) on StudyBridge (Set-backed); cosmetic at expected model counts (review I-10 agrees)"
    }
  ],
  "residual_risks": [
    "viewerPoseApplied → refreshTable full model reset while the Poses dialog is open: drag-end destroys+recreates visible delegates (~90 items, negligible perf) but a mid-edit cell is destroyed without commit-on-pool (reset destroys delegates, onPooled never fires) — silent edit-loss edge for correctness/reliability review (AppBridge.cpp ~L85-93)",
    "Profile session never exercised the Poses dialog / table scroll / cell editing (frame-picker regression cut it short) — U5 virtualized-table recycling unmeasured; plan's U7 re-profile threshold applies",
    "qml_lint gate: ~2-4 s per headless ctest run, no caching — negligible vs multi-minute suite; growth lever = checksum cache/changed-files filter",
    "RunBar churn lever deliberately not applied (F2) — recorded for U7 re-profile comparison",
    "PosesDialog's four QML FileDialogs use the portal path (single-select, so the multi-select defect doesn't apply); latency unmeasured"
  ],
  "testing_gaps": [
    "No repeated-open timing pin for FileDialogBridge / the member-reuse interim lever (F1)",
    "No Qt Quick Test asserts bounded delegate counts on the virtualized pose table (tst_PosesTable pins recycle behavior, not instance counts) — U5's O(visible) win unguarded",
    "U7 re-profile scope must cover: dialog open/close, table scroll + recycling, mid-scroll commit, run with stage transitions (stageText re-evals vs stage changes)",
    "Fakes exercise only single-digit model counts — no guard pins the small-model assumption"
  ]
}
```