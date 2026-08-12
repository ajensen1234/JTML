Artifact written and validated. Full analysis is in `.context/compound-engineering/ce-code-review/20260812-123833-8f0aeec8/ce-reliability-reviewer.json` (5 findings, all with evidence quotes).

```json
{
  "reviewer": "reliability",
  "findings": [
    {
      "title": "GPUMetrics destructor cudaFree/cudaFreeHosts uninitialized curvature pointers once AllocateCurvatureHausdorfScore early-returns (new 0-keypoint path)",
      "severity": "P1",
      "file": "src/compute/gpu_metrics.cu",
      "line": 102,
      "confidence": 100,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Init curvature_hausdorf_score_/dev_curvature_hausdorf_score_ to nullptr in the GPUMetrics ctor (or default member init in gpu_metrics.cuh:132-133)"
    },
    {
      "title": "Guard asymmetry: caller keys on buffer emptiness, GPUHeatmap ctor keys on keypoint count — nullptr+size>0 re-enters the pre-fix abort",
      "severity": "P3",
      "file": "src/coordinator/optimizer_manager.cpp",
      "line": 710,
      "confidence": 50,
      "autofix_class": "advisory",
      "owner": "review-fixer",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "Harden the ctor guard to `num_keypoints <= 0 || host_heatmaps == nullptr` (gpu_heatmaps.cu:25)"
    },
    {
      "title": "Per-frame full by-value copy of the flattened heatmap buffer on the GUI thread during Initialize",
      "severity": "P3",
      "file": "src/coordinator/optimizer_manager.cpp",
      "line": 704,
      "confidence": 75,
      "autofix_class": "advisory",
      "owner": "review-fixer",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "`const auto& frame_heatmaps = frames_A_[i].getCurvatureHeatmaps();` (lifetime-extension, no copy)"
    },
    {
      "title": "FileDialogBridge QSettings write failures are fully silent (status() never checked, no sync)",
      "severity": "P3",
      "file": "src/app/experimental/FileDialogBridge.cpp",
      "line": 54,
      "confidence": 100,
      "autofix_class": "advisory",
      "owner": "review-fixer",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "Check settings_.status() after setValue, qWarning on AccessError, consider explicit sync()"
    },
    {
      "title": "Dirty-close guard: 'Yes / Discard' does not discard anything — changes stay live in the bridge and the next run silently uses them",
      "severity": "P3",
      "file": "src/app/experimental/main.qml",
      "line": 147,
      "confidence": 75,
      "autofix_class": "advisory",
      "owner": "human",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "Product decision: revert-on-Yes or reword the confirm to 'Close without saving to file?'"
    }
  ],
  "residual_risks": [
    "GPUHeatmap 0-keypoint guard VERIFIED leak-free: dev_heatmap_=0 set before the early return (gpu_heatmaps.cu:12); cudaFree(0) in the dtor is a no-op; no allocation on that path; heatmap_on_gpu_ written on every path",
    "Optimizer run-state machine VERIFIED sound: pre-fix heatmap-upload failure was cleanly handled (ghost thread + M6 bind + M10a seed restore + Error state); post-fix the 0-kp run completes through OptimizedFrame->Completed->refreshTable; DIRECT_DILATION's at(i) stays frame-aligned; null device pointer never dereferenced (CurvatureHeatmapMetric is dead code)",
    "Drag-sync scene-index fix VERIFIED consistent: SetPrimaryActor(actor,index) atomic at both call sites; observer early-returns on null actor; StudyBridge::applyViewerPose + LocationStorage::SavePose bounds-checked",
    "discardConfirmed reset paths VERIFIED complete: both dialogs reset on every onOpened incl. No-reopen; RunBar sets both flags before close (main.qml:509-510)",
    "Pre-existing, out of scope: terminal-state ghost-thread window unlocks UI briefly; run() rejects with a message — handled",
    "GPUHeatmap default ctor declared but never defined; unused in this diff (latent link error only)"
  ],
  "testing_gaps": [
    "0-keypoint GPU path untestable headless (no GPU in tests) — guard semantics + GPUMetrics dtor safety pinned only by inspection; recommend compute-sanitizer on a real heatmap-less run + teardown",
    "tst_StudyFlows test_dirtyCloseGuard pins the signal + discardConfirmed suppression only — Yes/No outcomes (No-reopen, reopen->guard-rearm cycle, Yes-keeps-changes) untested",
    "FileDialogBridge QSettings failure behavior untested (no injectable unwritable settings path)",
    "Guard asymmetry (keypoints>0 with empty buffer) untested on both sides of the seam"
  ]
}
```