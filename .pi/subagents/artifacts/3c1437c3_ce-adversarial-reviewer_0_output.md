```json
{
  "reviewer": "adversarial",
  "findings": [
    {
      "title": "Heatmap guard trusts an uninitialized keypoint count: the 'no-segmentation' run the fix targets still aborts non-deterministically",
      "severity": "P1",
      "file": "src/coordinator/optimizer_manager.cpp",
      "line": 709,
      "confidence": 75,
      "autofix_class": "manual",
      "owner": "downstream-resolver",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Zero-init num_curvature_keypoints_ in the Frame ctor (include/compute/frame.h:92) and/or gate the GPUHeatmap no-upload path on frame_heatmaps.empty() instead of the int count."
    },
    {
      "title": "Commit-on-pool misses in-place recycling: an instant re-bind of a visible delegate (fast scroll / model refresh) silently drops the live edit without commit or validation",
      "severity": "P1",
      "file": "src/app/experimental/PosesTable.qml",
      "line": 232,
      "confidence": 75,
      "autofix_class": "advisory",
      "owner": "human",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Flush live edits in onReused too (commitIfEditing before resetDisplay) and on model-reset paths; the implementer's own test comment documents the drop for instant jumps."
    },
    {
      "title": "Reused cell retains focus after a pool round-trip: the next edit commits a stale (-1,-1,-1) tuple, is rejected by the range guard, and the typed value is lost",
      "severity": "P2",
      "file": "src/app/experimental/PoseCell.qml",
      "line": 101,
      "confidence": 50,
      "autofix_class": "advisory",
      "owner": "human",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Re-capture the commit tuple in resetDisplay/onReused when activeFocus is retained (or drop focus on pooling so onActiveFocusChanged re-captures)."
    },
    {
      "title": "Drag-sync index shift: RebuildModels skips failed-STL models, so the clamped 'primary' actor and reported scene index can name a different model — the pose lands on the wrong scene model and storage row",
      "severity": "P2",
      "file": "src/app/experimental/QmlVtkRenderer.cpp",
      "line": 302,
      "confidence": 50,
      "autofix_class": "advisory",
      "owner": "human",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Carry the true scene index with the actor through RebuildModels (skip at lines 274-277 shifts renderer-list indices vs scene indices); applyViewerPose's name-match only heals storage, not scene_->setModelPose."
    },
    {
      "title": "Dirty-close guard is blind to in-flight edits: closing the Poses dialog mid-edit either silently drops the typed value or commits it behind a discard prompt that cannot actually discard it",
      "severity": "P2",
      "file": "src/app/experimental/PosesDialog.qml",
      "line": 204,
      "confidence": 50,
      "autofix_class": "advisory",
      "owner": "human",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Surface in-flight (uncommitted) edits to the dirty contract; flush commitIfEditing deterministically before Loader deactivation on close."
    },
    {
      "title": "Discard-confirm Escape loop: Escape on the 'Discard unsaved changes?' dialog reopens the dirty dialog, whose next Escape re-triggers the guard — Escape can never dismiss it",
      "severity": "P3",
      "file": "src/app/experimental/main.qml",
      "line": 150,
      "confidence": 75,
      "autofix_class": "advisory",
      "owner": "human",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Exclude CloseOnEscape on discardDialog (buttons only) or clear pendingDialog so onRejected cannot re-arm the cycle."
    },
    {
      "title": "QSettings MRU read-modify-write races across concurrent app instances and defeats exact-match dedupe under symlinked paths",
      "severity": "P3",
      "file": "src/app/experimental/FileDialogBridge.cpp",
      "line": 50,
      "confidence": 50,
      "autofix_class": "advisory",
      "owner": "human",
      "requires_verification": false,
      "pre_existing": true,
      "suggested_fix": "Canonicalize entries before dedupe and document last-writer-wins (or serialize the RMW)."
    },
    {
      "title": "focusFirstCell runs before the ListView has instantiated row 0's delegate: itemAtIndex(0) is null on the first open, so the U4 focus-on-open feature silently no-ops",
      "severity": "P3",
      "file": "src/app/experimental/PosesTable.qml",
      "line": 258,
      "confidence": 50,
      "autofix_class": "advisory",
      "owner": "human",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Defer the focus via Qt.callLater or retry on the ListView's first layout pass; pin with a QML test."
    }
  ],
  "residual_risks": [
    "suppressFrameSync: no stuck-true path found — the deferral always clears the flag and clearDataset+loadImages are synchronous in one event-loop turn (pinned by test_datasetSwapNeverWritesNegativeOne); the guard is ordering-dependent if a future async study-load path emits datasetChanged without the synchronous tail. clearDataset emits datasetChanged twice (SessionStateController + StudyBridge), queueing two deferrals.",
    "AllocateCurvatureHausdorfScore re-allocates per (frame, stage) without cudaFree (pre-existing leak; the new 0-keypoint early return now spans 0→N transitions); allocation failure sets initialized_correctly_=false but DIRECT_DILATION never checks it.",
    "Dead-code landmine: CurvatureHeatmapMetric launches <<<1, num_kp>>> and gridDim.z=num_kp — 0 keypoints = invalid configuration; unreachable today and NOT protected by the new guard.",
    "0-keypoint GPUHeatmap path never calls cudaSetDevice and heatmap_on_gpu_ is never read — harmless while the distance-map loop sets the device first; a latent ordering trap.",
    "Mixed keypoint counts across frames: hausdorf buffer sized for the stage-init frame only — harmless while the curvature metric is unused.",
    "Frame::setCurvatureHeatmaps writes hm*.png/contour.png into CWD (pre-existing side effect).",
    "applyViewerPose silently drops drags when the renderer's reported index is out of the scene list's range."
  ],
  "testing_gaps": [
    "No unit test asserts GetNumCurvatureKeypoints() == 0 on a fresh Frame (headless-testable; the heatmap guard's premise is unpinned).",
    "The commit-on-pool test avoids the instant in-place recycle instead of pinning it (tst_PosesTable.qml:79-81).",
    "No test covers reuse-with-retained-focus → stale-tuple commit; the code comments' focus-behavior claims contradict the test comment.",
    "No test covers closing the Poses dialog mid-edit.",
    "focusFirstCell is never exercised.",
    "RebuildModels STL-skip index shift is untested (ClampedActiveIndex could be unit-tested headlessly).",
    "No MRU dedupe/concurrency test."
  ]
}
```