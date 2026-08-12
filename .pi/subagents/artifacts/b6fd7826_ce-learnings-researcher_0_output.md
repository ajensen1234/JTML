Artifact written. Here is the compact result:

## Institutional Learnings Search Results

**Search context:** Diff ywpkloqo→@ (plan 007 QML improvement pass). Keywords: queued-functor pose-sync, render-thread contract, double truncation, moc signals placement, layered-lib split, ListView.view nested-scope, commit-on-pool, pooling focus. **Files scanned:** 12 solution entries in `docs/solutions/` (5 subdirs: build-errors, conventions, logic-errors, tooling-decisions, ui-bugs; no `patterns/critical-patterns.md` — absent). **Relevant matches:** 8.

**Cross-check verdicts (verified against the diff):**
- **Drag-sync fix conforms** — direct `reportModelPoseAdjusted` emit retained, zero `invokeMethod` left in `src/app/experimental/`; scene-index fix is internally consistent (all `SetPrimaryActor` callers updated, clamped index pinned + written back). ⚠️ But a **pre-existing stale comment at QmlVtkRenderer.cpp:103-104 still prescribes the broken queued pattern** ("post a QUEUED invocation … `queueModelPoseSync` emits").
- **Harness respects the render-thread contract** — no VTK/bridges/jtml libs, offscreen env, ViewportPanel deliberately absent (matches the new 08-12 convention entry, verified claim-by-claim).
- **moc signals-last** holds in all 7 modified headers; **no double narrowing** added (PoseBridge parses text→double with finite check; bit-exact pins exist); **layered-lib** conventions hold (explicit sources, block-scoped, no new files in compute/coordinator).
- **Commit-on-pool implemented per the learning** (attached `ListView.onPooled`/`onReused`, `edited` flag) — but the PosesTable header comment **contradicts the file's own mechanism and the documented Qt 6.7 pooling finding**.

```json
{
  "reviewer": "ce-learnings-researcher",
  "findings": [
    {
      "title": "Stale comment on the drag-sync observer prescribes the documented-broken queued-functor pattern (QmlVtkRenderer.cpp:103-104)",
      "severity": "P2",
      "file": "src/app/experimental/QmlVtkRenderer.cpp",
      "line": 103,
      "confidence": 100,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": false,
      "pre_existing": true,
      "suggested_fix": "Rewrite comment to describe the actual direct-emit mechanism (reportModelPoseAdjusted -> emit, AutoConnection queues to GUI thread); remove the queueModelPoseSync/'QUEUED invocation' description of the dead, silently-broken pattern."
    },
    {
      "title": "PosesTable.qml header comment contradicts the file's own commit-on-pool mechanism and the documented Qt 6.7 pooling learning (PosesTable.qml:19-22 vs 223-229)",
      "severity": "P2",
      "file": "src/app/experimental/PosesTable.qml",
      "line": 19,
      "confidence": 100,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "Rewrite the 'commit-on-pool ordering' bullet: pooling does NOT drop focus, ListView.onPooled is the flush point, remove 'onPooled therefore has no text to protect' (the code does flush there)."
    },
    {
      "title": "Drag-sync scene-index fix (the diff's marquee owner fix) has no test pinning a non-zero scene index — every applyViewerPose test passes index 0",
      "severity": "P3",
      "file": "test/unit/experimental_selection_test.cpp",
      "line": 447,
      "confidence": 75,
      "autofix_class": "manual",
      "owner": "downstream-resolver",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "Add a >=2-model unit pin: toggleModelSelected(1); applyViewerPose(1, ...) must write storage/scene model 1 (the 'dragging the femur moved the tibia' regression); pin the name-match divergence path; optionally add a non-zero-index drag leg to the oracle render smoke."
    },
    {
      "title": "Heatmap 0-keypoint guards close the upload-abort path, but the guarded allocations' only consumer (CurvatureHeatmapMetric) is dead code that would crash on the same class if ever wired",
      "severity": "P3",
      "file": "src/compute/gpu_metrics.cu",
      "line": 98,
      "confidence": 75,
      "autofix_class": "advisory",
      "owner": "downstream-resolver",
      "requires_verification": false,
      "pre_existing": true,
      "suggested_fix": "Add the same 0-keypoint guard inside GPUMetrics::CurvatureHeatmapMetric (curvature_hausdorf_metric.cu:41-55) before the <<<1, num_kp>>> launch and score memcpy, or delete the dead metric until wired; extend guard comments to state the metric must not be called with num_kp <= 0."
    }
  ],
  "residual_risks": [
    "Full-path drag-sync coverage depends on the oracle render smoke (not in this diff); headless harness cannot cover it by constraint.",
    "0-keypoint contract enforced only in the two guards + comments; no test exercises a 0-keypoint study through the upload path (GPU-bound; oracle candidate).",
    "Two contradictory comments on the commit-on-pool contract — whichever the next maintainer believes first wins (finding 2)."
  ],
  "testing_gaps": [
    "No unit pin for applyViewerPose with a non-zero scene index on a multi-model scene — the exact regression shape the owner's fix addresses.",
    "No unit pin for the name-match divergence path in StudyBridge::applyViewerPose (scene order vs loaded_models order).",
    "No headless pin for the ViewportPanel Connections hop (deliberate — VTK territory; oracle smoke should get a non-zero-index leg).",
    "No 0-keypoint heatmap upload/run pin (oracle/render-labeled candidate)."
  ]
}
```