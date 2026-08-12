Artifact written. Here is the compact result:

```json
{
  "reviewer": "api-contract",
  "findings": [
    {
      "title": "FileDialogBridge getOpenFileNames purpose/MRU surface: only bridge with zero headless coverage; QSettings writes real user config; unvalidated magic purpose strings",
      "severity": "P2",
      "file": "src/app/experimental/FileDialogBridge.cpp",
      "line": 20,
      "confidence": 75,
      "autofix_class": "manual",
      "owner": "review-fixer",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "Add headless test for rememberDir/lastDir/mruUrls round-trip with XDG_CONFIG_HOME redirected to temp; pin purpose-bucket isolation + kMaxMruDirs=5"
    },
    {
      "title": "modelPoseAdjusted emitted sceneModelIndex semantics changed (hardcoded 0 -> dragged actor index) — owner-requested drag-sync fix; all consumers verified compatible (oracle ignores index, applyViewerPose guards <0), render smoke re-run pending",
      "severity": "P2",
      "file": "src/app/experimental/QmlVtkRenderer.cpp",
      "line": 121,
      "confidence": 75,
      "autofix_class": "manual",
      "owner": "downstream-resolver",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Re-run jtml.qml_render_smoke (xcb/GL) after landing; keep SetPrimaryActor two-caller index invariant documented"
    },
    {
      "title": "FakeOptimizerBridge surface drift: real bridge gained Q_INVOKABLE hasSeedPose (plus set/applySeedPose) not mirrored in the fake; nothing enforces fake<->real parity; QML never calls it today so nothing breaks",
      "severity": "P3",
      "file": "test/qml/FakeOptimizerBridge.qml",
      "line": 9,
      "confidence": 75,
      "autofix_class": "safe_auto",
      "owner": "review-fixer",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "Add property bool hasSeedPose: false + seed stub call-counters to the fake; note seed surface as intentionally unpinned in the header"
    },
    {
      "title": "D3 terminal-state leg (runStateChanged Completed/Error -> refreshTable) pinned nowhere — acknowledged in test comments as not headlessly drivable; wiring correct by inspection (refreshTable emits only modelReset, never poseTableChanged, so it cannot clear a seed)",
      "severity": "P3",
      "file": "src/app/experimental/AppBridge.cpp",
      "line": 84,
      "confidence": 75,
      "autofix_class": "manual",
      "owner": "review-fixer",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "If a controller state-injection seam is ever added, pin the lambda filter (exactly one refresh on Completed and Error, zero otherwise)"
    },
    {
      "title": "PoseTableModel::refresh() full-reset can destroy a mid-edit PoseCell while the modal:false pose dialog is open and a viewer drag lands; commit relies on unverified destruction-triggered focus-out editingFinished; harness cannot pin it (ListModel has no modelReset, fakes lack refreshTable)",
      "severity": "P3",
      "file": "src/app/experimental/PosesTable.qml",
      "line": 139,
      "confidence": 50,
      "autofix_class": "manual",
      "owner": "review-fixer",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Pin commit-on-reset with the real PoseTableModel in the pose_bridge fixture, or flush focused cells before applyViewerPose; also reconcile the contradictory onPooled-ordering comments (PosesTable.qml header vs PoseCell.qml:47-53)"
    }
  ],
  "residual_risks": [
    "jtml.qml_parity_check (IoU 0.993627) not re-runnable here (no GL/GPU); parity binary (12:08:26) postdates all bridge changes (10:24-10:25) and the heatmap guard (12:05-12:08:01), so the code under test includes every contract change; numeric attestation rests with oracle owner",
    "GPUHeatmap guard (num_keypoints<=0) vs OptimizerManager nullptr-when-empty: divergence only reachable if Frame's setCurvatureHeatmaps invariants break (verified in sync in frame.cu:138-160); re-check if Frame gains a second heatmap writer",
    "0-keypoint frames now enter gpu_heatmaps_ as initialized (was: run abort) — vector stays frame-aligned; cost-function no-op on GetNumKeypoints()==0 is the documented contract",
    "SettingsPanel absent from the app qmldir (pre-existing; resolves via directory auto-discovery — verified with qmllint 6.7.2 and an app boot reaching GL init with zero type errors); harness components-qmldir lists it — cosmetic asymmetry",
    "renderer.qml QtQuick.Window import drop is Qt6-safe (Window resolves from QtQuick); render-smoke binary rebuilt after the change (12:09:10), visual confirmation pending xcb/GL",
    "FileDialogBridge writes real ~/.config/JointTrackAutoGPU/jtml_experimental.conf on every accept — any future headless exercise must redirect XDG_CONFIG_HOME",
    "App boots to GL init with no QML type-resolution errors, but no visual verification possible on this box; owner feedback rounds attest visuals"
  ],
  "testing_gaps": [
    "FileDialogBridge: only bridge with no test — needs XDG_CONFIG_HOME-redirected QSettings round-trip pin (purpose isolation, MRU cap 5, startDir -> remembered -> Home fallback chain)",
    "D3 run-state leg (AppBridge.cpp:84-93) untestable without a controller state seam — currently oracle/manual-visual",
    "Harness cannot drive the D3 refresh surface (fakes lack refreshTable; ListModel cannot emit modelReset) — QML re-read-after-reset pinned only in C++ (resets==1)",
    "Mid-edit cell destroyed by modelReset (concurrent drag while typing in open pose dialog): commit-on-destroy unverified",
    "FakeOptimizerBridge seed surface missing (hasSeedPose/setSeedPose/applySeedPose)",
    "qml_render_smoke + qml_parity_check need xcb/GL/GPU; binaries current but not re-run here"
  ]
}
```

**Verification performed (read-only):** all 38 Qt Quick Test functions pass (offscreen, Material style); experimental_pose_bridge_test passes 12/12 cases (133 assertions, incl. new D3/D4 pins); experimental_ml_bridge_test passes 11/11 (91 assertions, incl. the D4 wiring pin); qmllint gate passes; app boots to GL init with zero QML type-resolution errors. Parity/render-smoke binaries postdate every changed source but need GL/GPU this box lacks.