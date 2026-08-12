Artifact written. Final compact report:

## Agent-Native Architecture Review — Plan 007 QML pass

**Summary.** Qt 6.7.2 + VTK experimental QML front-end. Every user action maps to a Q_INVOKABLE/Q_PROPERTY bridge primitive (loadImages/loadModels/setCurrentFrame/run/stop/setPoseValue/applyViewerPose…), components take injected `required property` bridges (D1) with objectName testability hooks, and both new gates pass locally (`jtml.qml_lint` 0.42s, `jtml.qml_view` 3.42s, 28 test functions). Assigned checks: FileDialogBridge's new 4-arg signature matches both QML call sites (main.qml:69-74, 85-90 pass `"images"`/`"models"`); the modal-dialog path is human-only sugar over scriptable primitives (parity holds); the lint ctest has a real CI-robustness hole (silent unregistration when no Qt6 qmllint is resolvable). The main functional gap: D7's owner-confirmed pose-table keyboard cell navigation is unimplemented and untested despite U4/U6 marked done.

### Capability Map (must/should-priority actions)

| UI Action | Location | Agent Tool | Priority | Status |
|---|---|---|---|---|
| Load calibration/images/models | main.qml:59-90 | studyBridge.loadCalibration/loadImages/loadModels | must | OK |
| Replace dataset | main.qml:106-120 | clearDataset + loadImages | must | OK (root-only, untested) |
| Frame select | StudyPanel.qml:87-92 | studyBridge.setCurrentFrame | must | OK |
| Model multi-select | StudyPanel.qml:159-170 | toggleModelSelected | must | OK |
| Segment/Estimate | MlStrip.qml | mlBridge.segmentCurrentFrame/estimateCurrentFrame | must | OK |
| Settings fields/save/reset | SettingsPanel.qml | settingsBridge.* props + save()/reset() | must | OK |
| Run/Stop | RunBar.qml:44-56 | optimizerBridge.run()/stop() | must | OK |
| Pose cell commit / copy / file IO | PoseCell/PosesDialog | poseBridge.setPoseValue/copyPrevious/savePoseFile/… | must | OK |
| Camera/Model mode | main.qml:235-262 | viewport.setInteractionMode (Q_INVOKABLE + Q_PROPERTY) | should | OK |
| ML seed observability | OptimizerBridge.h | hasSeedPose() (D4 additive read) | should | OK |
| Pose-table keyboard nav | (D7) | — | should | **MISSING (P2)** |

No LLM agent runtime exists in this app; parity is assessed as scriptable surface (Q_INVOKABLE/Q_PROPERTY + objectName), which is fully present.

```json
{
  "reviewer": "ce-agent-native-reviewer",
  "findings": [
    {"title": "D7 pose-table keyboard cell navigation (Left/Right cells, Up/Down rows, Esc revert) not implemented and untested, despite U4/U6 marked done", "severity": "P2", "file": "src/app/experimental/PoseCell.qml", "line": 1, "confidence": 75, "autofix_class": "manual", "owner": "downstream-resolver", "requires_verification": true, "pre_existing": false, "suggested_fix": "Implement owner-confirmed D7: Keys.onPressed per cell (Left/Right between the 6 cells, Up/Down rows via captured frameRow, Esc reverts via resetDisplay()); Enter already commits via editingFinished. Add tst_PosesTable keyboard pins per plan U6 scenario (e)."},
    {"title": "jtml.qml_lint ctest silently unregistered when no Qt6 qmllint resolvable — CI gate can vanish with no trace", "severity": "P2", "file": "test/CMakeLists.txt", "line": 1459, "confidence": 75, "autofix_class": "gated_auto", "owner": "review-fixer", "requires_verification": false, "pre_existing": false, "suggested_fix": "message(WARNING) on empty QML_LINT_QT6 at configure; or register always with SKIP_RETURN_CODE; probe Qt6_DIR//usr/lib/qt6/bin before giving up; add version sanity check."},
    {"title": "FileDialogBridge sidebar MRU replaces platform default sidebar entries once a purpose has history", "severity": "P3", "file": "src/app/experimental/FileDialogBridge.cpp", "line": 44, "confidence": 75, "autofix_class": "manual", "owner": "human", "requires_verification": true, "pre_existing": false, "suggested_fix": "Compose sidebar URLs as MRU + QStandardPaths home/desktop/documents places, or confirm at runtime and document."},
    {"title": "FileDialogBridge::getOpenFileNames is blocking modal exec() — scripted flows must use bridge primitives (which exist; advisory)", "severity": "P3", "file": "src/app/experimental/FileDialogBridge.cpp", "line": 35, "confidence": 75, "autofix_class": "advisory", "owner": "downstream-resolver", "requires_verification": false, "pre_existing": true, "suggested_fix": "No change needed for parity; document loadImages/loadModels as the scripted path."},
    {"title": "qml_lint.cmake allowlist comment stale: 'unqualified' removal path (U3 property injection) did not happen — main.qml still context-property coupled", "severity": "P3", "file": "test/qml_lint.cmake", "line": 26, "confidence": 75, "autofix_class": "safe_auto", "owner": "review-fixer", "requires_verification": false, "pre_existing": false, "suggested_fix": "Correct the comment: U6 D1 injected components only; root still uses context properties, entry must stay."},
    {"title": "qmllint gate lints only src/app/experimental — test/qml fakes + tst files not linted", "severity": "P3", "file": "test/qml_lint.cmake", "line": 38, "confidence": 75, "autofix_class": "gated_auto", "owner": "review-fixer", "requires_verification": false, "pre_existing": false, "suggested_fix": "Add test/qml glob (own accepted set) to the gate."},
    {"title": "Handoff claims 38 QML test functions; 28 delivered; plan U6 scenarios (a)/(d)/(e) (load-ordering, replace-confirm, run-close, re-run-after-Completed, pose-table keyboard) uncovered — main.qml excluded from harness", "severity": "P3", "file": "docs/handoff-2026-08-12-qml-experimental-improvement.md", "line": 52, "confidence": 75, "autofix_class": "manual", "owner": "downstream-resolver", "requires_verification": false, "pre_existing": false, "suggested_fix": "Correct count to 28 or add the missing tests; state explicitly that composition-root flows are manual-visual-only."},
    {"title": "No runtime version check on resolved qmllint binary (bounded: wrong binary fails loudly via missing JSON/FATAL)", "severity": "P3", "file": "test/qml_lint.cmake", "line": 48, "confidence": 50, "autofix_class": "advisory", "owner": "review-fixer", "requires_verification": false, "pre_existing": false, "suggested_fix": "Optional: qmllint --version gate at configure time."}
  ],
  "residual_risks": [
    "Composition-root flows (calibration-first, replace-confirm, run-close, discard chain) live only in main.qml and rely on manual-visual verification.",
    "Commit-on-pool correctness depends on Qt 6.7 ListView pooling semantics; exotic sequences (edit + run start, dialog destroyed mid-edit) unpinned.",
    "FileDialogBridge QSettings interaction and sidebar-replacement side effect need runtime confirmation on the target box.",
    "Run-lock is UI-binding enforced; bridge-level backstops covered by C++ unit tests, not QML tests.",
    "No LLM agent runtime exists; if one is added later, the system-prompt/context-injection layer does not exist yet."
  ],
  "testing_gaps": [
    "No pose-table keyboard tests (D7 unimplemented; plan U6 (e) uncovered).",
    "No composition-root tests: calibration-first messages, replace-confirm Yes/No/Esc, partial-load, run-closes-dialogs, re-run after Completed, discard-confirm reopen (plan U6 (a)/(d)).",
    "FileDialogBridge (4-arg signature, per-purpose dir memory, MRU) has no unit or QML test; signature match verified only by the two main.qml call sites.",
    "test/qml fakes + tst files are outside the qmllint gate.",
    "Nothing asserts jtml.qml_lint registration at configure time (silent-drop environment not detected)."
  ]
}
```

**Score: 20/20 high-priority user actions have scriptable bridge equivalents; 0 Critical, 2 Warning, 6 Observation findings. Verdict: NEEDS WORK (D7 keyboard contract + lint-gate registration robustness).**