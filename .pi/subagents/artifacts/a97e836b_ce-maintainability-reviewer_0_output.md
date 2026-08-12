Artifact written. Final compact report:

```json
{
  "reviewer": "maintainability",
  "findings": [
    {
      "title": "D5 'single run-lock source' is four per-panel copies plus two leftover raw `!optimizerBridge.running` spread sites (PosesDialog.qml:47, PosesTable.qml:142) — the exact spread the plan/review said causes lock escapes",
      "severity": "P3", "file": "src/app/experimental/main.qml", "line": 52,
      "confidence": 75, "autofix_class": "advisory", "owner": "review-fixer",
      "requires_verification": true, "pre_existing": false,
      "suggested_fix": "Convert the two raw sites to the per-panel derived-property pattern (or centralize by passing runLocked down); amend the plan's 'single root property' wording to match the delivered per-panel design."
    },
    {
      "title": "main.qml composition-root growth outran the plan's own ≤~300-line criterion (517 lines at @; 326 after U2, 478 after U4) and the plan was never amended; handoff still claims '1020 → ~230'",
      "severity": "P3", "file": "src/app/experimental/main.qml", "line": 264,
      "confidence": 75, "autofix_class": "advisory", "owner": "review-fixer",
      "requires_verification": false, "pre_existing": false,
      "suggested_fix": "Amend the plan's U2 verification or extract the ~140-line inline toolbar cluster (main.qml:264-470) into a Toolbar.qml component; the 4 dialogs are defensibly root-owned contracts."
    },
    {
      "title": "Redundant `dev_heatmap_ = 0;` in GPUHeatmap ctor's else branch — dead assignment introduced by the 0-keypoint guard hoist",
      "severity": "P3", "file": "src/compute/gpu_heatmaps.cu", "line": 42,
      "confidence": 100, "autofix_class": "safe_auto", "owner": "review-fixer",
      "requires_verification": false, "pre_existing": false,
      "suggested_fix": "Delete the else-branch `dev_heatmap_ = 0;` (line 12 already assigns unconditionally before the guard)."
    },
    {
      "title": "ViewportPanel I-03 guard (`if (!viewportItem)`) is unreachable and does not cover the failure mode it names — dead defensive code with a false premise",
      "severity": "P3", "file": "src/app/experimental/ViewportPanel.qml", "line": 135,
      "confidence": 75, "autofix_class": "manual", "owner": "review-fixer",
      "requires_verification": true, "pre_existing": false,
      "suggested_fix": "Remove it, or wire a real failure signal out of QmlVtkRenderer::initializeVTK (which currently never reports failure)."
    },
    {
      "title": "D1 injection alias layer: six `*Ref` aliases duplicate bridge names in root scope; documented convention, but `bridge: root.<bridge>` bindings would avoid the indirection",
      "severity": "P3", "file": "src/app/experimental/main.qml", "line": 40,
      "confidence": 50, "autofix_class": "advisory", "owner": "downstream-resolver",
      "requires_verification": false, "pre_existing": false,
      "suggested_fix": "Keep for now (documented); if the alias list grows, bind `root.appBridge` at use sites instead."
    },
    {
      "title": "renderer.qml hardcodes `#14161a` (== Theme.bg) — U2's 'zero hardcoded colors outside Theme.qml' verification has an undocumented exception",
      "severity": "P3", "file": "src/app/experimental/renderer.qml", "line": 17,
      "confidence": 50, "autofix_class": "manual", "owner": "review-fixer",
      "requires_verification": true, "pre_existing": true,
      "suggested_fix": "Add Theme to the smoke qrc and re-point, or document the exception in the plan."
    },
    {
      "title": "PosesTable.focusFirstCell reaches the first cell via positional child indexing (`row.children[1]`) — breaks silently if the delegate row is reordered",
      "severity": "P3", "file": "src/app/experimental/PosesTable.qml", "line": 261,
      "confidence": 50, "autofix_class": "advisory", "owner": "downstream-resolver",
      "requires_verification": false, "pre_existing": false,
      "suggested_fix": "Give the first cell an objectName or a delegate-level focus hook; the child-order contract is documented but unpinned."
    }
  ],
  "residual_risks": [
    "main.qml:234 empty onDilationBackgroundRequested handler (deliberate U7 placeholder, silently consumes the signal)",
    "OptimizerBridge orientationSymTrapUpdated emitted but never consumed in the QML layer (pre-existing dead signal on the experimental surface)",
    "configure-profiling task is a ~30-line near-copy of configure — flag drift risk; trailing whitespace after \"-S.\",",
    "FileDialogBridge free-form purpose strings as QSettings key suffixes — typo silently creates a new memory slot",
    "PosesDialog.qml:30 comment 'via root's open()/close()' is confusing (root is the Dialog itself)",
    "MlStrip baseName() splits only on '/' (Linux-only assumption, fine today)",
    "No gate enforces the main.qml line-count criterion — the drift had no tripwire"
  ],
  "testing_gaps": [
    "The D5 run-lock additions with the worst drift history — Camera/Model toggles (main.qml toolbar) + ViewportPanel lock — have NO headless pin (harness excludes main.qml/ViewportPanel; tst_StudyFlows covers only StudyPanel/MlStrip/RunBar/PosesDialog)",
    "No lint/test assertion fails on raw `!optimizerBridge.running` spreads outside RunBar — D5's 'prevent future drift' goal has no tripwire",
    "No pin on the delegate child-order contract focusFirstCell depends on",
    "renderer.qml palette duplicate is unverified by any gate"
  ]
}
```