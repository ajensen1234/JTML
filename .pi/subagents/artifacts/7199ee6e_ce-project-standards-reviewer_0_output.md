Artifact written. Final compact output:

```json
{
  "reviewer": "ce-project-standards-reviewer",
  "findings": [
    {
      "title": "Plan 007 D7 pose-table keyboard contract unimplemented: no cell navigation (Left/Right/Up/Down/Enter/Esc), no single-tab-stop, no U6 pin",
      "severity": "P2",
      "file": "src/app/experimental/PosesTable.qml",
      "line": 133,
      "confidence": 75,
      "autofix_class": "manual",
      "owner": "downstream-resolver",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Implement the D7 keyboard contract on PosesTable/PoseCell (single tab stop; Left/Right between cells, Up/Down rows, Enter commits, Esc reverts) and pin it in test/qml/tst_PosesTable.qml per plan U6 scenario (e)."
    },
    {
      "title": "Plan 007 U6 load-ordering scenario pins undelivered: no calibration-first, replace-confirm, or D10-merge tests in tst_StudyFlows.qml",
      "severity": "P3",
      "file": "test/qml/tst_StudyFlows.qml",
      "line": 1,
      "confidence": 50,
      "autofix_class": "manual",
      "owner": "downstream-resolver",
      "requires_verification": true,
      "pre_existing": false,
      "suggested_fix": "Componentize the root-owned load flows so the harness can pin them, or record the deliberate cut in the plan/handoff."
    },
    {
      "title": "Handoff line-count claim stale: 'main.qml 1020 → ~230 lines' — the final file is 517 lines",
      "severity": "P3",
      "file": "docs/handoff-2026-08-12-qml-experimental-improvement.md",
      "line": 6,
      "confidence": 50,
      "autofix_class": "advisory",
      "owner": "human",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "Correct the handoff/commit figure to the final ~517 lines or qualify it as the U2 intermediate state."
    },
    {
      "title": "AGENTS.md 'Test suite' section not updated for the new test/qml/ directory and Qt Quick Test harness",
      "severity": "P3",
      "file": "AGENTS.md",
      "line": 28,
      "confidence": 50,
      "autofix_class": "advisory",
      "owner": "human",
      "requires_verification": false,
      "pre_existing": false,
      "suggested_fix": "Add test/qml/ to the layout enumeration and note the QuickTest harness + jtml.qml_lint gate convention beside the existing QtTest/Catch2 bullet."
    }
  ],
  "residual_risks": [
    "Handoff '48/48 headless' claim vs 39 headless tests in the current .build cache — unverifiable here (stale cache possible).",
    "jtml.qml_lint silently unregisters if no Qt6 qmllint binary is found (CONDA_PREFIX or .pixi) — soft gate.",
    "Heatmap guard (gpu_heatmaps.cu/gpu_metrics.cu/optimizer_manager.cpp) has no automated pin — GPU/oracle territory; verification is the owner's manual app run; parity re-run is the ~1h oracle gate.",
    "QmlVtkRenderer scene-index drag fix unpinned (oracle/manual-visual).",
    "Plan R9 scope breach (src/compute, src/coordinator) is owner-requested and documented in the handoff.",
    "renderer.qml:17 hardcoded '#14161a' — pre-existing, oracle smoke scene, rule scope ambiguous (suppressed)."
  ],
  "testing_gaps": [
    "No test pins pose-table cell keyboard navigation (plan U6 scenario (e)).",
    "No automated coverage for main.qml-root-owned flows (calibration-first guard, replace-confirm Yes/No/Esc, D10 merge, run-closes-dialogs).",
    "No headless test drives the heatmap-less optimizer path or the renderer scene-index pose reporting."
  ]
}
```