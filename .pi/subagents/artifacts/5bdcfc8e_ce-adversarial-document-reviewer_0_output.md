All verification is complete. I have grounded every major challenge in the actual code: the Connections inventory, the seed ownership, the run-lock state, the pose-cell commit path, the dataset-swap deferral, and the style pin. Here is the review.

```json
{
  "reviewer": "ce-adversarial-document-reviewer",
  "findings": [
    {
      "title": "Wiring inventory wrong: 6 Connections blocks, not 5",
      "severity": "P1",
      "section": "Context / High-Level Technical Design (wiring contract)",
      "why_it_matters": "U2's enforcement step will apply a wrong inventory: the plan names 'the 5 glue blocks in main.qml (417-461)', but the file has six Connections blocks (main.qml:419, 447, 461, 472, 493, 510), and the optimizerBridge/mlBridge/poseBridge glue at 472-516 falls outside the cited range entirely. The 419 studyBridge block alone feeds three surfaces (frame list via datasetChanged, viewport via scene relays, message dialog), and studyBridge has two blocks that must split across StudyPanel and the center-panel composition — which the stated rule 'No component creates a second Connections to a bridge the root already observes' forbids as written. Consequence: glue blocks are dropped or the single-owner rule is violated during extraction, producing exactly the scope/glue regressions the plan's top risk mitigation exists to prevent; the 'viewport id referenced from 4 glue blocks' claim is also short by one (five glue blocks reference it, six if the Connections-target at 447 counts).",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "suggested_fix": "Recount the glue in U2 and the wiring contract: six Connections blocks spanning main.qml:419-516 (two target studyBridge); restate the single-owner rule per signal surface (one owner per bridge signal, not per bridge) and enumerate the split — datasetChanged/message → StudyPanel, scene relays + selection + viewport-target → center-panel composition, optimizer/ml/pose relays → their surface owners.",
      "confidence": 100,
      "evidence": [
        "5 `Connections` blocks to the bridges (main.qml:417-461).",
        "the 5 glue blocks in main.qml (417-461) move to the component that owns the surface they feed (viewport glue stays in the center-panel composition; dialog glue in the dialog components). No component creates a second `Connections` to a bridge the root already observes.",
        "The viewport id is referenced from the toolbar, 4 glue blocks, and the ML strip — the center region is a component exposing `property alias viewport`, never `findChild`."
      ]
    },
    {
      "title": "Mid-edit scroll-away: pool reset vs editingFinished commit race unpinned",
      "severity": "P1",
      "section": "D8 / U5 (Pose-table virtualization)",
      "why_it_matters": "If the user scrolls the pose list while a cell is mid-edit, the delegate is pooled (focus loss fires editingFinished) while onPooled 'drops edit state' — the plan never pins which happens first. If the text reset precedes the commit handler, the captured tuple commits an empty or recycled-row value (a wrong-value write, the C1 class the plan calls its 'data-integrity core'); if the delegate is reused before editingFinished fires, the new row's displayed text commits to the old row's captured tuple. The plan's own U5 happy-path pin ('edit row 3 → scroll far away → scroll back → commit still lands on row 3 with the typed value') requires the typed value to survive pooling, in direct tension with 'onPooled (drop edit state)' — the design text and the pin test assert contradictory semantics, and no U5 scenario covers scroll-away-mid-edit.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "suggested_fix": "Decide and pin the mid-edit pool semantics before U5: either commit-on-pool (flush the edit before any onPooled reset — commit reads the live text before reset) or discard-on-pool (drop the edit with no commit and a validation notice), and align the U5 scenario text with the chosen semantic; add an explicit scroll-away-mid-edit case to the C1 pin test.",
      "confidence": 75,
      "evidence": [
        "Repeater→`ListView` with `reuseItems: true`, fixed row height, `onPooled`/`onReused` reset of cell state, commit contract from D2.",
        "reuseItems: true with `onPooled` (drop edit state) / `onReused` (re-bind stored value) — reset per the qt-qml delegate rules.",
        "Happy path: with a large fake model (e.g., 500 frames), only visible rows are instantiated (child-count assertion via `objectName` on the table). Edge case (C1 pin): edit row 3 → scroll far away → scroll back → commit still lands on row 3 with the typed value; no cross-row write.",
        "user types → text binding detached by editing (unavoidable QQC2 behavior)"
      ]
    },
    {
      "title": "D7 index-sync fires setCurrentFrame during dataset model swap",
      "severity": "P2",
      "section": "D7 / U3 (keyboard contract)",
      "why_it_matters": "D7 adds `onCurrentIndexChanged → setCurrentFrame`, a synchronous bridge write that fires whenever ListView resets currentIndex — which happens during dataset replacement, the exact 'clearDataset deletes the old model instance' swap the Qt.callLater deferral exists to outlast. A transient -1/0 index write mid-swap calls setCurrentFrame(-1) ('-1 = none'), cascading into updateSceneBackground/syncSessionState and emitting selectionChanged, which clears a pending ML seed (MlBridge::clearEstimate) and re-points the pose table at the wrong moment. The plan pins the deferral's survival but never checks the new index-sync path against the swap, so the 'list highlight + bridge can't diverge' claim holds only outside dataset loads.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "suggested_fix": "Guard the D7 index-sync during dataset transitions (suppress setCurrentFrame while a dataset swap is in flight, or keep the bridge as the frame source of truth until the deferred sync runs), and add a U2/U6 pin asserting that a dataset replace does not write -1/0 to the bridge.",
      "confidence": 75,
      "evidence": [
        "D7 | **Keyboard contract:** frame list wires `onCurrentIndexChanged → studyBridge.setCurrentFrame` (single source of truth; list highlight + bridge can't diverge)",
        "The `Qt.callLater` deferral in `onDatasetChanged` (main.qml:421-424) must survive extraction — it exists to outlast the list-model swap (`clearDataset` deletes the old model instance)."
      ]
    },
    {
      "title": "D4 seed invalidation misses copy/load pose-write paths",
      "severity": "P2",
      "section": "D4 / U3 (ML seed invalidation)",
      "why_it_matters": "D4's principle ('any manual pose write') is implemented only for viewer drags and pose-table edits, but copyPrevious/copyNext and loadPoseFile/loadKinematics also write poses to the current frame / primary model via SavePose with no selection change — so a pending seed survives and the next run() silently overrides the user's copied or loaded arrangement, the same I3 failure class. The plan-006 guard clears the seed only on selectionChanged (frame/model changes), so the U3 empirical check will confirm the gap; the fix must also route through OptimizerBridge::clearSeedPose — the pending seed lives there (setSeedPose/clearSeedPose), not in MlBridge, despite U3's wording 'clears the pending-seed state in MlBridge', and the U3 file list does not mention OptimizerBridge.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "suggested_fix": "Extend D4's invalidation triggers to copyPrevious/copyNext and loadPoseFile/loadKinematics writes landing on the seeded frame+model, route the invalidation through OptimizerBridge::clearSeedPose (the seed's actual owner; MlBridge already forwards there in clearEstimate), and pin each path in the extended experimental_pose_bridge / experimental_ml_bridge suites.",
      "confidence": 75,
      "evidence": [
        "D4 | **ML seed invalidation:** any manual pose write (viewer drag or pose-table edit) on the seeded frame+model drops the pending ML seed.",
        "ML-seed invalidation (I3): a manual pose write (viewer drag via `applyViewerPose`, or a pose-table edit) on the seeded frame+model drops the pending seed — verify against the plan-006 stale-seed guard first; if the run controller already drops stale seeds, the fix is a view-layer signal that clears the pending-seed state in MlBridge.",
        "Modify: `src/app/experimental/PoseCell.qml` (D2 contract), `src/app/experimental/PoseBridge.cpp/.h` (refresh owner D3), `src/app/experimental/MlBridge.cpp/.h` (seed invalidation D4, plus the `hasSegmentModel`-aware enablement state if not already exposed)"
      ]
    },
    {
      "title": "No pass-level success criteria; U7 attests with zero improvement",
      "severity": "P2",
      "section": "U7 Verification / overall plan",
      "why_it_matters": "Every unit's Verification checks that work was performed (report exists, files written, gates green), not that anything improved: U7 'passes' with zero measured improvement ('top hotspots are mapped to source lines' is satisfiable with an empty hotspot list and no before/after delta), and the plan states no pass-level success criteria, exit decision, or do-nothing baseline. For an 8-unit pass over an app labeled 'experimental' while the widgets app is untouched, completion attestation can succeed without demonstrating any value — the 'full pass' premise is never made falsifiable.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "suggested_fix": "Add pass-level success criteria and an outcome threshold for U7 (e.g., before/after profile delta on the top-3 hotspots, zero qmllint errors on app QML, all I1-I6/C1/C2 pins green), so completion attestation requires demonstrated improvement rather than process completion.",
      "confidence": 50,
      "evidence": [
        "**Verification:** Report exists in `profiler/reports/` + `docs/reviews/` summary; top hotspots are mapped to source lines; fixes landed only for view-layer hotspots; second standalone report written after fixes; headless suite + parity green.",
        "R7. Performance baseline via qmlprofiler on the 2D chrome (lists, dialogs, bindings; the VTK viewport is out of the profiler's scope) with the top hotspots fixed and re-documented."
      ]
    },
    {
      "title": "U6 harness omits Material.Dark; tests run under default style",
      "severity": "P3",
      "section": "U6 (Qt Quick Test coverage)",
      "why_it_matters": "The app pins its style on the root (Material.theme: Material.Dark / Material.accent: Theme.accent in main.qml), but the U6 targets import only QtQuick/Controls/Layouts and no QT_QUICK_CONTROLS_STYLE is specified for the test target, so components are exercised under the default style with none of the app's attached-property environment. Style-dependent geometry and behavior (implicit control sizes, focus visuals, light-vs-dark palette) can diverge from production, so the headless pins (keyboard contract, enabled-binding matrices) may pass against a shell that renders and sizes differently under Material.Dark — the 'no Material style import' claim removes the style, not the divergence.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "suggested_fix": "Set QT_QUICK_CONTROLS_STYLE=Material in the test/qml/ ctest environment (and account for the light/dark theme default in any visual assertions), or explicitly document the style divergence as accepted in U6.",
      "confidence": 50,
      "evidence": [
        "The target must NOT instantiate `QmlVtkRenderer`/VTK (headless rule); the tested components are verified VTK-free (Theme/PoseCell/SettingsPanel import only QtQuick/Controls/Layouts).",
        "keep the deliberate `Material` style import (the app pins Material Dark by design — comment it as such)"
      ]
    }
  ],
  "residual_risks": [
    "D3 refresh granularity (per-cell vs full reset) stays deferred to U3/U7; if run-completion refresh uses full reset, delegate churn on every run completion remains unmeasured until U7.",
    "The I1 stale-on-reopen premise still rests on the U3 empirical check; the U6 (f) freshness pin shape may change if the premise fails.",
    "U7's interactive profiling leg depends on the owner's display; the offscreen fallback covers only startup bindings, so scroll/dialog hotspots may go unmeasured if the owner leg slips.",
    "Any component that still reads root-context property globals without declaring a required property resolves fine in the app but fails loudly in tests; the extraction must not silently rely on context properties, or U6 fakes will diverge from the app's wiring."
  ],
  "deferred_questions": [
    "Mid-edit scroll-away semantics: commit-on-pool or discard-on-pool? (author decision required before U5)",
    "Should copy-prev/next and pose-file loads also invalidate the pending seed? (D4 scope decision)",
    "Where should the D7 index-sync guard live — StudyPanel, or a bridge-side loading flag? (implementation decision)",
    "What is the pass-level success criterion for U7's profile delta, if any? (author decision)"
  ]
}
```