I've now read the full plan and grounded the design-lens review in the actual code (main.qml 1016 lines, PoseCell.qml, and grep checks for focus/keyboard handling). Key verifications: (1) the current pose table is 6 `TextField`s per row in a `Repeater`/`Column`/`ScrollView` with zero keyboard handling anywhere in the QML files; (2) the viewport (`QmlVtkRenderer`) has no `enabled` lock or lock visual today; (3) empty states (viewport placeholder, pose-dialog "No frames loaded" label, ML status labels) already exist; (4) the run-lock matrix in D5 is accurate against the code; (5) the DPR-2/contrast/hit-target items are well-covered.

```json
{
  "reviewer": "ce-design-lens-reviewer",
  "findings": [
    {
      "title": "Pose-table keyboard contract missing; virtualization strands off-screen rows",
      "severity": "P2",
      "section": "Key Technical Decisions (D7) / Implementation Units → U4, U5, U6",
      "why_it_matters": "A keyboard user editing the 500-row pose table can only Tab through the currently visible rows: with `reuseItems: true` the ListView never instantiates off-screen delegates, so the default QQC2 focus chain physically cannot reach rows below the viewport — directly contradicting U4's 'no keyboard traps' claim and making U4's 'Tab reaches every interactive control in order' test unpassable for the table. D7 specifies keyboard behavior for the frame and model lists only; the pose table (6 TextFields × N rows, the tool's primary data-entry surface) has no keyboard design: no arrow-key cell navigation, no Enter-commit/Esc-revert contract, no spec for how Tab enters/exits the table. Implementers hit this concretely in U5 (focus chain with pooled delegates) and U6 (writing the keyboard test) and must invent table-level keyboard semantics unprompted.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "suggested_fix": "Add a pose-table keyboard decision to D7/U4 — e.g., the ListView is a single tab stop with KeyNavigation (Up/Down between rows, Left/Right between cells, Enter commits, Esc reverts), or explicitly designate the table mouse-first with a documented Tab in/out behavior; then scope U6's keyboard test to the visible page of cells rather than 'every interactive control'.",
      "confidence": 75,
      "evidence": [
        "**Keyboard:** full Tab order (toolbar → lists → run bar), visible focus indicator on list rows and toolbar buttons, Escape closes dialogs (already `CloseOnEscape`), no keyboard traps.",
        "**Keyboard contract:** frame list wires `onCurrentIndexChanged → studyBridge.setCurrentFrame` (single source of truth; list highlight + bridge can't diverge); model list toggles on Space/Enter; rows are focusable (`activeFocusOnTab`), ListView handles Up/Down; visible focus indicator bound to `activeFocus`.",
        "Replace the `Repeater`-in-`Column`-in-`ScrollView` with a `ListView` (fixed `cellHeight`/row height, `boundsBehavior` default) inside the dialog; `reuseItems: true` with `onPooled` (drop edit state) / `onReused` (re-bind stored value)",
        "Edge case: Tab reaches every interactive control in order; focus is visible on the focused row/button (manual-visual + U6 keyboard test).",
        "Happy path: with a large fake model (e.g., 500 frames), only visible rows are instantiated (child-count assertion via `objectName` on the table)."
      ]
    },
    {
      "title": "Locked viewport during runs has no specified visual state",
      "severity": "P2",
      "section": "Key Technical Decisions (D5) / Implementation Units → U3, U4",
      "why_it_matters": "During a run the VTK viewport stays live (the glue forwards `onPoseUpdated`/`onFrameOptimized` → `viewport.updatePose`, so the scene visibly changes and looks fully interactive) yet D5's `enabled: false` silently swallows drags — `enabled` on a custom QQuickItem changes no rendering, so no Material-style disabled graying applies to the viewport, and the only run signal is the bottom progress bar. The plan itself labels mid-run drags 'a semantic clobber' (the same silent-failure class D4 exists to prevent) but specifies no locked-state affordance on the surface where the user's attention sits, and U4's audit list ('color is never the sole state carrier') does not include the run-lock state. A user who habitually drags the model during a multi-minute run gets dead input on a live-looking scene with no explanation.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "suggested_fix": "Specify a run-lock visual on the viewport in D5/U4 — e.g., a 'Running — interaction locked' overlay or dim on the renderer while `optimizerBridge.running` — and add it to the U4 audit checklist and the U3/U6 I4 pins, so the disabled state is visible where the user interacts, not only in the bottom bar.",
      "confidence": 75,
      "evidence": [
        "**Full run-lock matrix:** extend the existing `!optimizerBridge.running` lock to the Black-silhouette checkbox, the Fem/Tib buttons, and viewport interaction (`enabled: false` on the renderer — blocks mouse events).",
        "Completes the DisableAll mirror (I4); drags during a run are a semantic clobber.",
        "**Contrast:** verify every text/background pair ≥ 4.5:1 (audit suspects `Theme.fgDim` ≈ 4.0:1 on `Theme.bg` — bump if confirmed); color is never the sole state carrier (the dirty badge already pairs ● + text — keep).",
        "Connections { target: optimizerBridge ... function onPoseUpdated(modelIndex) { viewport.updatePose(modelIndex) } function onFrameOptimized(frameIndex, modelIndex) { viewport.updatePose(modelIndex) }"
      ]
    },
    {
      "title": "Dialog open/close focus management unspecified in keyboard contract",
      "severity": "P3",
      "section": "Implementation Units → U4 (Keyboard) / High-Level Technical Design (dialog ownership)",
      "why_it_matters": "The plan defines Tab order and CloseOnEscape for the two custom dialogs but never says where focus lands when Poses…/Optimizer Settings… opens (first field? first pose cell?) or returns after Esc (to the opener button?). QQC2 Dialog does not reliably place focus on a content control by default, so a keyboard-first walkthrough under R3 depends on an unstated decision. Implementers will either sprinkle ad-hoc `forceActiveFocus` calls inconsistently across the two dialogs or ship default focus behavior that the U6 keyboard test never checks.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "suggested_fix": "Add dialog focus management to U4: initial focus target on open (e.g., first field in SettingsPanel, first row/cell in PosesDialog) and focus return to the opener on close; add it as a U6 keyboard scenario.",
      "confidence": 50,
      "evidence": [
        "**Keyboard:** full Tab order (toolbar → lists → run bar), visible focus indicator on list rows and toolbar buttons, Escape closes dialogs (already `CloseOnEscape`), no keyboard traps.",
        "The Run button closes both edit dialogs (main.qml:980-981) — dialogs expose `open()`/`close()` or stay root-owned; the run bar does not reach into dialog internals."
      ]
    },
    {
      "title": "TypeScale roles ambiguous; U2 re-points before U4 pins values",
      "severity": "P3",
      "section": "Implementation Units → U4 (Typography) / Deferred to Implementation / U2",
      "why_it_matters": "The recommended modular scale lists two candidate values for label (13/14) and h2 (18/21), and the concrete values are explicitly deferred to the U4 audit — yet U2, which runs first, must already re-point every `font.pixelSize` site to Theme tokens (a major-second from base 16 yields 12.6/14.2, so 13 vs 14 and 18 vs 21 are real rounding decisions, not cosmetic). U2 extraction therefore either freezes provisional values that U4 later changes (re-touching every extracted component) or stalls the token-first rule (D11). Nothing breaks, but the interleaving churns U2/U4 and leaves the implementer of U2 to guess which value a role means.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "suggested_fix": "Pin each TypeScale role to exactly one value in the plan (resolve the 13/14 and 18/21 rounding, e.g. label 14, h2 18) and sequence the concrete values into U2's token-first re-point so U4 only verifies rather than re-specifies.",
      "confidence": 50,
      "evidence": [
        "**Typography:** define a modular scale in `Theme.qml` (recommended: base 16 body, compact major-second 1.125 for this dense data tool — caption 12, label 13/14, body 16, h2 18/21; max 3-4 roles per screen).",
        "**Concrete TypeScale values** (base/ratio) — the U4 audit's contrast pass decides; the plan fixes the role set, not the numbers.",
        "Extraction order (token-first, D11): extend Theme → re-point SettingsPanel and main.qml color/`font.pixelSize` sites → extract components in dependency order"
      ]
    }
  ],
  "residual_risks": [
    "Pose-table keyboard navigation and the viewport run-lock visual are both implicitly deferred to implementation by omission; if the owner accepts mouse-first table editing and progress-bar-only run feedback, R3/U4 wording ('no keyboard traps', 'Tab reaches every interactive control') must be rescoped explicitly in the plan.",
    "U2 will ship TypeScale roles with provisional values unless U4's concrete values are pulled forward; expect one re-point pass across all extracted components.",
    "Accessibility-role granularity (per-row vs per-list) is an owned deferral to U4, but tst_Theme/tst_StudyFlows assertions written in U6 depend on that decision landing first — if U4 slips, U6 role assertions are written blind.",
    "Empty-state coverage was verified present in code (viewport placeholder, pose-dialog empty labels, ML status/hint labels) and needs no plan work; the frame/model list count headers ('Frames (0)') are the only thin spot, accepted."
  ],
  "deferred_questions": [
    "Should the pose table get full keyboard cell navigation (arrows between cells, Enter commit, Esc revert) or an explicit mouse-first designation with documented Tab in/out? (decide in D7/U4)",
    "Does the viewport run-lock need a visible overlay/dim, or is the bottom progress bar sufficient feedback for the I4 lock? (owner call, then pin in U3/U6)",
    "Dialog open/close focus management: rely on QQC2 Dialog defaults or an explicit forceActiveFocus contract for both custom dialogs? (U4)",
    "TypeScale rounding: label 13 or 14, h2 18 or 21 — pin before U2's re-point so extraction and typography don't interleave twice."
  ]
}
```