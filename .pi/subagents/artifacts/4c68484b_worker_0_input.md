# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement plan 007 U4 for the JTML repo (cwd: /home/ajj/repo/uf/JTML). READ first: docs/plans/2026-08-12-007-refactor-qml-experimental-improvement-pass-plan.md (U4 section — including the owner-confirmed visual-composition pass), docs/reviews/2026-08-12-qml-experimental-review.md (D-09g debug readout, I-07 panel balance, I-11 visibility coupling, I-12 redundant re-click, I-13 dirty-close guard), and the current src/app/experimental/ components (U2 extracted, U3 fixed state gaps).

U4 GOAL: UI/UX audit + polish + the owner-confirmed visual-composition pass ('make it look fresh'). The qt-ui-design skill principles apply: CTA hierarchy, proximity/similarity grouping, spacing rhythm, dialog ergonomics, empty-state language, one-accent discipline.

CONTEXT (already known — do not re-ask): desktop tool, ~60 cm viewing distance, mouse+keyboard, DPR~2 box, dark theme (Material Dark + Theme overlay), English only, no RTL. TypeScale is PINNED: caption 12 / label 14 / body 16 / h2 18 (Theme.qml already has them).

SCOPE: only src/app/experimental/** (QML + Theme.qml + qmldir/qrc if needed). NO C++ changes. Do not touch renderer.qml. Do not run git/jj. Do not stage/commit. Run pixi run build + pixi run test at the end (must stay green).

WORK ITEMS:
1. TYPOGRAPHY COMPLETION: the U2 re-point applied tokens, but verify no raw font.pixelSize remains outside Theme.qml (grep); bump any remaining 10/11px sites to the nearest role. Max 3-4 type sizes per screen.
2. KEYBOARD COMPLETION (U3 wired the frame/model lists): full Tab order toolbar -> lists -> run bar; visible focus indicator on list rows + toolbar buttons (U3 added rows; add buttons if missing); DIALOG FOCUS MANAGEMENT: settings dialog opens with focus on the first field; Poses dialog opens with focus on the first pose cell; closing (Esc/button) returns focus to the opener control (forceActiveFocus on open/close). Escape already closes dialogs (CloseOnEscape) — keep.
3. ACCESSIBILITY: Accessible.role + Accessible.name on the custom list rows and the mode-toggle buttons; Accessible.ignored on decorative items (spacers, the readout if kept).
4. CONTRAST: compute contrast ratios for every Theme token pair actually used (text color vs its background); the audit suspects Theme.fgDim (#6b7280) on Theme.bg (#14161a) is ~4.0:1 — if any pair used for body-size text is < 4.5:1, bump the token. Add a short comment in Theme.qml with the measured ratios.
5. HIT TARGETS: list rows 22px -> 24px (StudyPanel); verify buttons/spinboxes are comfortable; the 240px left column may need ~260px if label-14 text truncates — decide by reading the layout (verify at the pinned type scale; do NOT break the ML strip's alignment).
6. DIRTY BADGE + CLOSE GUARD (I-13 + M6 fold-in): the dialogs show the dirty pill (U2 fixed the pill); ADD the owner-confirmed dirty-close guard: when PosesDialog or the settings Dialog has dirty state (poseBridge.dirty / settingsBridge.dirty) and the user closes it via press-outside or Esc, show a confirm ('Discard unsaved changes?') — Yes discards, No keeps it open. The Run-button close stays unconditional. Also add the SHELL-level dirty indicator: a small badge in the toolbar (or window title area) aggregating settingsBridge.dirty || poseBridge.dirty using the Theme badge tokens.
7. DEBUG READOUT (D-09g): the viewport poseReadout overlay is plan-005 residue — REMOVE it from ViewportPanel (the Poses dialog is the real surface).
8. VISUAL COMPOSITION PASS (the 'look fresh' work — placement, not just tokens):
   - Toolbar: group the load actions (Calibration/Images/Models) as one cluster with tighter spacing; separate the status label + Interact toggle + dialog openers into distinct visual groups (spacing separators); one visually primary action per cluster (the plan says CTA hierarchy — e.g. load cluster's primary = Images; the Poses/optimizer openers stay secondary).
   - MlStrip: the 8-ish rows of tiny controls — align picker buttons + path labels on a shared baseline; Segment/Estimate as one emphasized action pair; the Black-sil./implant/view rows as one settings sub-group with a divider; consistent label alignment.
   - Spacing rhythm: adopt a consistent 4/8px grid for panel margins and inter-control gaps (currently mixed 4/6/8); consistent list-row padding; panel radius consistent.
   - Dialogs: Save/Reset + the Poses action row get primary/secondary distinction; affirmative rightmost; dialog padding consistent with the grid.
   - Empty states: the viewport placeholder, the Poses dialog 'No frames loaded', the 'Select a model' hint, and the ML status/hint labels share one visual language (centered caption text, muted color — already tokenized; make the styling consistent).
   - One-accent discipline: Theme.accent only on interactive elements; the viewport lock overlay (U3) uses Theme.overlayDim — keep it non-accent.
   - Result bar: the app must read as a designed dark tool — consistent panel treatment, no stray default-style widgets.
9. RUN-LOCK visual check: U3 added the overlay; verify it covers the viewport and shows during a run (bindings correct).
10. BUTTONGROUP re-click (I-12, trivial): onClicked re-fires on the already-active button — if the current single-source pattern makes this harmless (idempotent setter), leave a comment; do not add guard complexity.

CONSTRAINTS:
- Behavior-neutral where possible; the only behavior changes are the owner-approved ones (dirty-close guard, shell badge, debug readout removal, focus management).
- Do not restructure components (that was U2); this is placement/polish edits within them.
- Do not run the GUI (no display) — but DO run pixi run build and pixi run test and keep 47/47 green.
- Keep the qmllint gate green: run .pixi/envs/default/lib/qt6/bin/qmllint --json to check no NEW non-accepted warnings.
- Report: per-item summary, the contrast table with measured ratios, the composition checklist result, test results, and a precise 'what the owner must visually verify under xcb' list.

IMPORTANT: the owner will do a visual sign-off under xcb after this unit. Make the composition changes confident and complete — they are the visible payoff of the whole pass. Do not leave TODOs.

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```