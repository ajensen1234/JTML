## Context Analyzer Report

**(a) Track + problem_type:** `knowledge` — `convention`

**(b) Category directory:** `docs/solutions/conventions/`

**(c) Component enum:** `frontend_stimulus` (repo precedent — the existing QML entries, `jtml-qml-view-testing-2026-08-12.md` and the 08-11 ui-bugs entry, both map the QML front-end to this enum; no QML-specific enum exists in the schema)

**(d) Module:** `jtml_view` (matches the existing QML entries)

**(e) Severity:** `medium` (framework gotchas that each cost a debugging session and silently corrupt state — pose commits, frame selection — but all have workarounds and are now pinned by the harness)

**(f) Suggested filename:** `qml-listview-delegate-required-property-patterns-2026-08-12.md`

**(g) YAML frontmatter skeleton:**

```yaml
---
title: "Qt 6.7 ListView delegate patterns: pooling, required properties, and keyboard contracts"
date: 2026-08-12
category: conventions
module: jtml_view
problem_type: convention
component: frontend_stimulus
severity: medium
applies_when:
  - "writing QML ListView delegates that use reuseItems/pooling"
  - "declaring required properties on delegates (incl. required property var model)"
  - "handling input inside nested delegate items (MouseArea, Keys)"
  - "implementing per-cell keyboard navigation contracts"
tags: [qml, qtquick, listview, delegate, pooling, required-property, keyboard, qmltest]
---
```

**(h) Doc title:** `Qt 6.7 ListView delegate patterns: pooling, required properties, and keyboard contracts`

**Overlap note (feeds the orchestrator's update-vs-create decision):** the existing `docs/solutions/conventions/jtml-qml-view-testing-2026-08-12.md` already documents three of the four behaviors (#1 ListView.view-null-in-nested, #2 pooling-retains-focus + attached-signals, and the FINAL-collision) in its "QML gotchas confirmed on this stack" section. **Missing from it:** #3 (required-property delegates leave the implicit `model` context object stale on pool reuse — the newest finding, test-proven via the diagnostic `frameIndex=3 / model.x=0`) and #4 (Keys.onPressed on the control root captures arrows/Escape before the internal editor — the D7 contract's enabler). **Verdict:** HIGH overlap on the problem space but with two genuinely new, additive findings — the orchestrator should UPDATE the existing conventions entry (append the two new gotchas + add `last_updated: 2026-08-12` to its frontmatter) rather than create a duplicate, OR create the focused delegate-patterns doc and cross-link. If updating, the existing entry's frontmatter (module: jtml_view, problem_type: convention, severity: medium) is consistent with this classification.

**Evidence grounded in the repo:**
- `src/app/experimental/PosesTable.qml:164-165` — `required property int frameIndex` + `required property var model` (the #3 fix, live)
- `src/app/experimental/PoseCell.qml:64` (Keys.onPressed), `:92` (Component.onDestruction), `:139` (commitIfEditing) — #4 + the flush mechanisms
- `src/app/experimental/StudyPanel.qml` — the #1 fix (`frameList.currentIndex = index`)
- `test/qml/tst_PosesTable.qml` / `tst_PoseCell.qml` — the pins (incl. `test_rejectedCommitSurvivesRecycle` which caught #3)
- Existing conventions entry read in full (overlap assessment above)