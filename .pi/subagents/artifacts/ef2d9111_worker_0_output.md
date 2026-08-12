I've completed the full review of all 5 QML files plus traced the layout context (bridges, dialog sizing). Here are my findings.

## Layout & Anchoring Review — `src/app/experimental/`

### Findings (confidence ≥ 80)

**[D-1] Badge "pill" Rectangles collapse to 0 width in their RowLayouts — the pill background never renders** — confidence 90
- `main.qml` pose-dialog header (`Rectangle { Layout.preferredHeight: 14; radius: 7; color: poseBridge.dirty ? "#e5b567" : "#3a4a3d" }` + inner Label) and identical pattern in `SettingsPanel.qml` (~line 24).
- **Trace:** A plain `Rectangle` has `implicitWidth: 0` (Qt does not derive implicit size from children). In a `RowLayout`, an item with no `Layout.preferredWidth` is sized to its implicitWidth → **0**. The child `Label` with `anchors.centerIn: parent` centers on a 0-width parent, so the "● unsaved"/"saved" text renders but the colored pill behind it is invisible in both dialogs. The `Layout.preferredHeight: 14` only fixes the height.
- **Mitigation:** give the badge an explicit width (`Layout.preferredWidth: Math.max(label.implicitWidth + 12, 28)` on the Rectangle, or a `width` binding derived from the label).

**[D-2] Pose-table delegate RowLayouts use `Layout.fillWidth: true` inside a plain `Column` — a no-op; rows render at implicit width** — confidence 85
- `main.qml` ~line 358 (`Column { width: poseScroll.availableWidth }` → `Repeater` → `delegate: RowLayout { Layout.fillWidth: true ... }`).
- **Trace:** `Layout.*` attached properties are only honored when the direct parent is a Qt Quick Layout. The parent here is a plain `Column`, so `fillWidth` is ignored; each row is sized to its implicit width (≈ 44 + 6×78 + 5×4 spacing ≈ **532px** vs ~636px available). Rows sit left-aligned with a ~100px dead band on the right, and the table does not stretch to the scroll width. (The upcoming ListView virtualization in the improvement plan should use layout-aware delegate sizing.)
- **Mitigation:** bind each row's width to the scroll content width, or make the `Column` itself the width authority and size rows from it.

**[D-3] Reusable `PoseCell` hardcodes `width: 78; height: 26` for layout-managed consumption** — confidence 85
- `PoseCell.qml:26-27`, consumed inside the delegate RowLayout.
- **Trace:** a reusable component with explicit width/height prevents consumer resizing and is the same LAY-2 class as the lint-flagged header sites (which qmllint could not see because `PoseCell` is defined in its own file). Alignment with the 78px column headers currently depends on Qt's implicitWidth-fallback for explicit widths (see I-1). The ML strip already shows the correct pattern: `implicitWidth: 40` on the Fem/Tib buttons.
- **Mitigation:** convert to `implicitWidth`/`implicitHeight` (or `Layout.preferredWidth` at the call sites).

**[D-4] SettingsPanel scrollbar gutter `width: root.width - 18` is a magic number that clips or wastes space in both scrollbar states** — confidence 90
- `SettingsPanel.qml` ~line 60 (`ColumnLayout { width: root.width - 18  // scrollbar gutter }`).
- **Trace:** the ScrollView's content width is `ScrollView.width - scrollbarWidth` when the vertical scrollbar is visible (it is — the 3-stage form overflows 680px). With `root.width` = dialog content width and the inner `anchors.margins: 6`, the Column ends up ~4-10px *wider* than the ScrollView's `availableWidth` when the scrollbar shows → the right edges of the SpinBox fields are clipped; when the scrollbar is hidden, an 18px dead gutter remains. The pose dialog already uses the correct mechanism (`width: poseScroll.availableWidth`).
- **Mitigation:** bind the Column width to the ScrollView's `availableWidth` (e.g. via an id'd ScrollView), not a hardcoded gutter.

### Investigation targets (60–79, human verification needed)

**[I-1]** (70) Header↔cell alignment in the pose table relies on Qt's implicitWidth-fallback for explicitly-set `width:` on layout-managed Labels/PoseCells — qmllint flags this as "undefined behavior". The columns *appear* aligned today, but the mechanism is Qt-version-dependent. Verify visually after D-3's fix lands and confirm the 44/78px grid still lines up with `Layout.preferredWidth`.
**[I-2]** (68) Scrollbar-gutter inconsistency: the column-header RowLayout spans the full dialog ColumnLayout width while the table Column spans `availableWidth` — when the vertical scrollbar appears, the header is ~scrollbar-width wider than the table (last column's right edge shifts). Minor, but verify at large frame counts.
**[I-3]** (65) Left-panel frame/model ListViews share `Layout.fillHeight: true` 1:1 — a 200-frame study with 2 models gives the model list a half-empty viewport. Consider `Layout.fillHeight` weights or `Layout.preferredHeight` based on content.
**[I-4]** (62) `Column { width: poseScroll.availableWidth }` re-evaluates when the scrollbar toggles (content height crosses the viewport boundary) — potential relayout jitter. Verify with a large frame count.
**[I-5]** (60) Visibility coupling: when the pose table is hidden (no frames/primary model), the column-header row and the actions row remain visible above the "No frames loaded" empty label — likely intended-by-default but visually odd; not a layout mechanics issue (state/UX territory).

### No findings in these categories
- **Anchoring to `visible: false` items:** none found (placeholder, readout, ML labels, and scroll region are all anchored to always-visible parents; children of hidden items are fine).
- **Anchors across unrelated visual branches:** none found — every `anchors.*` reference is parent-relative within the same subtree.
- **`anchors` + `Layout.*` mixed on the same item:** none found (all anchored RowLayouts/ColumnLayouts sit inside plain Rectangles, which is legal).
- **The two fillWidth `Item` spacers** in the pose-dialog actions row and toolbar are the correct pattern.