# QML Code Review Report — Experimental Front-End (plan 007 U1)

**Scope**: files: `src/app/experimental/main.qml`, `SettingsPanel.qml`, `PoseCell.qml`, `Theme.qml`, `renderer.qml`
**Files reviewed**: 5
**Issues found**: 63 (41 lint, 9 deep-analysis, 13 investigation targets)
**qmllint**: ran — Qt 6.7.2 binary (`$CONDA_PREFIX/lib/qt6/bin/qmllint`); env-level Qt5/Qt6 module-ambiguity noise excluded

---

## Lint findings (qt_qml_lint.py — deterministic, 41 findings)

### [L-01] PoseCell imperative text assignment kills the storedValue binding
- **File**: `src/app/experimental/PoseCell.qml:37`
- **Rule**: BND-2
- **Finding**: `text = storedValue.toFixed(3)` in `onEditingFinished` permanently destroys the `text: storedValue.toFixed(3)` binding (known C1; the file's own comment relies on "full table refreshes recreate this delegate" as the recovery path).
- **Mitigation**: U3's D2 commit contract — re-sync via `Binding`/`Qt.binding`, never one-shot assignment.

### [L-02] Layout-managed items sized with bare `width` (8 sites)
- **File**: `src/app/experimental/main.qml:275,282,289,296,303,310,317` (pose-dialog column-header labels) + `PoseCell.qml:26-27` (`width: 78; height: 26` in a RowLayout)
- **Rule**: LAY-2 (+ qmllint "Detected width on an item that is managed by a layout" ×8)
- **Finding**: RowLayout children set `width` directly; sizing/alignment relies on Qt's implicitWidth fallback (undefined-behavior per qmllint).
- **Mitigation**: U2 — `Layout.preferredWidth` at the call sites; `implicitWidth`/`implicitHeight` on PoseCell.

### [L-03] Transparent Rectangles used as spacers (4 sites)
- **File**: `main.qml:570,655,692`, `SettingsPanel.qml:24`
- **Rule**: PRF-1
- **Finding**: Transparent-fill Rectangles still create geometry nodes.
- **Mitigation**: U2 — `Item` instead.

### [L-04] Delegate role access without `required property` (~20 sites)
- **File**: `main.qml:365-399` (pose-table delegate), `665,702` (list delegates)
- **Rule**: DEL-1
- **Mitigation**: U5 (table) / U2 (lists) — `required property` + `model.roleName`.

### [L-05] Redundant/clipping imports
- **File**: `main.qml:5` + `renderer.qml:2` (IMP-1 `QtQuick.Window` redundant), `main.qml:8` (IMP-2 versioned import)
- **Rule**: IMP-1/IMP-2
- **Mitigation**: U2 import cleanup. (Note: `main.qml:3` IMP-3 plain `QtQuick.Controls` is a false positive — the file pins `QtQuick.Controls.Material` deliberately at line 4.)

### [L-06] Remaining style/JS/ordering findings
- `SettingsPanel.qml`: BND-1 `property var commit` ×4 (use typed/function properties), JS-1 `var` (line 89), PRF-3 clip (59), ~20 ORD-1 ordering nits.
- `main.qml`: BND-1 (123), JS-1 (161), JS-2 loose equality (695), PRF-3 clip ×3 (347, 652, 689), STY-3 anchors dot-notation (940), ~50 ORD-1 ordering nits.
- `Theme.qml:7`: STY-1 no `id: root`.
- **Mitigation**: U2 pass (qmlformat for ordering is NOT applied — repo has no QML format convention; manual ORD fixes only where touched).

**False positives recorded**: `main.qml:152` BND-2 (`messageDialog.title = ...` — literal initializers, no binding to kill); IMP-3 as above; PRF-3 clip on ScrollViews (acceptable — content overflow masking, noted as redundant per D-606).

---

## Deep analysis findings

### [D-01] Dirty-badge pill Rectangles collapse to 0 width — the badge background never renders
- **File**: `main.qml:221-227` (pose dialog) + `SettingsPanel.qml:24` (identical)
- **Category**: Layout & Anchoring — confidence 90
- **Finding**: A plain `Rectangle` has `implicitWidth: 0`; in a `RowLayout` with no `Layout.preferredWidth` it sizes to 0. The centered label ("● unsaved"/"saved") renders **without its colored pill** in both dialogs.
- **Trace**: `Layout.preferredHeight: 14` only fixes height; `anchors.centerIn` centers on a 0-width parent.
- **Mitigation**: U2 — `Layout.preferredWidth` derived from the label (e.g. `Math.max(label.implicitWidth + 12, 28)`).

### [D-02] Pose-row `Layout.fillWidth` is a no-op inside a plain Column
- **File**: `main.qml:358-363`
- **Category**: Layout & Anchoring — confidence 85
- **Finding**: `Layout.*` attached props are honored only under a Qt Quick Layout parent; the parent is a plain `Column`, so rows render at implicit width (~532px vs ~636px available) with a dead band on the right.
- **Mitigation**: U5 — size row delegates from `ListView.view.width`/cellWidth.

### [D-03] SettingsPanel scrollbar gutter `width: root.width - 18` clips/wastes
- **File**: `SettingsPanel.qml:60`
- **Category**: Layout & Anchoring — confidence 90
- **Finding**: With the vertical scrollbar visible, the Column is ~4-10px wider than the ScrollView's availableWidth → SpinBox right edges clipped; hidden scrollbar leaves an 18px dead gutter.
- **Mitigation**: U2 — bind to the ScrollView's `availableWidth` (the pose dialog already uses the correct pattern).

### [D-04] Pose table built eagerly at dataset load while the dialog is closed
- **File**: `main.qml:179,201,359-360`
- **Category**: Component Loading & Lifecycle — confidence 90
- **Finding**: `Dialog` contentItem + `Repeater` construct at app startup; a 500-frame study builds 3,500 TextField delegates at load time, not on first open. No `Loader` anywhere.
- **Mitigation**: U5 — wrap dialog content in a `Loader` bound to the dialog's open state, plus virtualization.

### [D-05] Dialog content persists across open/close — stale-table premise CONFIRMED
- **File**: `main.qml:179,201,344,359`
- **Category**: Component Loading & Lifecycle — confidence 85
- **Finding**: QQC2 `Dialog` never destroys `contentItem` on close; the pose table only re-reads on model notify. The I1/I2 stale-on-reopen premise is structurally certain (was a "verify empirically" residual risk).
- **Mitigation**: U3's PoseBridge refresh-owner fix is the correct repair; the U5 Loader makes open-time recreation a natural re-sync.

### [D-06] Whole-table delegate teardown is the app's re-sync mechanism
- **File**: `PoseCell.qml:33,37-49` + `main.qml:359-401`
- **Category**: Component Loading & Lifecycle — confidence 85
- **Finding**: Every `refresh()` destroys/recreates all cells (churns 6×N objects, destroys editing state). This is why the C1 binding kill is dangerous; virtualization must land WITH the U3 commit contract.
- **Mitigation**: U3 + U5 together remove the recreation dependency.

### [D-07] Run-lock matrix: 3 more controls escape the lock (NEW vs plan)
- **File**: `main.qml:599-608` (Camera/Model toggles), `822-826` (Black-sil.), `837-847` (Fem/Tib)
- **Category**: States & Structure — confidence 100
- **Finding**: The "everything else is locked" comment (959-963) is false: the interaction-mode toggles, Black-sil. checkbox, and Fem/Tib buttons have no `enabled` lock while 14 other sites do. The Camera/Model toggles were not in the plan's I4 inventory.
- **Mitigation**: U3 — extend D5's lock matrix to the mode toggles; introduce a `runLocked` readonly root property to prevent future drift.

### [D-08] ButtonGroup `checked:` bindings are killed by clicks (latent)
- **File**: `main.qml:592-608, 830-847, 861-880`
- **Category**: States & Structure — confidence 80
- **Finding**: `checked: bridge.x === N` + `onClicked: bridge.x = N`; QQC2 ButtonGroup writes `checked` imperatively on click, killing the binding on the clicked button (masked by group exclusivity). A programmatic bridge write would then rely on the sibling's surviving binding.
- **Mitigation**: U3/U4 — single-source checked state or restore binding; low priority (works today).

### [D-09] Performance cluster
- **File**: `main.qml:344-404, 751-789, 934-952`, `PoseBridge.cpp:155,240,281` + `notifyCellChanged`
- **Category**: Performance & Code Quality — confidence 80-100
- **Finding**: (a) Repeater 6 TextFields × frame (×100); (b) full `beginResetModel` on single-row copy/load changes; (c) `dataChanged` without roles re-evaluates all 6 cells per edit; (d) `baseName()` function calls in text bindings; (e) no `Text.PlainText` anywhere; (f) redundant `clip: true` on ScrollView (347); (g) debug pose readout ships as a permanent viewport overlay.
- **Mitigation**: U5 (a, b); U7 (c, d, e); U2 (f); U4 (g — remove or gate the debug readout).

---

## Investigation targets (human verification)

- **[I-01]** (75) Header↔cell alignment relies on Qt's implicitWidth fallback for explicit `width:` in layouts — verify visually after U2's `Layout.preferredWidth` conversion.
- **[I-02]** (75) Full model reset vs pooled delegates after virtualization — `PoseTableModel::refresh()` does begin/endResetModel (copy/load paths); per-cell commits use notifyCellChanged. Re-check reset-vs-notify call sites in U5.
- **[I-03]** (75) Silent no-renderer failure mode — if `QmlVtkRenderer` fails to instantiate, all 6 glue blocks silently no-op; add a `Component.onCompleted` guard in U2's ViewportPanel.
- **[I-04]** (70) Delegate `id` references (`frameList.width`, `modelList.width`) block component extraction — convert to `ListView.view` + required properties when delegates move (U2/U5).
- **[I-05]** (70) ListView built-in Up/Down currentIndex would diverge from the bridge (setCurrentFrame only fires on click) — D7's onCurrentIndexChanged wiring is the fix (U3).
- **[I-06]** (68) Column-header row is scrollbar-width wider than the table Column when the scrollbar appears — verify at large frame counts (U5).
- **[I-07]** (65) Left-panel frame/model lists share `Layout.fillHeight` 1:1 — a 200-frame/2-model study leaves the model list half-empty (U4 composition).
- **[I-08]** (65) `Layout.fillWidth` no-op on pose rows (D-02 companion — harmless today, fix in U5).
- **[I-09]** (62) `Column { width: poseScroll.availableWidth }` re-evaluates on scrollbar toggle — potential relayout jitter (U5).
- **[I-10]** (60) `selectedModels.indexOf(index)` is O(n) per delegate evaluation — cosmetic at current counts.
- **[I-11]** (60) Visibility coupling: actions/header rows remain visible above the empty-state label when the table is hidden — likely intended, confirm in U4.
- **[I-12]** (60) Redundant re-click setter fires on already-active toggle buttons (U4, trivial).
- **[I-13]** (60) Dirty dialogs (`modal: false` + `CloseOnPressOutside`) discard edits on press-outside with no confirm — the U4 badge fold-in addresses visibility, not the close guard (see recommendation below).

---

## Summary

| Category | Lint | Deep | Investigate | Total |
|----------|------|------|-------------|-------|
| Imports / ordering / style | 33 | — | — | 33 |
| Bindings / delegates | 21 | 2 | 4 | 27 |
| Layout & anchoring | 8 | 3 | 3 | 14 |
| Loading & lifecycle | — | 3 | 2 | 5 |
| States & structure | — | 2 | 1 | 3 |
| Performance | — | 1 (cluster) | 1 | 2+ |
| **Total** | **41** | **9** | **13** | **63** |

**Verified clean:** no binding loops; no `Qt.binding` closure captures; no `Loader` misuse (none exists); no anchors-on-hidden items; no anchors/Layout mixing; no `opacity`/`layer`/effects; teardown ordering safe; all 6 Connections use modern handler form; no Qt5 migration residue.

**Top-5 most impactful:** D-07 (unlocked mode toggles — run-state integrity), D-01 (badge pill invisible — visual bug in both dialogs), D-04/D-05 (eager O(n) table + confirmed staleness), L-02/D-02 (layout-managed sizing), D-09a (Repeater 6×frames).

## Triage → plan mapping

| Finding(s) | Destination |
|------------|-------------|
| L-01, L-02, L-04, L-05, L-06 (partial), D-02, D-03, I-03, I-04 | U2 (extraction + Layout/import fixes) |
| D-07, D-08, I-05 | U3 (run-lock matrix extension + keyboard wiring + ButtonGroup mitigation) |
| D-01, D-09g, I-07, I-11, I-12, I-13 | U4 (composition: badge pill fix, debug readout removal, panel balance, dirty-close guard) |
| D-04, D-05, D-06, L-04 (table), I-02, I-06, I-08, I-09 | U5 (virtualization + Loader + refresh granularity) |
| D-09b, D-09c, D-09d, D-09e | U7 (role-filtered dataChanged, reset granularity, PlainText, cached baseName) |
| D-08 (checked-binding) | U3/U4 boundary — U3 decides the single-source pattern, U4 applies it to the three ButtonGroups |

**Recommendation (I-13):** fold a dirty-close guard into U4 alongside the shell dirty badge: when a dialog with dirty state is closed via press-outside/Esc, confirm ("Discard unsaved changes?"); the Run-button close stays unconditional (deliberate, plan-005 behavior).

---

> AI assistance has been used to create this output.
