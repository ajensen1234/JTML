## Knowledge-track capture: Qt 6.7 ListView-delegate behaviors (four confirmed gotchas + the corrected commit-on-pool)

---

### Suggested frontmatter

```yaml
---
title: "Qt 6.7 ListView delegates: four behaviors that break naive assumptions (view, pooling, required properties, keys)"
date: 2026-08-12
category: conventions
module: jtml_view
problem_type: convention
component: frontend_stimulus
severity: high
applies_when:
  - writing or refactoring Qt 6 ListView/TableView delegates
  - virtualized lists with inline editing or per-row state
  - any delegate that accesses the view, the model, or does work on pool recycle
tags: [qml, qtquick, listview, delegates, pooling, qmltest]
related_components:
  - testing_framework
  - tooling
---
```

---

## Context

The experimental QML front-end (`src/app/experimental/`, Qt 6.7.2) virtualized its pose table and rebuilt its list delegates during plan 007. Four Qt 6 behaviors were confirmed the hard way — each one silently broke the app, and two were only caught by a new headless Qt Quick Test harness (`test/qml/`, see `docs/solutions/conventions/jtml-qml-view-testing-2026-08-12.md` for the harness recipe). All four are invisible to qmllint and three of the four produce **no error dialog** — just a dead control or a silently lost value. They compound: the fix for one (required properties for testability) activates another (stale implicit `model` context on reuse).

## Guidance

### 1. `ListView.view` is null inside NESTED delegate items — use the list's id

The attached `view` property is provided to the delegate **root only**. Inside a nested MouseArea (or any child), `ListView.view` evaluates to null, and `null.currentIndex = index` throws a silent `TypeError` per invocation.

```qml
// BROKEN (delegate root is a Rectangle; the MouseArea is nested):
delegate: Rectangle {
    MouseArea {
        onClicked: ListView.view.currentIndex = index   // null -> TypeError
    }
}

// FIXED — the list id is in file scope and visible from delegates:
delegate: Rectangle {
    MouseArea {
        onClicked: frameList.currentIndex = index
    }
}
```

Root-level uses (`width: ListView.view.width`, `color: ListView.view.currentIndex === index`) are fine — the root receives `view`.

### 2. Qt 6.7 pooling does NOT drop focus; the pool notifications are ATTACHED signals

Two assumptions fail together:

- A recycled delegate keeps `activeFocus` (hidden/reparented items retain it), so "focus loss fires `editingFinished` before pooling" **never holds** — relying on it silently clobbers the typed value on recycle.
- The pool emits **attached** `ListView.onPooled` / `ListView.onReused` signals. A delegate-root `signal pooled()` + `onPooled:` handler fails at load (`Cannot assign to non-existent property`) and, if declared, never fires.

The landed mechanism (PoseCell + PosesTable): an `edited` flag set by `onTextEdited`, a single `doCommit()` path reading a **captured commit tuple** (captured at focus-in, never at commit time — protects against mid-edit selection changes and re-binds), and an explicit flush from the attached handler:

```qml
// PoseCell.qml — the flag + single commit path:
property bool edited: false
onTextEdited: edited = true
function doCommit() {
    if (!edited) return
    edited = false
    if (!root.poseBridge.setPoseValue(commitFrame, commitModel,
                                      commitAxis, text)) {
        text = Qt.binding(() => storedValue.toFixed(3))  // re-arm, never one-shot
    }
}
function commitIfEditing() { if (edited) doCommit() }
function resetDisplay() {
    if (activeFocus) {
        // Pool round-trip RETAINED focus: frameRow already re-bound to the
        // NEW row — re-capture so the next edit commits the new row.
        commitFrame = frameRow; commitModel = root.studyBridge.primaryModelIndex
        commitAxis = axisIndex
    } else { commitFrame = -1; commitModel = -1; commitAxis = -1 }
    edited = false
    text = Qt.binding(() => storedValue.toFixed(3))
}

// PosesTable.qml delegate — the attached handlers (must be declared):
ListView.onPooled:  { cellX.commitIfEditing(); /* ... all six cells */ }
ListView.onReused:  { cellX.resetDisplay();    /* ... all six cells */ }
```

Follow-ups proven necessary by review pins: `Component.onDestruction: { if (edited && !suppressDestructionCommit) doCommit() }` (a model reset/dialog teardown destroys cells mid-edit — commit rather than lose the value), and `onFrameIndexChanged` flushing `commitIfEditing()` on all cells (an **in-place re-bind** — instant `positionViewAtIndex` jump — reuses the same delegate instance without pooling, so neither `onPooled` nor `onReused` fires; the flush commits to the captured old row).

### 3. With required properties, the IMPLICIT `model` context goes stale on reuse — declare `required property var model`

When a delegate declares any required properties, the implicit `model` context object **stops re-pointing on pool reuse**: a required `frameIndex` re-binds correctly (3) while a `storedValue: model.x` binding still reads the previous row (0). Test-proven diagnostic: `frameIndex=3 cellB.storedValue=0` on the same delegate. Fix: declare the model itself as required so it re-binds through the required-property mechanism:

```qml
delegate: RowLayout {
    required property int frameIndex
    required property var model      // re-binds per row; the implicit one goes stale
    // roles are still read via model.x/model.y/... (see the FINAL-collision note)
}
```

Related (same family): `required property double x/y/z` on a delegate root collides with Item's **FINAL** geometry properties (`Cannot override FINAL property` — the app becomes unloadable). Roles must be named anything but geometry names; read geometry-named roles only via `model.roleName`.

### 4. `Keys.onPressed` on a QQC2 TextField root captures arrows before the internal editor

For the D7 cell-navigation contract, arrows/Escape are intercepted at the control root; the internal text editor never sees them (acceptable for numeric cells — cursor positioning stays available by click):

```qml
Keys.onPressed: (event) => {
    switch (event.key) {
    case Qt.Key_Left:  root.navRequested(-1); event.accepted = true; break
    case Qt.Key_Right: root.navRequested(1);  event.accepted = true; break
    case Qt.Key_Up:    root.navRequested(-2); event.accepted = true; break
    case Qt.Key_Down:  root.navRequested(2);  event.accepted = true; break
    case Qt.Key_Escape:
        edited = false; resetDisplay(); event.accepted = true; break   // revert
    }
}
```

The delegate translates the direction: Left/Right → `focusCell(axis ± 1)` (sibling in the row); Up/Down → `tableList.positionViewAtIndex(frame ± 1, ListView.Contain)` then a `Qt.callLater` that focuses `itemAtIndex(...)`'s cell once the virtualized row materializes.

## Why This Matters

All four behaviors fail **silently**: a dead frame picker (no error surfaced — found via a qmlprofiler event census: 13 `onClicked` firings, 1 `onCurrentIndexChanged`), a silently clobbered typed value on scroll (pooling retains focus), stale displayed values on recycle (required-property model), and — at the other extreme — an unloadable app (FINAL collision). qmllint flags none of them, and "it looks fine on screen" does not survive a scroll. The harness pattern is the only reliable guard: each behavior has a headless pin (`mouseClick`/`keyClick` + `positionViewAtIndex` + `tryCompare` on the injected fake bridge's log). The pass shipped 40 QML test functions and every regression above is pinned.

## When to Apply

- Any Qt 6 ListView/TableView delegate that accesses `ListView.view`, reads `model.*`, holds per-row editable state, or relies on focus/`editingFinished` semantics.
- Adding `required property` to a delegate (for qmlsc/testability) — immediately also declare `required property var model` (and never geometry-named roles).
- Virtualized editable lists (recycling + typing) — always the `edited`-flag + attached-handler commit-on-pool pattern.
- Keyboard contracts on QQC2 TextField-derived cells.

## Examples

The pins that guard each behavior (`test/qml/tst_PosesTable.qml`, `test/qml/tst_PoseCell.qml`):

- `test_frameClickMovesIndex` / `test_modelClickTogglesRow` (StudyFlows) — mouse-click a delegate row; the bridge must receive the clicked index (guards #1; the old pin skipped the mouse path as "ListView-standard" — that gap is what let #1 through).
- `test_commitOnPoolMidEdit` / `test_rejectedCommitSurvivesRecycle` — type, scroll away/back, assert the commit landed on the captured row and the display re-armed (guards #2).
- `test_recycleRetainedFocusRecapturesTuple` — recycle with retained focus, type again, assert the NEW row is committed (guards #2's focus-retention follow-up).
- `test_instantJumpFlushesEdit` — instant `positionViewAtIndex(400)` mid-edit; the commit must land exactly once (guards #2's in-place path).
- `test_keyboardLeftRightMovesCell` / `test_keyboardUpDownMovesRow` / `test_escapeRevertsCell` — keyClick the arrow/Escape contract (guards #4).
- `test_storedValueChangeRerenders` — after a rejected commit, the re-armed binding tracks `storedValue` changes (guards the `Qt.binding` re-arm, never a one-shot assignment).

## Related

- `docs/solutions/conventions/jtml-qml-view-testing-2026-08-12.md` — the harness recipe (quick_test_main, qrc aliasing the real sources, injected fakes, headless ctest) and the first three gotchas in compact form; **this entry supersedes/extends its "QML gotchas" section — notably adding #3 (required-property stale `model`), the in-place re-bind flush, and #4 (keys)**.
- `docs/solutions/ui-bugs/jtml-qml-trace-driven-debugging-2026-08-12.md` — the qmlprofiler event-census method that found #1.
- Plan: `docs/plans/2026-08-12-007-refactor-qml-experimental-improvement-pass-plan.md` (U5/U6 + review round R1/R2).