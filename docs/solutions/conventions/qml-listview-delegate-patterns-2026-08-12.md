---
title: "Qt 6.7 ListView delegate patterns: pooling, required properties, and keyboard contracts"
date: 2026-08-12
category: conventions
module: jtml_view
problem_type: convention
component: frontend_stimulus
severity: high
applies_when:
  - "writing or refactoring Qt 6 ListView/TableView delegates"
  - "virtualized lists with inline editing or per-row state"
  - "declaring required properties on delegates"
  - "handling input inside nested delegate items (MouseArea, Keys)"
tags: [qml, qtquick, listview, delegates, pooling, required-property, keyboard, qmltest]
related_components:
  - testing_framework
---

# Qt 6.7 ListView delegate patterns: pooling, required properties, and keyboard contracts

## Context

The experimental QML front-end (`src/app/experimental/`, Qt 6.7.2) virtualized
its pose table and rebuilt its list delegates during plan 007. Four Qt 6
behaviors were confirmed the hard way — each silently broke the app, and two
were only caught by a new headless Qt Quick Test harness (`test/qml/`, see
`docs/solutions/conventions/jtml-qml-view-testing-2026-08-12.md` for the
harness recipe). All four are invisible to qmllint and three of the four
produce **no error dialog** — just a dead control or a silently lost value.
They compound: the fix for one (required properties for testability)
activates another (stale implicit `model` context on reuse).

## Guidance

### 1. `ListView.view` is null inside NESTED delegate items — use the list's id

The attached `view` property is provided to the delegate **root only**.
Inside a nested MouseArea (or any child), `ListView.view` evaluates to null,
and `null.currentIndex = index` throws a silent `TypeError` per invocation.

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

Root-level uses (`width: ListView.view.width`,
`color: ListView.view.currentIndex === index`) are fine — the root receives
`view`. This was the frame-picker regression (13 clicks fired, `currentIndex`
never moved); the trace-driven debugging method that found it is documented
in `docs/solutions/ui-bugs/jtml-qml-trace-driven-debugging-2026-08-12.md`.

### 2. Qt 6.7 pooling does NOT drop focus; the pool notifications are ATTACHED signals

Two assumptions fail together:

- A recycled delegate keeps `activeFocus` (hidden/reparented items retain
  it), so "focus loss fires `editingFinished` before pooling" **never
  holds** — relying on it silently clobbers the typed value on recycle.
- The pool emits **attached** `ListView.onPooled` / `ListView.onReused`
  signals. A delegate-root `signal pooled()` + `onPooled:` handler fails at
  load (`Cannot assign to non-existent property`).

The landed mechanism (PoseCell + PosesTable): an `edited` flag set by
`onTextEdited`, a single `doCommit()` path reading a **captured commit
tuple** (captured at focus-in, never at commit time — protects against
mid-edit selection changes and re-binds), and an explicit flush from the
attached handler:

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

// PosesTable.qml delegate — the attached handlers (must be DECLARED):
ListView.onPooled:  { cellX.commitIfEditing(); /* ... all six cells */ }
ListView.onReused:  { cellX.resetDisplay();    /* ... all six cells */ }
```

Follow-ups proven necessary by review pins:

- `Component.onDestruction: { if (edited && !suppressDestructionCommit) doCommit() }`
  — a model reset / dialog teardown destroys cells mid-edit; commit rather
  than lose the value (the discard-close path suppresses it).
- `onFrameIndexChanged` flushing `commitIfEditing()` on all cells — an
  **in-place re-bind** (instant `positionViewAtIndex` jump) reuses the same
  delegate instance without pooling, so neither `onPooled` nor `onReused`
  fires; the flush commits to the captured old row.

### 3. With required properties, the IMPLICIT `model` context goes stale on reuse — declare `required property var model`

When a delegate declares any required properties, the implicit `model`
context object **stops re-pointing on pool reuse**: a required `frameIndex`
re-binds correctly (3) while a `storedValue: model.x` binding still reads
the previous row (0). Test-proven diagnostic (`frameIndex=3,
cellB.storedValue=0` on the same delegate). Fix: declare the model itself as
required so it re-binds through the required-property mechanism:

```qml
delegate: RowLayout {
    required property int frameIndex
    required property var model      // re-binds per row; the implicit one goes stale
    // roles are still read via model.x/model.y/... (see the FINAL-collision note)
}
```

Related (same family): `required property double x/y/z` on a delegate root
collides with Item's **FINAL** geometry properties (`Cannot override FINAL
property` — the app becomes unloadable). Roles must be named anything but
geometry names; read geometry-named roles only via `model.roleName`.

### 4. `Keys.onPressed` on a QQC2 TextField root captures arrows before the internal editor

For the D7 cell-navigation contract (single tab stop; Left/Right between the
6 cells, Up/Down rows, Esc reverts, Enter commits), the keys are intercepted
at the control root via a `Keys.onPressed` switch that emits a
`navRequested(direction)` signal; the delegate implements `focusCell(axis)`
(sibling within the row) and `moveToRow(frame, axis)`
(`positionViewAtIndex` + `Qt.callLater` focus, because the target row may
not be instantiated under virtualization). Escape sets `edited = false` and
calls `resetDisplay()` — revert without committing. The trade-off is
accepted: arrow keys move between cells, so cursor positioning inside a
numeric field is click-only.

## Why This Matters

Each behavior silently corrupted user-visible state (lost pose edits, dead
frame selection, wrong-model pose writes) with no error dialog — the worst
kind of QML failure. qmllint flags none of them. The harness pins
(`test/qml/tst_PoseCell.qml`, `tst_PosesTable.qml`) are what made the
mechanisms verifiable: two of the four behaviors (#2, #3) were only caught
because the pins exercised the real delegate mechanics headlessly. Without
the pins, the "fix" for #2 would have been a symptom fix.

## When to Apply

- Any delegate that accesses the view, the model, or does work on pool
  recycle (virtualized lists with editing, selection, or per-row state).
- Any time a delegate gains `required property` declarations — check the
  implicit `model` and geometry-name collisions immediately.
- Keyboard contracts on editable table cells — capture at the control root,
  not in the delegate's parent.

## Examples

The live implementation is `src/app/experimental/PosesTable.qml` (delegate
with required `frameIndex` + required `model`, attached `onPooled`/
`onReused`, `onFrameIndexChanged` flush, `focusCell`/`moveToRow`) and
`src/app/experimental/PoseCell.qml` (edited flag, `doCommit`,
`commitIfEditing`, `resetDisplay` with focus-retained re-capture,
`Component.onDestruction`, `Keys.onPressed`). The pins that guard every
mechanism are in `test/qml/tst_PosesTable.qml` and `tst_PoseCell.qml` —
including `test_rejectedCommitSurvivesRecycle` (caught #3) and
`test_frameClickMovesIndex` (guards #1).

## Related

- `docs/solutions/conventions/jtml-qml-view-testing-2026-08-12.md` — the
  harness recipe, qmllint gate, profiling build, and the same gotchas
  summarized (updated with #3/#4 on 2026-08-12).
- `docs/solutions/ui-bugs/jtml-qml-trace-driven-debugging-2026-08-12.md` —
  how the qmlprofiler trace census found #1.
- Plan: `docs/plans/2026-08-12-007-refactor-qml-experimental-improvement-pass-plan.md`
