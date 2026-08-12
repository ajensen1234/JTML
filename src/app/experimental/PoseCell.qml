import QtQuick
import QtQuick.Controls
import "."  // Theme

// 005 U8: one editable pose cell of the Poses dialog table. Pure view glue
// (no logic — the commit + validation live in PoseBridge):
//  - the stored value comes from the table model's role (re-reads on
//    dataChanged / table reset — the model is the authoritative display
//    source);
//  - on editingFinished the text commits through PoseBridge::setPoseValue
//    (immediate per-cell SavePose). A rejected commit (non-numeric / NaN /
//    out-of-range) leaves the stored state unchanged: the field reverts to
//    the stored value and the bridge's inline validation message shows.
//
// 007 U2: explicit width/height -> implicitWidth/implicitHeight (LAY-2:
// the cell is consumed inside a RowLayout; explicit width on a
// layout-managed item is undefined behavior per qmllint).
//
// 007 U3 (D2, C1/C2): the commit tuple is CAPTURED when editing starts
// (focus-in), never read at commit time:
//  - a mid-edit selection change cannot commit to the new primary (C2);
//  - a delegate recycle cannot re-bind frameRow under a pending commit
//    (C1) — the captured row travels with the edit.
// A failed commit re-syncs the display with Qt.binding() so the storedValue
// binding is ALIVE again (a one-shot assignment would leave the cell stale
// after a recycle — the old "full table refreshes recreate this delegate"
// recovery is gone once U5 virtualizes the table).
TextField {
    id: root

    // 007 U6 (D1): injected bridge surface — the PosesTable delegate
    // passes its own injected props down to each cell.
    required property var poseBridge
    required property var studyBridge

    required property int frameRow
    required property int axisIndex
    required property double storedValue

    text: storedValue.toFixed(3)
    implicitWidth: 78
    implicitHeight: 26
    font.pixelSize: Theme.caption
    horizontalAlignment: Text.AlignRight
    selectByMouse: true

    // D2 capture tuple: (frame, model, axis) at edit start.
    property int commitFrame: -1
    property int commitModel: -1
    property int commitAxis: -1

    // R2 (review round): when a model reset / dialog teardown destroys
    // this cell mid-edit, commit the live edit instead of silently
    // dropping it — UNLESS the owner explicitly suppressed commits (the
    // discard-close path).
    property bool suppressDestructionCommit: false

    // D7 (review round): the pose-table keyboard contract — the table is
    // a single tab stop; arrows move between cells/rows, Escape reverts.
    // Direction encoding for navRequested: -2 up, -1 left, +1 right,
    // +2 down.
    signal navRequested(int direction)

    Keys.onPressed: (event) => {
        switch (event.key) {
        case Qt.Key_Left:
            root.navRequested(-1)
            event.accepted = true
            break
        case Qt.Key_Right:
            root.navRequested(1)
            event.accepted = true
            break
        case Qt.Key_Up:
            root.navRequested(-2)
            event.accepted = true
            break
        case Qt.Key_Down:
            root.navRequested(2)
            event.accepted = true
            break
        case Qt.Key_Escape:
            // Esc reverts the cell to the stored value (and re-arms the
            // binding) without committing.
            edited = false
            resetDisplay()
            event.accepted = true
            break
        }
    }

    Component.onDestruction: {
        // R2: a full model reset (refreshTable) or the dialog teardown
        // destroys this cell mid-edit — commit rather than lose the
        // typed value (the discard-close path sets
        // suppressDestructionCommit first).
        if (edited && !suppressDestructionCommit) doCommit()
    }

    // 007 U6 (proven by the recycle pins): Qt 6.7 ListView pooling does
    // NOT drop focus (hidden/reparented items keep activeFocus), so the
    // "focus loss fires editingFinished before pooling" assumption never
    // holds on recycle — the typed value would be silently clobbered by
    // resetDisplay without a commit. Track interactive edits and commit
    // them explicitly from the delegate's onPooled (commit-on-pool, made
    // real).
    property bool edited: false

    onActiveFocusChanged: {
        if (activeFocus) {
            commitFrame = frameRow
            commitModel = root.studyBridge.primaryModelIndex
            commitAxis = axisIndex
        }
    }

    // User typing marks the cell edited (the text binding is broken by
    // editing; this flag survives focus transitions).
    onTextEdited: edited = true

    onEditingFinished: doCommit()

    // The single commit path: captured tuple + live text. A rejected
    // commit re-arms the storedValue binding (C1).
    function doCommit() {
        if (!edited) return
        edited = false
        if (!root.poseBridge.setPoseValue(commitFrame, commitModel,
                                     commitAxis, text)) {
            // Rejected commit: re-arm the storedValue binding (C1). The
            // stored value is current — the display shows it again and
            // further edits keep working.
            text = Qt.binding(() => storedValue.toFixed(3))
        }
    }

    // Called by the delegate's onPooled: flush a live edit before the
    // pooled cell is re-bound to a new row (the real commit-on-pool).
    function commitIfEditing() {
        if (edited) doCommit()
    }

    // D8 (U5, hardened in U6): pooled-delegate re-sync. After an edit the
    // text binding is dead; a reused cell re-arms it from the (re-bound)
    // storedValue and clears the stale commit tuple. Called by the
    // delegate's onReused — any live edit was already flushed by
    // commitIfEditing in onPooled (the focus-loss ordering does not hold
    // on recycle, so the flush is explicit).
    function resetDisplay() {
        // Review fix (ce-code-review 2026-08-12, C1): a pool round-trip
        // can RETAIN focus (Qt 6.7 pooling does not drop it). If the
        // recycled cell still has focus, frameRow/axisIndex already
        // re-bound to the NEW row — re-capture the tuple so the next edit
        // commits the new row; a stale (-1,-1,-1) tuple would silently
        // reject (and lose) the typed value.
        if (activeFocus) {
            commitFrame = frameRow
            commitModel = root.studyBridge.primaryModelIndex
            commitAxis = axisIndex
        } else {
            commitFrame = -1
            commitModel = -1
            commitAxis = -1
        }
        edited = false
        text = Qt.binding(() => storedValue.toFixed(3))
    }
}
