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

    onActiveFocusChanged: {
        if (activeFocus) {
            commitFrame = frameRow
            commitModel = studyBridge.primaryModelIndex
            commitAxis = axisIndex
        }
    }

    onEditingFinished: {
        if (!poseBridge.setPoseValue(commitFrame, commitModel,
                                     commitAxis, text)) {
            // Rejected commit: re-arm the storedValue binding (C1). The
            // stored value is current — the display shows it again and
            // further edits keep working.
            text = Qt.binding(() => storedValue.toFixed(3))
        }
    }

    // D8 (U5): pooled-delegate re-sync. After an edit the text binding is
    // dead; a reused cell re-arms it from the (re-bound) storedValue and
    // clears the stale commit tuple. No live edit survives pooling — the
    // focus loss that preceded pooling already committed it (commit-on-pool
    // ordering), so resetting here can never clobber a pending commit.
    function resetDisplay() {
        commitFrame = -1
        commitModel = -1
        commitAxis = -1
        text = Qt.binding(() => storedValue.toFixed(3))
    }
}
