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
// Inline components are not supported in engine-root documents (main.qml is
// loaded by URL), so this is a regular QML file registered via qmldir.
// 007 U2: explicit width/height -> implicitWidth/implicitHeight (LAY-2:
// the cell is consumed inside a RowLayout; explicit width on a
// layout-managed item is undefined behavior per qmllint).
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

    onEditingFinished: {
        // The user's typing has broken the text binding; a rejected commit
        // restores the stored value with a plain assignment (the stored
        // value is current — full table refreshes recreate this delegate).
        // 007 U3 replaces this with the capture-at-edit-start contract.
        if (!poseBridge.setPoseValue(frameRow,
                                     studyBridge.primaryModelIndex,
                                     axisIndex, text)) {
            text = storedValue.toFixed(3)
        }
    }
}
