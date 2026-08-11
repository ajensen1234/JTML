import QtQuick
import QtQuick.Controls

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
TextField {
    id: root

    required property int frameRow
    required property int axisIndex
    required property double storedValue

    text: storedValue.toFixed(3)
    width: 78
    height: 26
    font.pixelSize: 11
    horizontalAlignment: Text.AlignRight
    selectByMouse: true

    onEditingFinished: {
        // The user's typing has broken the text binding; a rejected commit
        // restores the stored value with a plain assignment (the stored
        // value is current — full table refreshes recreate this delegate).
        if (!poseBridge.setPoseValue(frameRow,
                                     studyBridge.primaryModelIndex,
                                     axisIndex, text)) {
            text = storedValue.toFixed(3)
        }
    }
}
