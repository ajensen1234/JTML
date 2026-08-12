import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import "."  // Theme + PoseCell

// 007 U2: the editable pose table (extracted from main.qml, U8 surface):
// the column header + empty states + the Repeater-backed table body.
// NOTE: virtualization (ListView reuse) is plan 007 U5 — this unit
// relocates the existing Repeater-in-ScrollView as-is (with the Layout.*
// sizing fixes: header/cell widths via Layout.preferredWidth).

ColumnLayout {
    id: root
    Layout.fillWidth: true
    Layout.fillHeight: true
    spacing: 4

    // ---- Column header (fixed widths match the cell fields) -------------
    RowLayout {
        Layout.fillWidth: true
        spacing: 4
        Label {
            text: qsTr("Frame")
            Layout.preferredWidth: 44
            horizontalAlignment: Text.AlignRight
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("X")
            Layout.preferredWidth: 78
            horizontalAlignment: Text.AlignHCenter
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("Y")
            Layout.preferredWidth: 78
            horizontalAlignment: Text.AlignHCenter
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("Z")
            Layout.preferredWidth: 78
            horizontalAlignment: Text.AlignHCenter
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("XA")
            Layout.preferredWidth: 78
            horizontalAlignment: Text.AlignHCenter
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("YA")
            Layout.preferredWidth: 78
            horizontalAlignment: Text.AlignHCenter
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("ZA")
            Layout.preferredWidth: 78
            horizontalAlignment: Text.AlignHCenter
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
    }

    // ---- Empty states -----------------------------------------------------
    Label {
        Layout.fillWidth: true
        visible: poseBridge.rowCount === 0
        color: Theme.fgDim
        wrapMode: Text.Wrap
        text: qsTr("No frames loaded — load a study first.")
    }
    Label {
        Layout.fillWidth: true
        visible: poseBridge.rowCount > 0
                 && studyBridge.primaryModelIndex < 0
        color: Theme.fgDim
        wrapMode: Text.Wrap
        text: qsTr("Select a model in the model list to edit its "
                   + "poses (v1: pose ops edit the primary model).")
    }

    // ---- The editable table (scrolls when the column is short) -----------
    ScrollView {
        id: poseScroll
        Layout.fillWidth: true
        Layout.fillHeight: true
        clip: true
        // Hidden only without a dataset + primary model; during an
        // optimizer run it stays visible but disabled (the U6 locking —
        // the Run button also closes this dialog).
        visible: poseBridge.rowCount > 0
                 && studyBridge.primaryModelIndex >= 0
        enabled: !optimizerBridge.running

        Column {
            width: poseScroll.availableWidth
            spacing: 2

            Repeater {
                model: poseBridge.tableModel
                delegate: RowLayout {
                    Layout.fillWidth: true
                    spacing: 4
                    Label {
                        text: qsTr("F%1").arg(model.frameIndex)
                        Layout.preferredWidth: 44
                        horizontalAlignment: Text.AlignRight
                        color: Theme.fgMuted
                        font.pixelSize: Theme.caption
                    }
                    PoseCell {
                        frameRow: model.frameIndex
                        axisIndex: 0
                        storedValue: model.x
                    }
                    PoseCell {
                        frameRow: model.frameIndex
                        axisIndex: 1
                        storedValue: model.y
                    }
                    PoseCell {
                        frameRow: model.frameIndex
                        axisIndex: 2
                        storedValue: model.z
                    }
                    PoseCell {
                        frameRow: model.frameIndex
                        axisIndex: 3
                        storedValue: model.xa
                    }
                    PoseCell {
                        frameRow: model.frameIndex
                        axisIndex: 4
                        storedValue: model.ya
                    }
                    PoseCell {
                        frameRow: model.frameIndex
                        axisIndex: 5
                        storedValue: model.za
                    }
                }
            }
        }
    }
}
