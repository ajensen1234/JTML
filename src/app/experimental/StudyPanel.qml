import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import "."  // Theme

// 007 U2: left-column study lists (extracted from main.qml): the frame
// list (current-frame contract) over the model list (bridge-owned
// multi-select) + the selection summary. Delegate selection contract
// unchanged — no QItemSelectionModel; currentIndex drives the bridge.
// The onDatasetChanged relay keeps the Qt.callLater deferral (it outlasts
// the clearDataset list-model swap). 007 U2 review fixes: delegate root-id
// references converted to ListView.view + required properties (I-04).

ColumnLayout {
    id: root
    Layout.fillWidth: true
    Layout.fillHeight: true
    spacing: 6

    // Frame list (delegate selection contract: currentIndex drives the
    // bridge — no QItemSelectionModel).
    Label {
        text: qsTr("Frames (%1)").arg(appBridge.frameCount)
        color: Theme.fg
        font.bold: true
        font.pixelSize: Theme.label
    }
    ListView {
        id: frameList
        Layout.fillWidth: true
        Layout.fillHeight: true
        clip: true
        enabled: !optimizerBridge.running
        model: studyBridge.frameListModel
        delegate: Rectangle {
            required property string display
            width: ListView.view.width
            height: 22
            color: ListView.view.currentIndex === index
                   ? Theme.selection : "transparent"
            Text {
                anchors.fill: parent
                anchors.leftMargin: 4
                verticalAlignment: Text.AlignVCenter
                color: Theme.fg
                text: display
                elide: Text.ElideRight
            }
            MouseArea {
                anchors.fill: parent
                onClicked: {
                    ListView.view.currentIndex = index
                    studyBridge.setCurrentFrame(index)
                }
            }
        }
        highlightFollowsCurrentItem: true
    }

    // Model list (multi-select via the bridge-owned set).
    Label {
        text: qsTr("Models (%1)").arg(appBridge.modelCount)
        color: Theme.fg
        font.bold: true
        font.pixelSize: Theme.label
    }
    ListView {
        id: modelList
        Layout.fillWidth: true
        Layout.fillHeight: true
        clip: true
        enabled: !optimizerBridge.running
        model: studyBridge.modelListModel
        delegate: Rectangle {
            required property string display
            width: ListView.view.width
            height: 22
            color: studyBridge.selectedModels.indexOf(index) !== -1
                   ? Theme.selection : "transparent"
            Text {
                anchors.fill: parent
                anchors.leftMargin: 4
                verticalAlignment: Text.AlignVCenter
                color: Theme.fg
                text: display
                elide: Text.ElideRight
            }
            MouseArea {
                anchors.fill: parent
                onClicked: studyBridge.toggleModelSelected(index)
            }
        }
    }
    Label {
        text: studyBridge.selectedModelCount > 0
              ? qsTr("Selected %1 · primary %2")
                    .arg(studyBridge.selectedModelCount)
                    .arg(studyBridge.primaryModelIndex)
              : qsTr("No model selected")
        color: Theme.fgMuted
        font.pixelSize: Theme.caption
    }

    // Dataset-changed relay (moved with the lists): the deferral outlasts
    // the list-model swap (clearDataset deletes the old model instance).
    Connections {
        target: studyBridge
        function onDatasetChanged() {
            Qt.callLater(function() {
                frameList.currentIndex = studyBridge.currentFrame
            })
        }
    }
}
