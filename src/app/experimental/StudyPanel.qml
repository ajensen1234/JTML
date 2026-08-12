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
//
// 007 U3 (D7, I6): keyboard wiring — every currentIndex change on the
// frame list (click, keyboard Up/Down, programmatic) syncs to the bridge
// via onCurrentIndexChanged, so the highlight and the bridge state cannot
// diverge. A dataset-swap guard suppresses the sync while the clearDataset
// model replacement transiently resets currentIndex (the deferral re-syncs
// it to the bridge's frame afterwards) — a transient -1/0 write must never
// reach the bridge (it would cascade into selectionChanged -> seed clear +
// pose-table re-point). Model rows toggle on Space/Enter; rows are
// focusable (activeFocusOnTab) with a visible focus indicator (the full
// keyboard audit is U4).

ColumnLayout {
    id: root
    Layout.fillWidth: true
    Layout.fillHeight: true
    spacing: Theme.spacingSm

    // D5 (plan 007 U3): single run-lock source for this panel.
    readonly property bool runLocked: optimizerBridge.running
    // D7 dataset-swap guard: set when a dataset replace is in flight so a
    // transient currentIndex reset never syncs to the bridge.
    property bool suppressFrameSync: false

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
        enabled: !root.runLocked
        model: studyBridge.frameListModel
        // D7 (I6): the bridge is the single source of truth — every
        // currentIndex change syncs to it (the delegate MouseArea only
        // sets currentIndex; Up/Down arrives via the ListView's key
        // handling once a row has focus).
        onCurrentIndexChanged: {
            if (root.suppressFrameSync) return
            studyBridge.setCurrentFrame(frameList.currentIndex)
        }
        delegate: Rectangle {
            required property string display
            width: ListView.view.width
            height: 24
            color: ListView.view.currentIndex === index
                   ? Theme.selection : "transparent"
            // D7: rows are reachable by keyboard (Tab in, arrows move).
            activeFocusOnTab: true
            // D7: visible focus indicator for keyboard users.
            Rectangle {
                visible: parent.activeFocus
                anchors.fill: parent
                radius: 2
                color: "transparent"
                border.color: Theme.accent
                border.width: 1
            }
            // Plan 007 U4: screen-reader surface (the row is a custom
            // item built from primitives).
            Accessible.role: Accessible.ListItem
            Accessible.name: display
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
                onClicked: ListView.view.currentIndex = index
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
        enabled: !root.runLocked
        model: studyBridge.modelListModel
        delegate: Rectangle {
            required property string display
            width: ListView.view.width
            height: 24
            color: studyBridge.selectedModels.indexOf(index) !== -1
                   ? Theme.selection : "transparent"
            // D7: rows are reachable by keyboard; Space/Enter toggles the
            // row (the bridge-owned multi-select set).
            activeFocusOnTab: true
            // D7: visible focus indicator for keyboard users.
            Rectangle {
                visible: parent.activeFocus
                anchors.fill: parent
                radius: 2
                color: "transparent"
                border.color: Theme.accent
                border.width: 1
            }
            // Plan 007 U4: screen-reader surface (the row is a custom
            // item built from primitives).
            Accessible.role: Accessible.ListItem
            Accessible.name: display
            Text {
                anchors.fill: parent
                anchors.leftMargin: 4
                verticalAlignment: Text.AlignVCenter
                color: Theme.fg
                text: display
                elide: Text.ElideRight
            }
            Keys.onPressed: (event) => {
                if (event.key === Qt.Key_Space
                        || event.key === Qt.Key_Return
                        || event.key === Qt.Key_Enter) {
                    studyBridge.toggleModelSelected(index)
                    event.accepted = true
                }
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
    // D7: the suppression flag is raised here so the transient
    // currentIndex resets during the swap never sync to the bridge; the
    // deferral re-syncs currentIndex to the bridge's frame, then clears
    // the flag.
    Connections {
        target: studyBridge
        function onDatasetChanged() {
            root.suppressFrameSync = true
            Qt.callLater(function() {
                frameList.currentIndex = studyBridge.currentFrame
                root.suppressFrameSync = false
            })
        }
    }
}
