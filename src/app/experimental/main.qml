import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15
import QtQuick.Window 2.15
import jtml.experimental 1.0

// 005 U2: the jtml_experimental shell. All v1 surfaces (R17) are allocated:
//  - left column: study lists (frame list over model list, direct-compiled
//    FrameListModel/ModelListModel) + the ML controls strip (U7);
//  - center: the single main viewport — QmlVtkRenderer (QQuickVTKItem, U3)
//    renders the models at pose over the fluoro background; the small red
//    badge shows the last applied pose of model 0 (debug readout only — the
//    real pose table is U8).
//  - right panel: settings area (U5) stacked over the pose-table area (U8);
//  - bottom: progress bar (U6).
// Functional, not polished (v1). AppBridge is the QML-exposed hub (counts +
// placeholder signals in U2; the thin per-seam adapters land in U4/U6/U7/U8).

Window {
    id: root
    visible: true
    width: 1280
    height: 800
    title: qsTr("JTML experimental (QML)")
    color: "#14161a"

    ColumnLayout {
        anchors.fill: parent
        spacing: 4

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 4

            // ---- Left column: study lists + ML controls strip -----------
            Rectangle {
                Layout.preferredWidth: 240
                Layout.fillHeight: true
                color: "#1b1e24"
                radius: 4

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 6
                    spacing: 6

                    // Frame list (U4's StudyBridge populates + drives the
                    // current-frame selection; delegate selection contract
                    // via ListView.currentIndex — no QItemSelectionModel).
                    Label {
                        text: qsTr("Frames (%1)").arg(appBridge.frameCount)
                        color: "#cfd3da"
                        font.bold: true
                    }
                    ListView {
                        id: frameList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        model: frameListModel
                        delegate: Rectangle {
                            width: frameList.width
                            height: 22
                            color: ListView.isCurrentItem ? "#2a3f5f" : "transparent"
                            Text {
                                anchors.fill: parent
                                anchors.leftMargin: 4
                                verticalAlignment: Text.AlignVCenter
                                color: "#cfd3da"
                                text: model.display
                                elide: Text.ElideRight
                            }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: frameList.currentIndex = index
                            }
                        }
                        highlightFollowsCurrentItem: true
                    }

                    // Model list (multi-select state comes with U4).
                    Label {
                        text: qsTr("Models (%1)").arg(appBridge.modelCount)
                        color: "#cfd3da"
                        font.bold: true
                    }
                    ListView {
                        id: modelList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        model: modelListModel
                        delegate: Rectangle {
                            width: modelList.width
                            height: 22
                            color: modelList.currentIndex === index ? "#2a3f5f" : "transparent"
                            Text {
                                anchors.fill: parent
                                anchors.leftMargin: 4
                                verticalAlignment: Text.AlignVCenter
                                color: "#cfd3da"
                                text: model.display
                                elide: Text.ElideRight
                            }
                        }
                    }

                    // ML controls strip placeholder (U7 wires the .pt
                    // pickers + segment/estimate buttons + estimate
                    // display; graceful degradation without models).
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Button { text: qsTr("Segment"); enabled: false; Layout.fillWidth: true }
                        Button { text: qsTr("Estimate"); enabled: false; Layout.fillWidth: true }
                    }
                    Label {
                        text: qsTr("ML path (U7)")
                        color: "#6b7280"
                        font.pixelSize: 11
                    }
                }
            }

            // ---- Center: the single main viewport -----------------------
            // U3: QmlVtkRenderer drives the app-owned ExperimentalScene via
            // its GUI-thread slots (dispatch_async to the Qt Quick render
            // thread). The scene is populated by the bridges in later units
            // (StudyBridge U4, PoseBridge U8); the readout badge below
            // mirrors the last applied pose of model 0 for debugging.
            QmlVtkRenderer {
                id: viewport
                Layout.fillWidth: true
                Layout.fillHeight: true

                Rectangle {
                    visible: viewport.poseReadout.length > 0
                    z: 1
                    width: 260
                    height: 18
                    radius: 3
                    color: "#c0392b"
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.margins: 6

                    Text {
                        anchors.fill: parent
                        anchors.leftMargin: 6
                        verticalAlignment: Text.AlignVCenter
                        color: "white"
                        font.pixelSize: 11
                        text: viewport.poseReadout
                        elide: Text.ElideRight
                    }
                }
            }

            // ---- Right panel: settings over pose table ------------------
            Rectangle {
                Layout.preferredWidth: 260
                Layout.fillHeight: true
                color: "#1b1e24"
                radius: 4

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 6
                    spacing: 6

                    // Settings area placeholder (U5): per-stage
                    // ranges/budgets/dilation backed by OptimizerSettings +
                    // settings_constants.h, cost-variant combo per stage
                    // via CostFunctionManager, explicit save via
                    // SettingsService.
                    Label {
                        text: qsTr("Settings (U5)")
                        color: "#cfd3da"
                        font.bold: true
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Label { text: qsTr("Trunk range"); color: "#8b929c" }
                        TextField { text: "35,35,35,35,35,35"; enabled: false; Layout.fillWidth: true }
                        Label { text: qsTr("Trunk budget"); color: "#8b929c" }
                        TextField { text: "20000"; enabled: false; Layout.fillWidth: true }
                    }

                    Item { Layout.fillHeight: true }

                    // Pose-table area placeholder (U8): editable per-frame
                    // pose rows over LocationStorage + pose_file_io +
                    // pose_copy, disabled during an optimizer run.
                    Label {
                        text: qsTr("Pose table (U8)")
                        color: "#cfd3da"
                        font.bold: true
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "#14161a"
                        border.color: "#2a2f38"
                        Label {
                            anchors.centerIn: parent
                            text: qsTr("pose rows (U8)")
                            color: "#6b7280"
                        }
                    }
                }
            }
        }

        // ---- Bottom: progress bar ---------------------------------------
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            color: "#1b1e24"
            radius: 4

            RowLayout {
                anchors.fill: parent
                anchors.margins: 6
                spacing: 8

                // Progress placeholder (U6): stage / calls / current min /
                // pose updates from the OptimizerManager signal binds.
                ProgressBar {
                    id: progress
                    Layout.fillWidth: true
                    from: 0
                    to: 100
                    value: 0
                }
                Label {
                    text: qsTr("Progress (U6)")
                    color: "#6b7280"
                    font.pixelSize: 11
                }
            }
        }
    }
}
