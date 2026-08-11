import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15
import QtQuick.Window 2.15
import QtQuick.Dialogs
import jtml.experimental 1.0

// 005 U2/U4: the jtml_experimental shell. All v1 surfaces (R17) are
// allocated:
//  - left column: study load buttons (U4 — three-action mirror of the
//    widgets buttons: calibration → images → models) + the study lists
//    (frame list over model list, direct-compiled FrameListModel/
//    ModelListModel owned by StudyBridge) + the ML controls strip (U7);
//  - center: the single main viewport — QmlVtkRenderer (QQuickVTKItem, U3)
//    renders the models at pose over the fluoro background; a placeholder
//    overlay covers it until a study loads (U4 pre-load shell state, R17);
//    the small red badge shows the last applied pose of model 0 (debug
//    readout only — the real pose table is U8).
//  - right panel: settings area (U5) stacked over the pose-table area (U8);
//  - bottom: progress bar + run placeholder (U6).
// Functional, not polished (v1). AppBridge is the QML-exposed hub (counts +
// placeholder signals); StudyBridge (U4) drives the study load + the
// delegate-based selection contract (no QItemSelectionModel): frame list =
// currentIndex, model list = multi-select set owned by the bridge (primary =
// first selected).

Window {
    id: root
    visible: true
    width: 1280
    height: 800
    title: qsTr("JTML experimental (QML)")
    color: "#14161a"

    // ---- Study load + replace confirm + error dialogs (U4) ------------
    FileDialog {
        id: calibrationFileDialog
        title: qsTr("Load Calibration")
        nameFilters: ["Calibration File (*.txt)"]
        onAccepted: studyBridge.loadCalibration(selectedFile)
    }
    FileDialog {
        id: imageFileDialog
        title: qsTr("Load Image(s)")
        nameFilters: ["Image File(s) (*.tif *.tiff *.png)"]
        fileMode: FileDialog.OpenFiles
        onAccepted: {
            // A second image set is a new study: confirm, then replace the
            // dataset before loading (review fix; the widgets app appends —
            // the experimental app deliberately replaces).
            if (studyBridge.frameCount > 0) {
                replaceDialog.pendingPaths = selectedFiles
                replaceDialog.open()
            } else {
                studyBridge.loadImages(selectedFiles)
            }
        }
    }
    FileDialog {
        id: modelFileDialog
        title: qsTr("Load Implant Model(s)")
        nameFilters: ["CAD File(s) (*.stl)"]
        fileMode: FileDialog.OpenFiles
        onAccepted: studyBridge.loadModels(selectedFiles)
    }
    Dialog {
        id: replaceDialog
        property var pendingPaths: []
        title: qsTr("Replace dataset?")
        modal: true
        implicitWidth: 420  // break the contentItem implicitWidth loop
        standardButtons: Dialog.Yes | Dialog.No
        contentItem: Label {
            text: qsTr("Loading a new image set replaces the current dataset "
                       + "(frames and models). The calibration stays loaded. "
                       + "Continue?")
            wrapMode: Text.Wrap
        }
        onAccepted: {
            studyBridge.clearDataset()
            studyBridge.loadImages(pendingPaths)
        }
    }
    Dialog {
        id: messageDialog
        property string messageText: ""
        title: ""
        modal: true
        implicitWidth: 420
        standardButtons: Dialog.Ok
        contentItem: Label {
            text: messageDialog.messageText
            wrapMode: Text.Wrap
        }
    }
    function showMessage(title, text) {
        messageDialog.title = title
        messageDialog.messageText = text
        messageDialog.open()
    }

    // ---- Bridge → view glue (U4): scene signals drive the renderer's
    // GUI-thread slots; dataset changes re-point the frame selection.
    Connections {
        target: studyBridge
        function onDatasetChanged() {
            // The list models may be fresh instances (dataset replace):
            // re-sync the frame highlight to the bridge's current frame.
            // Deferred so the model bindings re-evaluate first.
            Qt.callLater(function() {
                frameList.currentIndex = studyBridge.currentFrame
            })
        }
        function onMessageRequested(title, message) {
            showMessage(title, message)
        }
        function onSceneBackgroundChanged() {
            viewport.updateBackground()
        }
        function onSceneModelsChanged() {
            viewport.updateModels()
        }
        function onSceneCameraChanged() {
            viewport.updateCamera()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 4

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 4

            // ---- Left column: load buttons + study lists + ML strip -----
            Rectangle {
                Layout.preferredWidth: 240
                Layout.fillHeight: true
                color: "#1b1e24"
                radius: 4

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 6
                    spacing: 6

                    // Study load (U4): three-action mirror of the widgets
                    // buttons. Load Calibration is one-use (disabled once
                    // calibrated, widgets parity); Load Images / Load Models
                    // guard on calibration with the widgets "Load Calibration
                    // First!" prompt.
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Button {
                            text: qsTr("Calibration")
                            enabled: !studyBridge.hasCalibration
                            onClicked: calibrationFileDialog.open()
                        }
                        Button {
                            text: qsTr("Images")
                            Layout.fillWidth: true
                            onClicked: {
                                if (!studyBridge.hasCalibration) {
                                    showMessage(qsTr("Error!"),
                                                qsTr("Load Calibration First!"))
                                } else {
                                    imageFileDialog.open()
                                }
                            }
                        }
                        Button {
                            text: qsTr("Models")
                            Layout.fillWidth: true
                            onClicked: {
                                if (!studyBridge.hasCalibration) {
                                    showMessage(qsTr("Error!"),
                                                qsTr("Load Calibration First!"))
                                } else {
                                    modelFileDialog.open()
                                }
                            }
                        }
                    }
                    Label {
                        text: studyBridge.hasCalibration
                              ? (studyBridge.calibratedForBiplane
                                 ? qsTr("Calibrated (biplane)")
                                 : qsTr("Calibrated (monoplane)"))
                              : qsTr("No calibration")
                        color: studyBridge.hasCalibration ? "#5a8a5f" : "#8b929c"
                        font.pixelSize: 11
                    }

                    // Frame list (delegate selection contract: currentIndex
                    // drives the bridge — no QItemSelectionModel).
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
                        model: studyBridge.frameListModel
                        delegate: Rectangle {
                            width: frameList.width
                            height: 22
                            color: frameList.currentIndex === index
                                   ? "#2a3f5f" : "transparent"
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
                                onClicked: {
                                    frameList.currentIndex = index
                                    studyBridge.setCurrentFrame(index)
                                }
                            }
                        }
                        highlightFollowsCurrentItem: true
                    }

                    // Model list (multi-select via the bridge-owned set —
                    // toggle on click; the delegate highlight follows
                    // studyBridge.selectedModels).
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
                        model: studyBridge.modelListModel
                        delegate: Rectangle {
                            width: modelList.width
                            height: 22
                            color: studyBridge.selectedModels.indexOf(index) !== -1
                                   ? "#2a3f5f" : "transparent"
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
                        color: "#8b929c"
                        font.pixelSize: 11
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
            // thread). U4: StudyBridge populates the scene (background =
            // current frame's original image, models at stored poses) and
            // signals the renderer slots through the Connections above; the
            // readout badge mirrors the last applied pose of model 0.
            QmlVtkRenderer {
                id: viewport
                Layout.fillWidth: true
                Layout.fillHeight: true

                // Pre-load shell state (R17): the placeholder covers the
                // viewport until a study loads.
                Rectangle {
                    visible: appBridge.frameCount === 0
                    anchors.fill: parent
                    color: "#101216"
                    Label {
                        anchors.centerIn: parent
                        width: parent.width - 40
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.Wrap
                        text: qsTr("No study loaded — load a calibration, "
                                   + "then images and models.")
                        color: "#6b7280"
                    }
                }

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

        // ---- Bottom: progress bar + run placeholder ---------------------
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            color: "#1b1e24"
            radius: 4

            RowLayout {
                anchors.fill: parent
                anchors.margins: 6
                spacing: 8

                // Run placeholder (U6 wires the optimizer; R17: run controls
                // stay disabled until a dataset loads — the placeholder is
                // disabled regardless).
                Button {
                    text: qsTr("Run (U6)")
                    enabled: false
                }
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
