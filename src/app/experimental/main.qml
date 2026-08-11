import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Window
import QtQuick.Dialogs
import "."  // qmldir: singleton Theme
import jtml.experimental 1.0

// 005 U2/U4/U5 shell — plan-005 feedback restructure (#5): the main screen
// holds only the study lists + viewport + progress; optimizer settings and
// the pose table live in button-opened Dialogs (nothing permanent on the
// right edge). Interaction mode toggle (Camera/Model, #1/#4) sits in the
// toolbar. Versionless QML imports (Qt 6 form — context7-verified).
//
// Surfaces (R17): left column = study lists (frame over model) + ML strip
// (U7); center = the single QmlVtkRenderer viewport; bottom = progress (U6);
// toolbar = load actions + interaction mode + dialog openers.

Window {
    id: root
    visible: true
    width: 1280
    height: 800
    title: qsTr("JTML experimental (QML)")
    color: Theme.bg
    Material.theme: Material.Dark
    Material.accent: Theme.accent

    // ---- Study load + replace confirm + error dialogs (U4) ------------
    FileDialog {
        id: calibrationFileDialog
        title: qsTr("Load Calibration")
        nameFilters: ["Calibration File (*.txt)"]
        onAccepted: studyBridge.loadCalibration(selectedFile)
    }
    FileDialog {
        id: imageFileDialog
        title: qsTr("Load Images")
        nameFilters: [
            "Image Files (*.tif *.tiff *.TIF *.TIFF *.png *.PNG)",
            "All files (*)"
        ]
        // OpenFiles = multi-select (Qt 6.7 verified via context7); a second
        // image set is a new study: confirm, then replace the dataset.
        fileMode: FileDialog.OpenFiles
        onAccepted: {
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
        title: qsTr("Load Implant Models")
        nameFilters: ["CAD File (*.stl *.STL)", "All files (*)"]
        fileMode: FileDialog.OpenFiles
        onAccepted: studyBridge.loadModels(selectedFiles)
    }
    Dialog {
        id: replaceDialog
        property var pendingPaths: []
        title: qsTr("Replace dataset?")
        modal: true
        implicitWidth: 420
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

    // ---- Optimizer settings + pose table live in Dialogs (#5) ----------
    Dialog {
        id: settingsDialog
        title: qsTr("Optimizer Settings")
        modal: false
        width: 560
        height: 680
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        contentItem: SettingsPanel {
            width: settingsDialog.availableWidth
            height: settingsDialog.availableHeight
        }
    }
    Dialog {
        id: poseDialog
        title: qsTr("Poses")
        modal: false
        width: 560
        height: 420
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        contentItem: ColumnLayout {
            spacing: 6
            Label {
                text: qsTr("Pose table (U8)")
                color: Theme.fgDim
            }
            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: Theme.surface
                border.color: Theme.border
                Label {
                    anchors.centerIn: parent
                    text: qsTr("pose rows (U8)")
                    color: Theme.fgDim
                }
            }
        }
    }

    // ---- Bridge → view glue (U4) ---------------------------------------
    Connections {
        target: studyBridge
        function onDatasetChanged() {
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

    // ---- Bridge → view glue (U6) ---------------------------------------
    // Optimizer run: the bridge already wrote the scene poses; the glue
    // forwards the pose relays to the renderer's GUI-thread slots, and the
    // error/notice channel reuses the single QML Dialog.
    Connections {
        target: optimizerBridge
        function onPoseUpdated(modelIndex) {
            viewport.updatePose(modelIndex)
        }
        function onFrameOptimized(frameIndex, modelIndex) {
            viewport.updatePose(modelIndex)
        }
        function onMessageRequested(title, message) {
            showMessage(title, message)
        }
        function onDilationBackgroundRequested() {
            // v1 relay: no dilation display mode yet (U7+) — ignored.
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 4

        // ---- Toolbar: study actions + interaction mode + dialog openers
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            color: Theme.panel
            radius: 4

            RowLayout {
                anchors.fill: parent
                anchors.margins: 6
                spacing: 6

                Button {
                    text: qsTr("Calibration")
                    enabled: !studyBridge.hasCalibration
                             && !optimizerBridge.running
                    onClicked: calibrationFileDialog.open()
                }
                Button {
                    text: qsTr("Images")
                    enabled: !optimizerBridge.running
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
                    enabled: !optimizerBridge.running
                    onClicked: {
                        if (!studyBridge.hasCalibration) {
                            showMessage(qsTr("Error!"),
                                        qsTr("Load Calibration First!"))
                        } else {
                            modelFileDialog.open()
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 1
                    color: "transparent"
                }

                Label {
                    text: studyBridge.hasCalibration
                          ? (studyBridge.calibratedForBiplane
                             ? qsTr("Calibrated (biplane)")
                             : qsTr("Calibrated (monoplane)"))
                          : qsTr("No calibration")
                    color: studyBridge.hasCalibration ? Theme.ok : Theme.fgMuted
                    font.pixelSize: 11
                }

                // Interaction mode (#1/#4): camera-centric (trackball camera,
                // pivots at the primary model) vs model-centric (rotates the
                // primary model about its center — the widgets app's
                // trackball-actor mode).
                Label {
                    text: qsTr("Interact:")
                    color: Theme.fgMuted
                    font.pixelSize: 11
                }
                ButtonGroup {
                    id: interactGroup
                    buttons: [cameraModeButton, modelModeButton]
                }
                Button {
                    id: cameraModeButton
                    text: qsTr("Camera")
                    checkable: true
                    checked: viewport.interactionMode === 0
                    onClicked: viewport.setInteractionMode(0)
                }
                Button {
                    id: modelModeButton
                    text: qsTr("Model")
                    checkable: true
                    checked: viewport.interactionMode === 1
                    onClicked: viewport.setInteractionMode(1)
                }

                Button {
                    text: qsTr("Optimizer Settings…")
                    enabled: !optimizerBridge.running
                    onClicked: settingsDialog.open()
                }
                Button {
                    text: qsTr("Poses…")
                    enabled: !optimizerBridge.running
                    onClicked: poseDialog.open()
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 4

            // ---- Left column: study lists + ML strip -------------------
            Rectangle {
                Layout.preferredWidth: 240
                Layout.fillHeight: true
                color: Theme.panel
                radius: 4

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 6
                    spacing: 6

                    // Frame list (delegate selection contract: currentIndex
                    // drives the bridge — no QItemSelectionModel).
                    Label {
                        text: qsTr("Frames (%1)").arg(appBridge.frameCount)
                        color: Theme.fg
                        font.bold: true
                    }
                    ListView {
                        id: frameList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        enabled: !optimizerBridge.running
                        model: studyBridge.frameListModel
                        delegate: Rectangle {
                            width: frameList.width
                            height: 22
                            color: frameList.currentIndex === index
                                   ? Theme.selection : "transparent"
                            Text {
                                anchors.fill: parent
                                anchors.leftMargin: 4
                                verticalAlignment: Text.AlignVCenter
                                color: Theme.fg
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

                    // Model list (multi-select via the bridge-owned set).
                    Label {
                        text: qsTr("Models (%1)").arg(appBridge.modelCount)
                        color: Theme.fg
                        font.bold: true
                    }
                    ListView {
                        id: modelList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        enabled: !optimizerBridge.running
                        model: studyBridge.modelListModel
                        delegate: Rectangle {
                            width: modelList.width
                            height: 22
                            color: studyBridge.selectedModels.indexOf(index) !== -1
                                   ? Theme.selection : "transparent"
                            Text {
                                anchors.fill: parent
                                anchors.leftMargin: 4
                                verticalAlignment: Text.AlignVCenter
                                color: Theme.fg
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
                        color: Theme.fgMuted
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
                        color: Theme.fgDim
                        font.pixelSize: 11
                    }
                }
            }

            // ---- Center: the single main viewport ----------------------
            QmlVtkRenderer {
                id: viewport
                Layout.fillWidth: true
                Layout.fillHeight: true

                // Pre-load shell state (R17): the placeholder covers the
                // viewport until a study loads.
                Rectangle {
                    visible: appBridge.frameCount === 0
                    anchors.fill: parent
                    color: Theme.surface
                    Label {
                        anchors.centerIn: parent
                        width: parent.width - 40
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.Wrap
                        text: qsTr("No study loaded — load a calibration, "
                                   + "then images and models.")
                        color: Theme.fgDim
                    }
                }

                // Debug readout (U3): last applied pose of model 0. The real
                // pose table lives in the Poses dialog (U8).
                Rectangle {
                    visible: viewport.poseReadout.length > 0
                    z: 1
                    width: 260
                    height: 18
                    radius: 3
                    color: Theme.badge
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
        }

        // ---- Bottom: run/stop + live progress (U6) ----------------------
        // Run-state machine drives the buttons: Run enabled in
        // idle/completed/error, Stop while running/stopping (the widgets
        // DisableAll mirror — everything else on the shell is locked while
        // optimizerBridge.running). Progress is driven by the manager's
        // UpdateDisplay bind (stage/calls/min + calls vs cumulative budget).
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 44
            color: Theme.panel
            radius: 4

            RowLayout {
                anchors.fill: parent
                anchors.margins: 6
                spacing: 8

                Button {
                    text: qsTr("Run")
                    enabled: optimizerBridge.canRun
                    onClicked: {
                        // DisableAll mirror: close the edit dialogs so a
                        // mid-run settings/pose edit cannot race the run.
                        settingsDialog.close()
                        poseDialog.close()
                        optimizerBridge.run()
                    }
                }
                Button {
                    text: qsTr("Stop")
                    enabled: optimizerBridge.running
                    onClicked: optimizerBridge.stop()
                }
                ProgressBar {
                    id: progress
                    Layout.fillWidth: true
                    from: 0
                    to: 100
                    value: optimizerBridge.progress * 100
                }
                Label {
                    text: optimizerBridge.stageText
                    color: Theme.fg
                    font.pixelSize: 11
                }
                Label {
                    text: qsTr("calls %1").arg(optimizerBridge.costCalls)
                    color: Theme.fgDim
                    font.pixelSize: 11
                }
                Label {
                    text: qsTr("min %1")
                              .arg(optimizerBridge.currentMinimum.toFixed(3))
                    color: Theme.fgDim
                    font.pixelSize: 11
                }
            }
        }
    }
}
