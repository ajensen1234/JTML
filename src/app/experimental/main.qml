import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material  // deliberate: the app pins Material Dark
import QtQuick.Dialogs
import "."  // qmldir: singleton Theme + the U2-extracted components
import jtml.experimental 1.0

// 005 U2/U4/U5 shell + 007 U2 bootstrap — the main screen holds only the
// study lists + viewport + progress; settings + pose table live in
// button-opened Dialogs; interaction mode toggle sits in the toolbar.
//
// 007 U2: this file is the composition root. Extracted components:
// StudyPanel (lists), MlStrip (ML + its .pt pickers), ViewportPanel
// (renderer + placeholder + readout + scene glue, `property alias
// viewport`), RunBar (run/stop + progress, emits runRequested),
// PosesDialog (pose table + its pickers), PosesTable (table body). Glue
// split per the wiring contract (single owner per bridge signal
// surface): dataset/frame sync → StudyPanel; scene relays + selection
// → ViewportPanel; this file keeps the studyBridge message channel +
// the optimizer/ml/pose relays (reaching the renderer via
// viewportPanel.viewport).

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
    // Multi-select via the native Qt file dialog (FileDialogBridge — Qt's
    // in-process dialog, DontUseNativeDialog: immune to the portal backend
    // dispatch that breaks multi-select on this box; see FileDialogBridge.h).
    function pickImages() {
        const files = fileDialogBridge.getOpenFileNames(
                    qsTr("Load Images"),
                    "Image Files (*.tif *.tiff *.png *.TIF *.TIFF *.PNG)",
                    "")
        if (files.length === 0) return
        // A second image set is a new study: confirm, then replace the
        // dataset before loading.
        if (studyBridge.frameCount > 0) {
            replaceDialog.pendingPaths = files
            replaceDialog.open()
        } else {
            studyBridge.loadImages(files)
        }
    }
    function pickModels() {
        const files = fileDialogBridge.getOpenFileNames(
                    qsTr("Load Implant Models"),
                    "CAD File (*.stl *.STL)",
                    "")
        if (files.length === 0) return
        studyBridge.loadModels(files)
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
    PosesDialog {
        id: poseDialog
    }

    // ---- Bridge → view glue -------------------------------------------
    // Study message channel (dataset/frame sync → StudyPanel; scene
    // relays + selection → ViewportPanel).
    Connections {
        target: studyBridge
        function onMessageRequested(title, message) {
            showMessage(title, message)
        }
    }

    // Optimizer run (U6): forward the pose relays to the renderer's
    // GUI-thread slots; errors reuse the single QML Dialog.
    Connections {
        target: optimizerBridge
        function onPoseUpdated(modelIndex) {
            viewportPanel.viewport.updatePose(modelIndex)
        }
        function onFrameOptimized(frameIndex, modelIndex) {
            viewportPanel.viewport.updatePose(modelIndex)
        }
        function onMessageRequested(title, message) {
            showMessage(title, message)
        }
        function onDilationBackgroundRequested() {
            // v1 relay: no dilation display mode yet (U7+) — ignored.
        }
    }

    // ML (U7): scene relays → renderer; message channel → Dialog.
    Connections {
        target: mlBridge
        function onMessageRequested(title, message) {
            showMessage(title, message)
        }
        function onSceneBackgroundChanged() {
            viewportPanel.viewport.updateBackground()
        }
        function onPoseEstimated(modelIndex) {
            viewportPanel.viewport.updatePose(modelIndex)
        }
    }

    // Pose table (U8): scene-pose relay → renderer; message channel.
    Connections {
        target: poseBridge
        function onMessageRequested(title, message) {
            showMessage(title, message)
        }
        function onScenePoseChanged(modelIndex) {
            viewportPanel.viewport.updatePose(modelIndex)
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
                            pickImages()
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
                            pickModels()
                        }
                    }
                }

                Item {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 1
                }

                Label {
                    text: studyBridge.hasCalibration
                          ? (studyBridge.calibratedForBiplane
                             ? qsTr("Calibrated (biplane)")
                             : qsTr("Calibrated (monoplane)"))
                          : qsTr("No calibration")
                    color: studyBridge.hasCalibration ? Theme.ok : Theme.fgMuted
                    font.pixelSize: Theme.caption
                }

                // Interaction mode (#1/#4): camera-centric (trackball camera,
                // pivots at the primary model) vs model-centric (rotates the
                // primary model about its center — the widgets app's
                // trackball-actor mode).
                Label {
                    text: qsTr("Interact:")
                    color: Theme.fgMuted
                    font.pixelSize: Theme.caption
                }
                ButtonGroup {
                    id: interactGroup
                    buttons: [cameraModeButton, modelModeButton]
                }
                Button {
                    id: cameraModeButton
                    text: qsTr("Camera")
                    checkable: true
                    checked: viewportPanel.viewport.interactionMode === 0
                    onClicked: viewportPanel.viewport.setInteractionMode(0)
                }
                Button {
                    id: modelModeButton
                    text: qsTr("Model")
                    checkable: true
                    checked: viewportPanel.viewport.interactionMode === 1
                    onClicked: viewportPanel.viewport.setInteractionMode(1)
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

                    StudyPanel {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                    }
                    MlStrip {
                        Layout.fillWidth: true
                    }
                }
            }

            // ---- Center: the single main viewport ----------------------
            ViewportPanel {
                id: viewportPanel
                Layout.fillWidth: true
                Layout.fillHeight: true
            }
        }

        // ---- Bottom: run/stop + live progress (U6) ----------------------
        // Run-state machine drives the buttons: Run enabled in
        // idle/completed/error, Stop while running/stopping (the widgets
        // DisableAll mirror — everything else on the shell is locked while
        // optimizerBridge.running). Run closes the edit dialogs (a mid-run
        // settings/pose edit cannot race the run) then starts the run.
        RunBar {
            Layout.fillWidth: true
            onRunRequested: {
                settingsDialog.close()
                poseDialog.close()
                optimizerBridge.run()
            }
        }
    }
}
