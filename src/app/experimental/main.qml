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

    // D5 (plan 007 U3): single run-lock source for the shell root — every
    // toolbar control binds to it (the review D-07 found the Camera/Model
    // toggles escaped the original inventory; the spread
    // !optimizerBridge.running bindings are how controls keep escaping
    // the lock).
    readonly property bool runLocked: optimizerBridge.running
    // Plan 007 U4: shell-level unsaved indicator (M6 fold-in) — any dirty
    // settings or pose state shows in the toolbar pill.
    readonly property bool shellDirty: settingsBridge.dirty
                                       || poseBridge.dirty

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

    // Plan 007 U4 (I-13): dirty-close guard. When a dialog with unsaved
    // state closes via Esc/press-outside, this confirm asks before the
    // close stands; No reopens the dialog. The Run-button close is
    // unconditional (the run handler sets discardConfirmed first).
    Dialog {
        id: discardDialog
        title: qsTr("Discard unsaved changes?")
        modal: true
        implicitWidth: 420
        // The dialog whose close triggered this confirm; Yes leaves it
        // closed, No reopens it.
        property var pendingDialog: null
        standardButtons: Dialog.Yes | Dialog.No
        contentItem: Label {
            text: qsTr("This dialog has unsaved changes. "
                       + "Discard them?")
            wrapMode: Text.Wrap
        }
        onAccepted: {
            discardDialog.pendingDialog = null
        }
        onRejected: {
            if (discardDialog.pendingDialog) {
                discardDialog.pendingDialog.open()
            }
            discardDialog.pendingDialog = null
        }
    }

    // ---- Optimizer settings + pose table live in Dialogs (#5) ----------
    Dialog {
        id: settingsDialog
        title: qsTr("Optimizer Settings")
        modal: false
        width: 560
        height: 680
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        // Plan 007 U4 (I-13): set before the Run-button close so the
        // dirty-close guard does not fire for the deliberate close; reset
        // on every open.
        property bool discardConfirmed: false
        contentItem: SettingsPanel {
            id: settingsPanelContent
            width: settingsDialog.availableWidth
            height: settingsDialog.availableHeight
        }
        onOpened: {
            settingsDialog.discardConfirmed = false
            // Plan 007 U4: initial focus lands on the first field.
            settingsPanelContent.focusFirstField()
        }
        onClosed: {
            // Plan 007 U4 (I-13): Esc/press-outside on dirty state asks
            // first; the Run close (discardConfirmed) and a clean dialog
            // close without a guard.
            if (settingsBridge.dirty && !settingsDialog.discardConfirmed) {
                discardDialog.pendingDialog = settingsDialog
                discardDialog.open()
            }
            // Plan 007 U4: focus returns to the opener.
            if (!root.runLocked) settingsOpenButton.forceActiveFocus()
        }
    }
    PosesDialog {
        id: poseDialog
        onDiscardRequested: {
            discardDialog.pendingDialog = poseDialog
            discardDialog.open()
        }
        onClosed: {
            // Plan 007 U4: focus returns to the opener (the run-close
            // path skips it — the Run button keeps focus during a run).
            if (!root.runLocked) posesOpenButton.forceActiveFocus()
        }
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
        spacing: Theme.spacingXs

        // ---- Toolbar: study actions + interaction mode + dialog openers
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            color: Theme.panel
            radius: 4

            RowLayout {
                anchors.fill: parent
                anchors.margins: Theme.spacingSm
                spacing: Theme.spacingXs

                // ---- Cluster 1: study load actions --------------------
                RowLayout {
                    spacing: Theme.spacingXs
                    Button {
                        text: qsTr("Calibration")
                        enabled: !studyBridge.hasCalibration
                                 && !root.runLocked
                        onClicked: calibrationFileDialog.open()
                    }
                    Button {
                        text: qsTr("Images")
                        enabled: !root.runLocked
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
                        enabled: !root.runLocked
                        onClicked: {
                            if (!studyBridge.hasCalibration) {
                                showMessage(qsTr("Error!"),
                                            qsTr("Load Calibration First!"))
                            } else {
                                pickModels()
                            }
                        }
                    }
                }

                // ---- Cluster separators (visual grouping) --------------
                Rectangle {
                    Layout.preferredWidth: 1
                    Layout.preferredHeight: 18
                    color: Theme.border
                    Accessible.ignored: true
                }

                // ---- Cluster 2: calibration status ---------------------
                Label {
                    text: studyBridge.hasCalibration
                          ? (studyBridge.calibratedForBiplane
                             ? qsTr("Calibrated (biplane)")
                             : qsTr("Calibrated (monoplane)"))
                          : qsTr("No calibration")
                    color: studyBridge.hasCalibration
                           ? Theme.ok : Theme.fgMuted
                    font.pixelSize: Theme.caption
                }

                Rectangle {
                    Layout.preferredWidth: 1
                    Layout.preferredHeight: 18
                    color: Theme.border
                    Accessible.ignored: true
                }

                // ---- Cluster 3: interaction mode -----------------------
                // Interaction mode (#1/#4): camera-centric (trackball
                // camera, pivots at the primary model) vs model-centric
                // (rotates the primary model about its center — the
                // widgets app's trackball-actor mode).
                RowLayout {
                    spacing: Theme.spacingXs
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
                        // D5 (plan 007 U3, I4): locked during a run (review
                        // D-07 — the mode toggles escaped the inventory).
                        enabled: !root.runLocked
                        Accessible.name: qsTr("Camera interaction mode")
                        onClicked: viewportPanel.viewport.setInteractionMode(0)
                    }
                    // D-08 (plan 007 U3): an inline `checked:` binding dies on
                    // the first click (AbstractButton + ButtonGroup write the
                    // property imperatively), so programmatic bridge changes
                    // would leave the toggle stale. A Binding object re-asserts
                    // from the bridge value (the single source); the group
                    // keeps click exclusivity. A re-click on the active toggle
                    // re-sets the same bridge value (idempotent — I-12).
                    Binding {
                        target: cameraModeButton
                        property: "checked"
                        value: viewportPanel.viewport.interactionMode === 0
                    }
                    Button {
                        id: modelModeButton
                        text: qsTr("Model")
                        checkable: true
                        // D5 (plan 007 U3, I4): locked during a run.
                        enabled: !root.runLocked
                        Accessible.name: qsTr("Model interaction mode")
                        onClicked: viewportPanel.viewport.setInteractionMode(1)
                    }
                    Binding {
                        target: modelModeButton
                        property: "checked"
                        value: viewportPanel.viewport.interactionMode === 1
                    }
                }

                Rectangle {
                    Layout.preferredWidth: 1
                    Layout.preferredHeight: 18
                    color: Theme.border
                    Accessible.ignored: true
                }

                // ---- Cluster 4: dialog openers -------------------------
                RowLayout {
                    spacing: Theme.spacingXs
                    Button {
                        id: settingsOpenButton
                        text: qsTr("Optimizer Settings…")
                        enabled: !root.runLocked
                        onClicked: settingsDialog.open()
                    }
                    Button {
                        id: posesOpenButton
                        text: qsTr("Poses…")
                        enabled: !root.runLocked
                        onClicked: poseDialog.open()
                    }
                }

                Item { Layout.fillWidth: true }

                // ---- Shell unsaved indicator (plan 007 U4, M6) ---------
                Rectangle {
                    id: shellDirtyBadge
                    Layout.preferredHeight: 14
                    Layout.preferredWidth:
                        Math.max(shellDirtyLabel.implicitWidth + 12, 28)
                    radius: 7
                    color: root.shellDirty ? Theme.badgeDirtyBg
                                           : Theme.badgeCleanBg
                    Accessible.role: Accessible.StatusBar
                    Label {
                        id: shellDirtyLabel
                        anchors.centerIn: parent
                        text: root.shellDirty ? qsTr("● unsaved")
                                              : qsTr("saved")
                        color: root.shellDirty ? Theme.badgeDirtyFg
                                               : Theme.badgeCleanFg
                        font.pixelSize: Theme.caption
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: Theme.spacingXs

            // ---- Left column: study lists + ML strip -------------------
            Rectangle {
                Layout.preferredWidth: 240
                Layout.fillHeight: true
                color: Theme.panel
                radius: 4

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: Theme.spacingSm
                    spacing: Theme.spacingSm

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
                // DisableAll mirror: close the edit dialogs so a mid-run
                // settings/pose edit cannot race the run. The close is
                // deliberate — the dirty-close guard must not fire.
                settingsDialog.discardConfirmed = true
                poseDialog.discardConfirmed = true
                settingsDialog.close()
                poseDialog.close()
                optimizerBridge.run()
            }
        }
    }
}
