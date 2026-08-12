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

    // 007 U6 (D1): the components take injected `required property`
    // bridges with the SAME names as the root-context properties. Passing
    // them down as `bridge: bridge` is self-referential inside the child
    // declaration (an object's own properties are in scope for its
    // initializers — QML binding loop). Alias each at the root so the
    // child declarations bind a distinct name.
    readonly property var appBridgeRef: appBridge
    readonly property var studyBridgeRef: studyBridge
    readonly property var settingsBridgeRef: settingsBridge
    readonly property var optimizerBridgeRef: optimizerBridge
    readonly property var mlBridgeRef: mlBridge
    readonly property var poseBridgeRef: poseBridge
    readonly property var fileDialogBridgeRef: fileDialogBridge

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
    // Multi-select via the native Qt file dialog — the picker logic moved
    // into Toolbar.qml (R3): the toolbar emits showMessageRequested /
    // replaceRequested / calibrationRequested for the root-owned dialogs.
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
        // Review fix (2026-08-12, finding 12): NOT dismissible by Escape —
        // Escape on this confirm would reject() and reopen the dirty dialog
        // (onRejected -> pendingDialog.open()), whose next Esc re-triggers
        // the guard: an unreachable Escape loop. Buttons only.
        closePolicy: Popup.NoAutoClose
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
            // R2 (review round): the discard path must drop in-flight pose
            // edits — tell the Poses dialog to suppress its destruction
            // commits before the close stands (defensive: Agent A adds
            // prepareDiscard to PosesDialog).
            if (discardDialog.pendingDialog === poseDialog
                    && poseDialog.prepareDiscard) {
                poseDialog.prepareDiscard()
            }
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
            // 007 U6 (D1): inject the real bridge at the use site.
            settingsBridge: root.settingsBridgeRef
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
            // Plan 007 U4: focus returns to the opener (R3: inside the
            // toolbar component).
            if (!root.runLocked) toolbar.focusSettingsOpener()
        }
    }
    PosesDialog {
        id: poseDialog
        // 007 U6 (D1): inject the real bridges at the use site.
        poseBridge: root.poseBridgeRef
        studyBridge: root.studyBridgeRef
        optimizerBridge: root.optimizerBridgeRef
        onDiscardRequested: {
            discardDialog.pendingDialog = poseDialog
            discardDialog.open()
        }
        onClosed: {
            // Plan 007 U4: focus returns to the opener (the run-close
            // path skips it — the Run button keeps focus during a run;
            // R3: the opener lives in the toolbar component).
            if (!root.runLocked) toolbar.focusPosesOpener()
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

        // ---- Toolbar (007 R3): extracted to Toolbar.qml so the harness
        // pins the load-flow guards + the Camera/Model run lock. The
        // toolbar emits requests; the root owns the dialogs + message
        // channel. `viewport` is the renderer (declared below — forward
        // id references are fine in QML).
        Toolbar {
            id: toolbar
            studyBridge: root.studyBridgeRef
            optimizerBridge: root.optimizerBridgeRef
            fileDialogBridge: root.fileDialogBridgeRef
            viewport: viewportPanel.viewport
            shellDirty: root.shellDirty
            onShowMessageRequested: (t, m) => showMessage(t, m)
            onReplaceRequested: (paths) => {
                replaceDialog.pendingPaths = paths
                replaceDialog.open()
            }
            onCalibrationRequested: calibrationFileDialog.open()
            onSettingsRequested: settingsDialog.open()
            onPosesRequested: poseDialog.open()
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
                        // 007 U6 (D1): inject the real bridges.
                        appBridge: root.appBridgeRef
                        studyBridge: root.studyBridgeRef
                        optimizerBridge: root.optimizerBridgeRef
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                    }
                    MlStrip {
                        mlBridge: root.mlBridgeRef
                        studyBridge: root.studyBridgeRef
                        optimizerBridge: root.optimizerBridgeRef
                        Layout.fillWidth: true
                    }
                }
            }

            // ---- Center: the single main viewport ----------------------
            ViewportPanel {
                id: viewportPanel
                // 007 U6 (D1): inject the real bridges.
                appBridge: root.appBridgeRef
                studyBridge: root.studyBridgeRef
                optimizerBridge: root.optimizerBridgeRef
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
            optimizerBridge: root.optimizerBridgeRef
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
