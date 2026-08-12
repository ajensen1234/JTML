import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Dialogs
import "."  // Theme + PosesTable

// 007 U2: the Poses dialog (extracted from main.qml, U8 surface): header
// (model context + dirty badge), copy-prev/next + save/load action row,
// the PosesTable, and the inline validation message. Self-contained: the
// pose/kinematics FileDialogs moved in with the dialog (single owner per
// dialog — the wiring contract). Behavior unchanged (U6 run locking; the
// Run button also closes this dialog via root's open()/close()).

Dialog {
    id: root
    title: qsTr("Poses")
    modal: false
    width: 660
    height: 460
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    // 007 U6 (D1): injected bridge surface — the composition root passes
    // the real bridges; tests pass fakes. No context-property coupling.
    required property var poseBridge
    required property var studyBridge
    required property var optimizerBridge

    // Plan 007 U4 (I-13): dirty-close guard. Declared here (was a
    // dangling reference — the app threw on dialog open/close); the root
    // shows the discard-confirm dialog when this signal fires, and sets
    // discardConfirmed before the deliberate Run-button close.
    signal discardRequested()
    property bool discardConfirmed: false

    // U8 pose table (net-new UI — the widgets app never numerically edits
    // poses; review-fixed semantics): rows = frames, 6 editable cells per
    // row (x/y/z/xa/ya/za) for the PRIMARY model's stored poses; each cell
    // commits immediately through SavePose on editingFinished; non-numeric
    // input is rejected with an inline message (the field reverts to the
    // stored value); the dirty badge shows unsaved in-memory edits
    // (cleared by a successful save); copy-prev/next delegate to the
    // pose_copy seam; save/load wrap pose_file_io. Every control is
    // disabled during an optimizer run (the U6 locking — the Run button
    // also closes this dialog; the toolbar opener is disabled too).
    property bool canEditPoses: root.poseBridge.rowCount > 0
                                && root.studyBridge.primaryModelIndex >= 0
                                && !root.optimizerBridge.running

    // ---- Pose/kinematics file actions (U8): the pose_file_io wrappers
    // surface false returns (unwritable path) + parse failures through the
    // single QML Dialog; the bridge keeps the in-memory state.
    FileDialog {
        id: savePoseFileDialog
        title: qsTr("Save Pose")
        fileMode: FileDialog.SaveFile
        nameFilters: ["JTA Pose File (*.jtap)", "Pose File (*.txt)"]
        onAccepted: root.poseBridge.savePoseFile(selectedFile)
    }
    FileDialog {
        id: loadPoseFileDialog
        title: qsTr("Load Pose")
        nameFilters: [
            "JTA Pose File (*.jtap)",
            "JointTrack Pose File (*.jtp)",
            "Pose File (*.txt)"
        ]
        onAccepted: root.poseBridge.loadPoseFile(selectedFile)
    }
    FileDialog {
        id: saveKinematicsFileDialog
        title: qsTr("Save Kinematics")
        fileMode: FileDialog.SaveFile
        nameFilters: ["JTA Kinematics File (*.jtak)", "Kinematics File (*.txt)"]
        onAccepted: root.poseBridge.saveKinematics(selectedFile)
    }
    FileDialog {
        id: loadKinematicsFileDialog
        title: qsTr("Load Kinematics")
        nameFilters: [
            "JTA Kinematics File (*.jtak)",
            "JointTrack Kinematics File (*.jts)",
            "Kinematics File (*.txt)"
        ]
        onAccepted: root.poseBridge.loadKinematics(selectedFile)
    }

    contentItem: ColumnLayout {
        spacing: Theme.spacingSm

        // ---- Header: model context + dirty badge -------------------------
        RowLayout {
            Layout.fillWidth: true
            spacing: Theme.spacingXs
            Label {
                text: root.studyBridge.primaryModelIndex >= 0
                      ? qsTr("Model %1 · %2 frames")
                            .arg(root.studyBridge.primaryModelIndex)
                            .arg(root.poseBridge.rowCount)
                      : qsTr("%1 frames").arg(root.poseBridge.rowCount)
                color: Theme.fg
                font.bold: true
                font.pixelSize: Theme.label
            }
            Item { Layout.fillWidth: true }
            Rectangle {
                Layout.preferredHeight: 14
                Layout.preferredWidth: Math.max(dirtyLabel.implicitWidth + 12, 28)
                radius: 7
                color: root.poseBridge.dirty ? Theme.badgeDirtyBg
                                        : Theme.badgeCleanBg
                Label {
                    id: dirtyLabel
                    anchors.centerIn: parent
                    text: root.poseBridge.dirty ? qsTr("● unsaved")
                                           : qsTr("saved")
                    color: root.poseBridge.dirty ? Theme.badgeDirtyFg
                                            : Theme.badgeCleanFg
                    font.pixelSize: Theme.caption
                }
            }
        }

        // ---- Actions: copy-prev/next + save/load -------------------------
        RowLayout {
            Layout.fillWidth: true
            spacing: Theme.spacingXs
            Button {
                text: qsTr("◀ Copy Prev")
                enabled: root.canEditPoses
                onClicked: root.poseBridge.copyPrevious()
            }
            Button {
                text: qsTr("Copy Next ▶")
                enabled: root.canEditPoses
                onClicked: root.poseBridge.copyNext()
            }
            Item { Layout.fillWidth: true }
            Button {
                text: qsTr("Save Pose…")
                enabled: root.canEditPoses
                onClicked: savePoseFileDialog.open()
            }
            Button {
                text: qsTr("Load Pose…")
                enabled: root.canEditPoses
                onClicked: loadPoseFileDialog.open()
            }
            Button {
                text: qsTr("Save Kin…")
                enabled: root.canEditPoses
                onClicked: saveKinematicsFileDialog.open()
            }
            Button {
                text: qsTr("Load Kin…")
                enabled: root.canEditPoses
                onClicked: loadKinematicsFileDialog.open()
            }
        }

        // 007 U5 (D-04/D-05): Loader-gated table — the PosesTable (6
        // PoseCell TextFields per row) is only built while the dialog is
        // open. Closing destroys it (frees the delegates; reopen is a
        // natural re-sync per D-05). The dirty badge, action row, and
        // validation label stay alive above (they live in this dialog's
        // contentItem, outside the Loader).
        Loader {
            id: poseTableLoader
            Layout.fillWidth: true
            Layout.fillHeight: true
            active: root.visible
            sourceComponent: PosesTable {
                id: poseTable
                // 007 U6 (D1): the Loader cannot satisfy required
                // properties, so the table's bridges are passed at the
                // instantiation site from the dialog's injected props.
                poseBridge: root.poseBridge
                studyBridge: root.studyBridge
                optimizerBridge: root.optimizerBridge
            }
        }

        // ---- Inline validation message (review fix) -----------------------
        Label {
            Layout.fillWidth: true
            visible: root.poseBridge.validationMessage.length > 0
            color: Theme.badge
            font.pixelSize: Theme.caption
            wrapMode: Text.Wrap
            text: root.poseBridge.validationMessage
        }
    }

    // Plan 007 U4: focus + dirty-guard lifecycle. U5: the table id lives
    // inside the Loader's component scope — reach it via loader.item (the
    // old direct `poseTable` reference was a dangling id, fixed here). The
    // Loader creates the item synchronously on open; guard anyway.
    onOpened: {
        root.discardConfirmed = false
        // Initial focus lands on the first editable cell.
        if (poseTableLoader.item) {
            poseTableLoader.item.focusFirstCell()
        }
    }
    onClosed: {
        if (root.poseBridge.dirty && !root.discardConfirmed) {
            root.discardRequested()
        }
    }

    // R2 (review round): the discard-close path calls this BEFORE closing
    // the dialog so the table's in-flight cell edits are dropped instead
    // of committed during teardown (Component.onDestruction honors
    // suppressDestructionCommit). No-op when the table is already
    // destroyed — the Esc-close flow tears the table down before the
    // discard decision, so the caller must invoke this before close().
    function prepareDiscard() {
        if (poseTableLoader.item) {
            poseTableLoader.item.suppressDestructionCommit = true
        }
    }
}
