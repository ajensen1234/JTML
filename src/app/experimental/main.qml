import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Window
import QtQuick.Dialogs
import "."  // qmldir: singleton Theme + the U8 PoseCell type
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
    // Multi-select pickers (plan-005 feedback — final fix): the native/portal
    // and Qt built-in dialogs both failed to deliver multi-select on this
    // box, so the image/model pickers are pure-QML checkbox pickers with
    // identical behavior on every backend. Calibration stays native (single
    // file — that path never misbehaved).
    MultiFilePicker {
        id: imagePicker
        pickerTitle: qsTr("Load Images")
        nameFilter: ["*.tif", "*.tiff", "*.png", "*.TIF", "*.TIFF", "*.PNG"]
        startFolder: "file://" + studyBridge.homeDir()
        onFilesSelected: function(urls) {
            // A second image set is a new study: confirm, then replace the
            // dataset before loading.
            if (studyBridge.frameCount > 0) {
                replaceDialog.pendingPaths = urls
                replaceDialog.open()
            } else {
                studyBridge.loadImages(urls)
            }
        }
    }
    MultiFilePicker {
        id: modelPicker
        pickerTitle: qsTr("Load Implant Models")
        nameFilter: ["*.stl", "*.STL"]
        startFolder: "file://" + studyBridge.homeDir()
        onFilesSelected: function(urls) {
            studyBridge.loadModels(urls)
        }
    }
    // ---- ML model pickers (U7): per-implant segment .pt + one estimate
    // .pt (the plan's review fix); the loaded path/state is shown in the
    // ML strip. The bridge normalizes file:// URLs to local paths.
    FileDialog {
        id: segFemPtDialog
        title: qsTr("Load Femur Segmentation Model (.pt)")
        nameFilters: ["Torch File (*.pt)", "All files (*)"]
        onAccepted: mlBridge.setSegmentFemPt(selectedFile)
    }
    FileDialog {
        id: segTibPtDialog
        title: qsTr("Load Tibia Segmentation Model (.pt)")
        nameFilters: ["Torch File (*.pt)", "All files (*)"]
        onAccepted: mlBridge.setSegmentTibPt(selectedFile)
    }
    FileDialog {
        id: estimatePtDialog
        title: qsTr("Load Pose Estimation Model (.pt)")
        nameFilters: ["Torch File (*.pt)", "All files (*)"]
        onAccepted: mlBridge.setEstimatePt(selectedFile)
    }
    // ---- Pose/kinematics file actions (U8): the pose_file_io wrappers
    // surface false returns (unwritable path) + parse failures through the
    // single QML Dialog; the bridge keeps the in-memory state.
    FileDialog {
        id: savePoseFileDialog
        title: qsTr("Save Pose")
        fileMode: FileDialog.SaveFile
        nameFilters: ["JTA Pose File (*.jtap)", "Pose File (*.txt)"]
        onAccepted: poseBridge.savePoseFile(selectedFile)
    }
    FileDialog {
        id: loadPoseFileDialog
        title: qsTr("Load Pose")
        nameFilters: [
            "JTA Pose File (*.jtap)",
            "JointTrack Pose File (*.jtp)",
            "Pose File (*.txt)"
        ]
        onAccepted: poseBridge.loadPoseFile(selectedFile)
    }
    FileDialog {
        id: saveKinematicsFileDialog
        title: qsTr("Save Kinematics")
        fileMode: FileDialog.SaveFile
        nameFilters: ["JTA Kinematics File (*.jtak)", "Kinematics File (*.txt)"]
        onAccepted: poseBridge.saveKinematics(selectedFile)
    }
    FileDialog {
        id: loadKinematicsFileDialog
        title: qsTr("Load Kinematics")
        nameFilters: [
            "JTA Kinematics File (*.jtak)",
            "JointTrack Kinematics File (*.jts)",
            "Kinematics File (*.txt)"
        ]
        onAccepted: poseBridge.loadKinematics(selectedFile)
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

    // Loaded-.pt label helper: the last path segment (the full path is in
    // the bridge's property; the strip shows the basename to fit the
    // 240px column).
    function baseName(path) {
        var parts = String(path).split('/')
        return parts[parts.length - 1]
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
        width: 660
        height: 460
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

        // U8 pose table (net-new UI — the widgets app never numerically
        // edits poses; review-fixed semantics): rows = frames, 6 editable
        // cells per row (x/y/z/xa/ya/za) for the PRIMARY model's stored
        // poses; each cell commits immediately through SavePose on
        // editingFinished; non-numeric input is rejected with an inline
        // message (the field reverts to the stored value); the dirty badge
        // shows unsaved in-memory edits (cleared by a successful save);
        // copy-prev/next delegate to the pose_copy seam; save/load wrap
        // pose_file_io. Every control is disabled during an optimizer run
        // (the U6 locking — the Run button also closes this dialog; the
        // toolbar opener is disabled too).
        property bool canEditPoses: poseBridge.rowCount > 0
                                    && studyBridge.primaryModelIndex >= 0
                                    && !optimizerBridge.running

        contentItem: ColumnLayout {
            spacing: 6

            // ---- Header: model context + dirty badge -------------------
            RowLayout {
                Layout.fillWidth: true
                spacing: 6
                Label {
                    text: studyBridge.primaryModelIndex >= 0
                          ? qsTr("Model %1 · %2 frames")
                                .arg(studyBridge.primaryModelIndex)
                                .arg(poseBridge.rowCount)
                          : qsTr("%1 frames").arg(poseBridge.rowCount)
                    color: Theme.fg
                    font.bold: true
                }
                Item { Layout.fillWidth: true }
                Rectangle {
                    Layout.preferredHeight: 14
                    radius: 7
                    color: poseBridge.dirty ? "#e5b567" : "#3a4a3d"
                    Label {
                        anchors.centerIn: parent
                        text: poseBridge.dirty ? qsTr("● unsaved")
                                               : qsTr("saved")
                        color: poseBridge.dirty ? "#2a2118" : "#8fbf96"
                        font.pixelSize: 10
                    }
                }
            }

            // ---- Actions: copy-prev/next + save/load -------------------
            RowLayout {
                Layout.fillWidth: true
                spacing: 4
                Button {
                    text: qsTr("◀ Copy Prev")
                    enabled: poseDialog.canEditPoses
                    onClicked: poseBridge.copyPrevious()
                }
                Button {
                    text: qsTr("Copy Next ▶")
                    enabled: poseDialog.canEditPoses
                    onClicked: poseBridge.copyNext()
                }
                Item { Layout.fillWidth: true }
                Button {
                    text: qsTr("Save Pose…")
                    enabled: poseDialog.canEditPoses
                    onClicked: savePoseFileDialog.open()
                }
                Button {
                    text: qsTr("Load Pose…")
                    enabled: poseDialog.canEditPoses
                    onClicked: loadPoseFileDialog.open()
                }
                Button {
                    text: qsTr("Save Kin…")
                    enabled: poseDialog.canEditPoses
                    onClicked: saveKinematicsFileDialog.open()
                }
                Button {
                    text: qsTr("Load Kin…")
                    enabled: poseDialog.canEditPoses
                    onClicked: loadKinematicsFileDialog.open()
                }
            }

            // ---- Column header (fixed widths match the cell fields) ----
            RowLayout {
                Layout.fillWidth: true
                spacing: 4
                Label {
                    text: qsTr("Frame")
                    width: 44
                    horizontalAlignment: Text.AlignRight
                    color: Theme.fgMuted
                    font.pixelSize: 11
                }
                Label {
                    text: qsTr("X")
                    width: 78
                    horizontalAlignment: Text.AlignHCenter
                    color: Theme.fgMuted
                    font.pixelSize: 11
                }
                Label {
                    text: qsTr("Y")
                    width: 78
                    horizontalAlignment: Text.AlignHCenter
                    color: Theme.fgMuted
                    font.pixelSize: 11
                }
                Label {
                    text: qsTr("Z")
                    width: 78
                    horizontalAlignment: Text.AlignHCenter
                    color: Theme.fgMuted
                    font.pixelSize: 11
                }
                Label {
                    text: qsTr("XA")
                    width: 78
                    horizontalAlignment: Text.AlignHCenter
                    color: Theme.fgMuted
                    font.pixelSize: 11
                }
                Label {
                    text: qsTr("YA")
                    width: 78
                    horizontalAlignment: Text.AlignHCenter
                    color: Theme.fgMuted
                    font.pixelSize: 11
                }
                Label {
                    text: qsTr("ZA")
                    width: 78
                    horizontalAlignment: Text.AlignHCenter
                    color: Theme.fgMuted
                    font.pixelSize: 11
                }
            }

            // ---- Empty states -------------------------------------------
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

            // ---- The editable table (scrolls when the column is short) --
            ScrollView {
                id: poseScroll
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                // Hidden only without a dataset + primary model; during an
                // optimizer run it stays visible but disabled (the U6
                // locking — the Run button also closes this dialog).
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
                                width: 44
                                horizontalAlignment: Text.AlignRight
                                color: Theme.fgMuted
                                font.pixelSize: 11
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

            // ---- Inline validation message (review fix) -----------------
            Label {
                Layout.fillWidth: true
                visible: poseBridge.validationMessage.length > 0
                color: Theme.badge
                font.pixelSize: 11
                wrapMode: Text.Wrap
                text: poseBridge.validationMessage
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
        function onViewerPoseApplied(sceneModelIndex) {
            // Refresh the readout + idempotently re-apply the synced pose.
            viewport.updatePose(sceneModelIndex)
        }
    }

    // Model-centric pose sync (plan-005 feedback #2): the renderer reports
    // the interaction-end transform; the bridge writes it into the storage
    // + scene (the optimizer starts from the visually arranged pose).
    Connections {
        target: viewport
        function onModelPoseAdjusted(sceneModelIndex, x, y, z, xa, ya, za) {
            studyBridge.applyViewerPose(
                        sceneModelIndex, x, y, z, xa, ya, za)
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

    // ---- Bridge → view glue (U7) ---------------------------------------
    // ML strip: the bridge wrote the scene (background mode after a
    // segment, model pose after an estimate); the glue forwards the scene
    // relays to the renderer's GUI-thread slots; the error/notice channel
    // reuses the single QML Dialog.
    Connections {
        target: mlBridge
        function onMessageRequested(title, message) {
            showMessage(title, message)
        }
        function onSceneBackgroundChanged() {
            viewport.updateBackground()
        }
        function onPoseEstimated(modelIndex) {
            viewport.updatePose(modelIndex)
        }
    }

    // ---- Bridge → view glue (U8) ---------------------------------------
    // Pose table: the bridge wrote the scene pose for a mutation on the
    // current frame; the glue forwards the relay to the renderer's GUI-
    // thread slot; the error/notice channel reuses the single QML Dialog.
    Connections {
        target: poseBridge
        function onMessageRequested(title, message) {
            showMessage(title, message)
        }
        function onScenePoseChanged(modelIndex) {
            viewport.updatePose(modelIndex)
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
                            imagePicker.open()
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
                            modelPicker.open()
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

                    // ---- ML controls strip (U7) ------------------------
                    // Per-implant .pt pickers (femur/tibia segment + one
                    // estimate model — the plan's review fix) with loaded-
                    // path labels; Segment/Estimate act on the CURRENT
                    // frame only (v1 loop scope); the estimate seeds the
                    // optimizer; graceful degradation without models
                    // (AE4 — buttons disabled with a hint label, clear
                    // message if invoked anyway, plain-optimize path
                    // untouched). Disabled-during-run respects the U6
                    // locking.
                    Label {
                        text: qsTr("ML models")
                        color: Theme.fg
                        font.bold: true
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Button {
                            text: qsTr("Femur…")
                            enabled: !optimizerBridge.running
                            onClicked: segFemPtDialog.open()
                        }
                        Label {
                            Layout.fillWidth: true
                            elide: Text.ElideMiddle
                            color: mlBridge.segmentFemPt.length > 0
                                   ? Theme.fg : Theme.fgMuted
                            font.pixelSize: 10
                            text: mlBridge.segmentFemPt.length > 0
                                  ? baseName(mlBridge.segmentFemPt)
                                  : qsTr("not set")
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Button {
                            text: qsTr("Tibia…")
                            enabled: !optimizerBridge.running
                            onClicked: segTibPtDialog.open()
                        }
                        Label {
                            Layout.fillWidth: true
                            elide: Text.ElideMiddle
                            color: mlBridge.segmentTibPt.length > 0
                                   ? Theme.fg : Theme.fgMuted
                            font.pixelSize: 10
                            text: mlBridge.segmentTibPt.length > 0
                                  ? baseName(mlBridge.segmentTibPt)
                                  : qsTr("not set")
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Button {
                            text: qsTr("Estimate…")
                            enabled: !optimizerBridge.running
                            onClicked: estimatePtDialog.open()
                        }
                        Label {
                            Layout.fillWidth: true
                            elide: Text.ElideMiddle
                            color: mlBridge.estimatePt.length > 0
                                   ? Theme.fg : Theme.fgMuted
                            font.pixelSize: 10
                            text: mlBridge.estimatePt.length > 0
                                  ? baseName(mlBridge.estimatePt)
                                  : qsTr("not set")
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Button {
                            text: qsTr("Segment")
                            Layout.fillWidth: true
                            enabled: !optimizerBridge.running
                                     && studyBridge.hasDataset
                                     && studyBridge.currentFrame >= 0
                                     && mlBridge.hasSegmentModel
                            onClicked: mlBridge.segmentCurrentFrame()
                        }
                        Button {
                            text: qsTr("Estimate")
                            Layout.fillWidth: true
                            enabled: !optimizerBridge.running
                                     && studyBridge.hasDataset
                                     && studyBridge.currentFrame >= 0
                                     && studyBridge.primaryModelIndex >= 0
                                     && mlBridge.hasEstimateModel
                            onClicked: mlBridge.estimateCurrentFrame()
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        CheckBox {
                            text: qsTr("Black sil.")
                            font.pixelSize: 10
                            checked: mlBridge.blackSilhouette
                            onToggled: mlBridge.blackSilhouette = checked
                        }
                        Label {
                            text: qsTr("Implant:")
                            color: Theme.fgMuted
                            font.pixelSize: 10
                        }
                        ButtonGroup {
                            id: implantGroup
                            buttons: [femurKindButton, tibiaKindButton]
                        }
                        Button {
                            id: femurKindButton
                            text: qsTr("Fem")
                            checkable: true
                            checked: mlBridge.implantKind === 0
                            font.pixelSize: 10
                            implicitWidth: 40
                            onClicked: mlBridge.implantKind = 0
                        }
                        Button {
                            id: tibiaKindButton
                            text: qsTr("Tib")
                            checkable: true
                            checked: mlBridge.implantKind === 1
                            font.pixelSize: 10
                            implicitWidth: 40
                            onClicked: mlBridge.implantKind = 1
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Label {
                            text: qsTr("View:")
                            color: Theme.fgMuted
                            font.pixelSize: 10
                        }
                        ButtonGroup {
                            id: viewGroup
                            buttons: [origViewButton, segViewButton]
                        }
                        Button {
                            id: origViewButton
                            text: qsTr("Orig")
                            checkable: true
                            checked: mlBridge.backgroundMode === 0
                            enabled: !optimizerBridge.running
                            font.pixelSize: 10
                            implicitWidth: 44
                            onClicked: mlBridge.setBackgroundMode(0)
                        }
                        Button {
                            id: segViewButton
                            text: qsTr("Seg")
                            checkable: true
                            checked: mlBridge.backgroundMode === 1
                            enabled: !optimizerBridge.running
                            font.pixelSize: 10
                            implicitWidth: 44
                            onClicked: mlBridge.setBackgroundMode(1)
                        }
                    }
                    // Estimate result display (the pose that seeds the
                    // optimizer).
                    Label {
                        Layout.fillWidth: true
                        visible: mlBridge.hasEstimate
                        elide: Text.ElideMiddle
                        color: Theme.ok
                        font.pixelSize: 10
                        text: mlBridge.estimateText
                    }
                    // Status/hint label (the AE4 degradation surface).
                    Label {
                        Layout.fillWidth: true
                        visible: mlBridge.statusText.length > 0
                        color: Theme.fgDim
                        font.pixelSize: 10
                        wrapMode: Text.Wrap
                        text: mlBridge.statusText
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
