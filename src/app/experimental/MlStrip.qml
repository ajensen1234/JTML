import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Dialogs
import "."  // Theme

// 007 U2: ML controls strip (extracted from main.qml, U7 surface):
// per-implant .pt pickers + Segment/Estimate + the black-silhouette /
// implant-kind / view rows + estimate/status labels. Self-contained: the
// three .pt FileDialogs moved in with the strip. Behavior unchanged (AE4
// degradation, run-lock via optimizerBridge.running).

ColumnLayout {
    id: root
    Layout.fillWidth: true
    spacing: 6

    // Loaded-.pt label helper: the last path segment (the full path is in
    // the bridge's property; the strip shows the basename to fit the
    // 240px column).
    function baseName(path) {
        const parts = String(path).split('/')
        return parts[parts.length - 1]
    }

    // ---- .pt pickers (per-implant segment + one estimate) --------------
    // The loaded path/state is shown in the strip; the bridge normalizes
    // file:// URLs to local paths.
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

    // ---- The strip -------------------------------------------------------
    // Per-implant .pt pickers (femur/tibia segment + one estimate model —
    // the plan's review fix) with loaded-path labels; Segment/Estimate act
    // on the CURRENT frame only (v1 loop scope); the estimate seeds the
    // optimizer; graceful degradation without models (AE4 — buttons
    // disabled with a hint label, clear message if invoked anyway,
    // plain-optimize path untouched). Disabled-during-run respects the U6
    // locking.
    Label {
        text: qsTr("ML models")
        color: Theme.fg
        font.bold: true
        font.pixelSize: Theme.label
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
            font.pixelSize: Theme.caption
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
            font.pixelSize: Theme.caption
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
            font.pixelSize: Theme.caption
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
            font.pixelSize: Theme.caption
            checked: mlBridge.blackSilhouette
            onToggled: mlBridge.blackSilhouette = checked
        }
        Label {
            text: qsTr("Implant:")
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
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
            font.pixelSize: Theme.caption
            implicitWidth: 40
            onClicked: mlBridge.implantKind = 0
        }
        Button {
            id: tibiaKindButton
            text: qsTr("Tib")
            checkable: true
            checked: mlBridge.implantKind === 1
            font.pixelSize: Theme.caption
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
            font.pixelSize: Theme.caption
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
            font.pixelSize: Theme.caption
            implicitWidth: 44
            onClicked: mlBridge.setBackgroundMode(0)
        }
        Button {
            id: segViewButton
            text: qsTr("Seg")
            checkable: true
            checked: mlBridge.backgroundMode === 1
            enabled: !optimizerBridge.running
            font.pixelSize: Theme.caption
            implicitWidth: 44
            onClicked: mlBridge.setBackgroundMode(1)
        }
    }
    // Estimate result display (the pose that seeds the optimizer).
    Label {
        Layout.fillWidth: true
        visible: mlBridge.hasEstimate
        elide: Text.ElideMiddle
        color: Theme.ok
        font.pixelSize: Theme.caption
        text: mlBridge.estimateText
    }
    // Status/hint label (the AE4 degradation surface).
    Label {
        Layout.fillWidth: true
        visible: mlBridge.statusText.length > 0
        color: Theme.fgDim
        font.pixelSize: Theme.caption
        wrapMode: Text.Wrap
        text: mlBridge.statusText
    }
}
