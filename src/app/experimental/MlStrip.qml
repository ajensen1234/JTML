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
    spacing: Theme.spacingXs

    // 007 U6 (D1): injected bridge surface — the composition root passes
    // the real bridges; tests pass fakes. No context-property coupling.
    required property var mlBridge
    required property var studyBridge
    required property var optimizerBridge

    // Testability (plan 007 U6): the action buttons are reachable from
    // the Qt Quick Test via findChild (degradation matrix + run-lock pins).
    readonly property string segmentButtonObjectName: "mlSegmentButton"
    readonly property string estimateButtonObjectName: "mlEstimateButton"
    readonly property string blackSilButtonObjectName: "mlBlackSilButton"
    readonly property string femurKindButtonObjectName: "mlFemButton"
    readonly property string tibiaKindButtonObjectName: "mlTibButton"

    // D5 (plan 007 U3): single run-lock source for this panel — every
    // locked control binds to it (Black-sil., Fem/Tib, and the view
    // toggles joined the matrix in U3; the review found the spread
    // !optimizerBridge.running bindings are how controls keep escaping
    // the lock).
    readonly property bool runLocked: root.optimizerBridge.running

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
        onAccepted: root.mlBridge.setSegmentFemPt(selectedFile)
    }
    FileDialog {
        id: segTibPtDialog
        title: qsTr("Load Tibia Segmentation Model (.pt)")
        nameFilters: ["Torch File (*.pt)", "All files (*)"]
        onAccepted: root.mlBridge.setSegmentTibPt(selectedFile)
    }
    FileDialog {
        id: estimatePtDialog
        title: qsTr("Load Pose Estimation Model (.pt)")
        nameFilters: ["Torch File (*.pt)", "All files (*)"]
        onAccepted: root.mlBridge.setEstimatePt(selectedFile)
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
        spacing: Theme.spacingXs
        Button {
            text: qsTr("Femur…")
            Layout.preferredWidth: 64
            enabled: !root.runLocked
            onClicked: segFemPtDialog.open()
        }
        Label {
            Layout.fillWidth: true
            elide: Text.ElideMiddle
            color: root.mlBridge.segmentFemPt.length > 0
                   ? Theme.fg : Theme.fgMuted
            font.pixelSize: Theme.caption
            text: root.mlBridge.segmentFemPt.length > 0
                  ? baseName(root.mlBridge.segmentFemPt)
                  : qsTr("not set")
        }
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: Theme.spacingXs
        Button {
            text: qsTr("Tibia…")
            Layout.preferredWidth: 64
            enabled: !root.runLocked
            onClicked: segTibPtDialog.open()
        }
        Label {
            Layout.fillWidth: true
            elide: Text.ElideMiddle
            color: root.mlBridge.segmentTibPt.length > 0
                   ? Theme.fg : Theme.fgMuted
            font.pixelSize: Theme.caption
            text: root.mlBridge.segmentTibPt.length > 0
                  ? baseName(root.mlBridge.segmentTibPt)
                  : qsTr("not set")
        }
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: Theme.spacingXs
        Button {
            text: qsTr("Estimate…")
            Layout.preferredWidth: 64
            enabled: !root.runLocked
            onClicked: estimatePtDialog.open()
        }
        Label {
            Layout.fillWidth: true
            elide: Text.ElideMiddle
            color: root.mlBridge.estimatePt.length > 0
                   ? Theme.fg : Theme.fgMuted
            font.pixelSize: Theme.caption
            text: root.mlBridge.estimatePt.length > 0
                  ? baseName(root.mlBridge.estimatePt)
                  : qsTr("not set")
        }
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: Theme.spacingXs
        Button {
            id: segmentButton
            objectName: root.segmentButtonObjectName
            text: qsTr("Segment")
            Layout.fillWidth: true
            enabled: !root.runLocked
                     && root.studyBridge.hasDataset
                     && root.studyBridge.currentFrame >= 0
                     && root.mlBridge.hasSegmentModel
            onClicked: root.mlBridge.segmentCurrentFrame()
        }
        Button {
            id: estimateButton
            objectName: root.estimateButtonObjectName
            text: qsTr("Estimate")
            Layout.fillWidth: true
            enabled: !root.runLocked
                     && root.studyBridge.hasDataset
                     && root.studyBridge.currentFrame >= 0
                     && root.studyBridge.primaryModelIndex >= 0
                     && root.mlBridge.hasEstimateModel
                     // D6 (plan 007 U3, I5): the widgets estimate actions
                     // segment first and REQUIRE the segment model — the
                     // button reflects that (the bridge guard already
                     // messages; the view disables first).
                     && root.mlBridge.hasSegmentModel
            onClicked: root.mlBridge.estimateCurrentFrame()
        }
    }
    // D6 hint (plan 007 U3, I5): when an estimate .pt is set but no
    // segment .pt is, the Estimate button is disabled — the hint explains
    // why (mirrors the bridge's guard message).
    Label {
        Layout.fillWidth: true
        visible: root.mlBridge.hasEstimateModel && !root.mlBridge.hasSegmentModel
        color: Theme.fgDim
        font.pixelSize: Theme.caption
        wrapMode: Text.Wrap
        text: qsTr("Estimate needs a segmentation model too — pick a "
                   + "femur or tibia .pt first.")
    }
    // Plan 007 U4: visual divider between the action pair and the
    // settings sub-group (Black sil. / implant / view).
    Rectangle {
        Layout.fillWidth: true
        Layout.preferredHeight: 1
        color: Theme.border
        Accessible.ignored: true
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: Theme.spacingXs
        CheckBox {
            id: blackSilCheck
            objectName: root.blackSilButtonObjectName
            text: qsTr("Black sil.")
            // Owner fix (2026-08-12): the Material style's label color did
            // not reach this control (black text on the dark panel) — the
            // app's convention is explicit Theme colors, so pin the text
            // via a contentItem override (CheckBox has no `color`/`label`
            // property; the style keeps drawing background + indicator).
            contentItem: Text {
                text: blackSilCheck.text
                color: Theme.fg
                font: blackSilCheck.font
                verticalAlignment: Text.AlignVCenter
                leftPadding: blackSilCheck.indicator
                             ? blackSilCheck.indicator.width
                               + blackSilCheck.spacing
                             : 0
            }
            font.pixelSize: Theme.caption
            checked: root.mlBridge.blackSilhouette
            // D5 (plan 007 U3, I4): locked during a run — a mid-run
            // black-silhouette toggle races the segmentation pipeline.
            enabled: !root.runLocked
            onToggled: root.mlBridge.blackSilhouette = checked
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
            objectName: root.femurKindButtonObjectName
            text: qsTr("Fem")
            checkable: true
            // D5 (plan 007 U3, I4): locked during a run (review D-07 —
            // the implant-kind toggle escaped the original inventory).
            enabled: !root.runLocked
            font.pixelSize: Theme.caption
            implicitWidth: 40
            Accessible.name: qsTr("Femur implant kind")
            onClicked: root.mlBridge.implantKind = 0
        }
        // D-08 (plan 007 U3): Binding re-asserts from the bridge value —
        // an inline `checked:` binding dies on the first click (see the
        // toolbar comment).
        Binding {
            target: femurKindButton
            property: "checked"
            value: root.mlBridge.implantKind === 0
        }
        Button {
            id: tibiaKindButton
            objectName: root.tibiaKindButtonObjectName
            text: qsTr("Tib")
            checkable: true
            // D5 (plan 007 U3, I4): locked during a run (review D-07).
            enabled: !root.runLocked
            font.pixelSize: Theme.caption
            implicitWidth: 40
            Accessible.name: qsTr("Tibia implant kind")
            onClicked: root.mlBridge.implantKind = 1
        }
        Binding {
            target: tibiaKindButton
            property: "checked"
            value: root.mlBridge.implantKind === 1
        }
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: Theme.spacingXs
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
            enabled: !root.runLocked
            font.pixelSize: Theme.caption
            implicitWidth: 44
            Accessible.name: qsTr("Original view")
            onClicked: root.mlBridge.setBackgroundMode(0)
        }
        // D-08 (plan 007 U3): the segment flow calls setBackgroundMode(1)
        // programmatically (MlBridge.cpp), so the buttons MUST re-assert
        // from the bridge — an inline `checked:` binding dies on the first
        // click and would leave the toggle stale after a segment.
        Binding {
            target: origViewButton
            property: "checked"
            value: root.mlBridge.backgroundMode === 0
        }
        Button {
            id: segViewButton
            text: qsTr("Seg")
            checkable: true
            enabled: !root.runLocked
            font.pixelSize: Theme.caption
            implicitWidth: 44
            Accessible.name: qsTr("Segmented view")
            onClicked: root.mlBridge.setBackgroundMode(1)
        }
        Binding {
            target: segViewButton
            property: "checked"
            value: root.mlBridge.backgroundMode === 1
        }
    }
    // Estimate result display (the pose that seeds the optimizer).
    Label {
        Layout.fillWidth: true
        visible: root.mlBridge.hasEstimate
        elide: Text.ElideMiddle
        color: Theme.ok
        font.pixelSize: Theme.caption
        text: root.mlBridge.estimateText
    }
    // Status/hint label (the AE4 degradation surface).
    Label {
        Layout.fillWidth: true
        visible: root.mlBridge.statusText.length > 0
        color: Theme.fgDim
        font.pixelSize: Theme.caption
        wrapMode: Text.Wrap
        text: root.mlBridge.statusText
    }
}
