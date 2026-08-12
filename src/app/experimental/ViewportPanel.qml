import QtQuick
import QtQuick.Controls
import "."  // Theme
import jtml.experimental 1.0

// 007 U2: center viewport region (extracted from main.qml): the single
// QmlVtkRenderer + placeholder + debug readout, plus the scene-glue
// Connections that forward the bridge's scene relays to the renderer's
// GUI-thread slots (the render-thread contract is unchanged — all VTK
// state stays render-thread-owned). Exposes `property alias viewport` so
// the toolbar + root glue reach the renderer without findChild.

Item {
    id: root

    property alias viewport: viewportItem

    // D5 (plan 007 U3): single run-lock source — every locked control on
    // this panel binds to it (the review found the spread
    // !optimizerBridge.running bindings are how controls keep escaping
    // the lock).
    readonly property bool runLocked: optimizerBridge.running

    QmlVtkRenderer {
        id: viewportItem
        anchors.fill: parent
        // D5 (plan 007 U3): block mouse events during a run — a mid-run
        // drag is a semantic clobber (the run would overwrite it).
        enabled: !root.runLocked

        // Pre-load shell state (R17): the placeholder covers the viewport
        // until a study loads.
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

        // Debug readout (U3): last applied pose of model 0. The real pose
        // table lives in the Poses dialog (U8).
        Rectangle {
            visible: viewportItem.poseReadout.length > 0
            z: 1
            width: 260
            height: 18
            radius: 3
            color: Theme.badge
            anchors { top: parent.top; left: parent.left; margins: 6 }

            Text {
                anchors.fill: parent
                anchors.leftMargin: 6
                verticalAlignment: Text.AlignVCenter
                color: Theme.badgeText
                font.pixelSize: Theme.caption
                text: viewportItem.poseReadout
                elide: Text.ElideRight
            }
        }
    }

    // D5 (plan 007 U3): the run lock has a VISIBLE state — dead input on
    // a live-looking scene is the same silent-failure class D4 exists to
    // prevent. Dim + a centered "Running — interaction locked" pill while
    // a run is in flight; clears automatically on completion.
    Rectangle {
        visible: root.runLocked
        anchors.fill: parent
        z: 2
        color: Theme.overlayDim
        radius: 4

        Rectangle {
            anchors.centerIn: parent
            width: label.implicitWidth + 32
            height: label.implicitHeight + 16
            radius: 6
            color: Theme.panel
            border.color: Theme.accent
            border.width: 1

            Label {
                id: label
                anchors.centerIn: parent
                text: qsTr("Running — interaction locked")
                color: Theme.fg
                font.bold: true
                font.pixelSize: Theme.body
            }
        }
    }

    // ---- Scene glue (U4): bridge wrote the scene; forward the relays ----
    Connections {
        target: studyBridge
        function onSceneBackgroundChanged() {
            viewportItem.updateBackground()
        }
        function onSceneModelsChanged() {
            viewportItem.updateModels()
        }
        function onSceneCameraChanged() {
            viewportItem.updateCamera()
        }
        function onViewerPoseApplied(sceneModelIndex) {
            // Refresh the readout + idempotently re-apply the synced pose.
            viewportItem.updatePose(sceneModelIndex)
        }
    }

    // Model-centric pose sync (plan-005 feedback #2): the renderer reports
    // the interaction-end transform; the bridge writes it into the storage
    // + scene (the optimizer starts from the visually arranged pose).
    Connections {
        target: viewportItem
        function onModelPoseAdjusted(sceneModelIndex, x, y, z, xa, ya, za) {
            studyBridge.applyViewerPose(
                        sceneModelIndex, x, y, z, xa, ya, za)
        }
    }

    // Which model the interactor moves (owner feedback 2026-08-11): the
    // model-centric mode follows the session's PRIMARY selection — select
    // a model row and that model becomes the one you can drag. Also keeps
    // the camera-mode pivot on the selected model. Fires on load (the
    // sync tail emits selectionChanged after populate) and on every
    // selection toggle.
    Connections {
        target: studyBridge
        function onSelectionChanged() {
            viewportItem.setActiveModel(studyBridge.primaryModelIndex)
        }
    }

    // Review I-03 guard: if the renderer failed to materialize (GL/xcb
    // failure), surface it instead of silently no-opping the glue.
    Component.onCompleted: {
        if (!viewportItem) {
            console.warn("ViewportPanel: QmlVtkRenderer failed to "
                         + "instantiate — viewport glue will no-op")
        }
    }
}
