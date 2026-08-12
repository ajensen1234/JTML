import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import "."  // Theme

// 007 U2: bottom run bar (extracted from main.qml, U6 surface): Run/Stop +
// live progress (stage/calls/min + calls vs cumulative budget). Run emits
// runRequested — the composition root closes the edit dialogs and starts
// the run (the run bar does not reach into dialog internals; the root owns
// the dialog close contract from plan 005).

Rectangle {
    id: root
    Layout.fillWidth: true
    Layout.preferredHeight: 44
    color: Theme.panel
    radius: 4

    // 007 U6 (D1): injected bridge surface — the composition root passes
    // the real bridge; tests pass a fake. No context-property coupling.
    required property var optimizerBridge

    // Testability (plan 007 U6): the run/stop buttons are reachable from
    // the Qt Quick Test via findChild (run-state pins).
    readonly property string runButtonObjectName: "runBarRunButton"
    readonly property string stopButtonObjectName: "runBarStopButton"

    // Emitted when Run is clicked. The root closes the edit dialogs (so a
    // mid-run settings/pose edit cannot race the run) and calls
    // optimizerBridge.run().
    signal runRequested()

    RowLayout {
        anchors.fill: parent
        anchors.margins: Theme.spacingSm
        spacing: Theme.spacingSm

        // Plan 007 U4: Run is the app's primary action (CTA hierarchy) —
        // the accent highlight marks it; Stop stays secondary.
        Button {
            id: runButton
            objectName: root.runButtonObjectName
            text: qsTr("Run")
            highlighted: true
            enabled: root.optimizerBridge.canRun
            onClicked: root.runRequested()
        }
        Button {
            id: stopButton
            objectName: root.stopButtonObjectName
            text: qsTr("Stop")
            enabled: root.optimizerBridge.running
            onClicked: root.optimizerBridge.stop()
        }
        ProgressBar {
            id: progress
            Layout.fillWidth: true
            from: 0
            to: 100
            value: root.optimizerBridge.progress * 100
        }
        Label {
            text: root.optimizerBridge.stageText
            color: Theme.fg
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("calls %1").arg(root.optimizerBridge.costCalls)
            color: Theme.fgDim
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("min %1")
                      .arg(root.optimizerBridge.currentMinimum.toFixed(3))
            color: Theme.fgDim
            font.pixelSize: Theme.caption
        }
    }
}
