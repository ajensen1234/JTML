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

    // Emitted when Run is clicked. The root closes the edit dialogs (so a
    // mid-run settings/pose edit cannot race the run) and calls
    // optimizerBridge.run().
    signal runRequested()

    RowLayout {
        anchors.fill: parent
        anchors.margins: 6
        spacing: 8

        Button {
            text: qsTr("Run")
            enabled: optimizerBridge.canRun
            onClicked: root.runRequested()
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
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("calls %1").arg(optimizerBridge.costCalls)
            color: Theme.fgDim
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("min %1")
                      .arg(optimizerBridge.currentMinimum.toFixed(3))
            color: Theme.fgDim
            font.pixelSize: Theme.caption
        }
    }
}
