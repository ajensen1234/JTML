// Plan 007 U6 — fake optimizer-bridge surface (see FakePoseBridge.qml header).

import QtQuick

QtObject {
    id: root

    property bool running: false
    property bool canRun: true
    property real progress: 0
    property string stageText: ""
    property int costCalls: 0
    property double currentMinimum: 0
    // D4 additive read (plan 007 U3): mirrors OptimizerBridge::hasSeedPose.
    property bool hasSeedPose: false

    property int runCalls: 0
    property int stopCalls: 0

    signal runStateChanged(int state)
    signal messageRequested(string title, string message)

    function run() { runCalls = runCalls + 1 }
    function stop() { stopCalls = stopCalls + 1 }
}
