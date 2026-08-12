// Plan 007 U6 — fake settings-bridge surface (see FakePoseBridge.qml header).
// Mirrors SettingsBridge's session-local editor state (the full field
// surface the panel binds) + save/reset call counters.

import QtQuick

QtObject {
    id: root

    property bool dirty: false
    property var trunkCostFunctions: ["DIRECT", "DIRECT_MAHFOUZ"]
    property int trunkCostFunctionIndex: 0
    property double trunkRangeX: 12
    property double trunkRangeY: 12
    property double trunkRangeZ: 15
    property double trunkRangeXA: 15
    property double trunkRangeYA: 15
    property double trunkRangeZA: 15
    property int trunkBudget: 3000
    property int trunkDilation: 6
    property bool trunkHasDilation: true

    property var branchCostFunctions: ["DIRECT", "DIRECT_MAHFOUZ"]
    property int branchCostFunctionIndex: 0
    property double branchRangeX: 12
    property double branchRangeY: 12
    property double branchRangeZ: 15
    property double branchRangeXA: 15
    property double branchRangeYA: 15
    property double branchRangeZA: 15
    property int branchBudget: 0
    property int branchDilation: 6
    property bool branchHasDilation: false
    property int numberBranches: 10
    property bool enableBranch: true

    property var leafCostFunctions: ["DIRECT", "DIRECT_MAHFOUZ"]
    property int leafCostFunctionIndex: 0
    property double leafRangeX: 12
    property double leafRangeY: 12
    property double leafRangeZ: 15
    property double leafRangeXA: 15
    property double leafRangeYA: 15
    property double leafRangeZA: 15
    property int leafBudget: 0
    property int leafDilation: 6
    property bool leafHasDilation: false
    property bool enableLeaf: true

    property int saveCalls: 0
    property int resetCalls: 0

    signal settingsChanged()

    function save() { saveCalls = saveCalls + 1; dirty = false }
    function reset() { resetCalls = resetCalls + 1; dirty = false }
}
