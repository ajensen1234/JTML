// Plan 007 U6 — fake ML-bridge surface (see FakePoseBridge.qml header).

import QtQuick

QtObject {
    id: root

    property string segmentFemPt: ""
    property string segmentTibPt: ""
    property string estimatePt: ""
    property bool hasSegmentModel: false
    property bool hasEstimateModel: false
    property bool hasEstimate: false
    property string estimateText: ""
    property string statusText: ""
    property bool blackSilhouette: false
    property int implantKind: 0
    property int backgroundMode: 0

    property var setFemCalls: 0
    property var setTibCalls: 0
    property var setEstimateCalls: 0
    property var segmentCalls: 0
    property var estimateCalls: 0
    property var backgroundModeLog: []

    signal messageRequested(string title, string message)
    signal sceneBackgroundChanged()

    function setSegmentFemPt(path) { segmentFemPt = String(path); hasSegmentModel = true }
    function setSegmentTibPt(path) { segmentTibPt = String(path); hasSegmentModel = true }
    function setEstimatePt(path) { estimatePt = String(path); hasEstimateModel = true }
    function segmentCurrentFrame() { segmentCalls = segmentCalls + 1 }
    function estimateCurrentFrame() { estimateCalls = estimateCalls + 1 }
    function setBackgroundMode(m) { backgroundMode = m; backgroundModeLog.push(m) }
}
