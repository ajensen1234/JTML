// Plan 007 U6 — fake study-bridge surface (see FakePoseBridge.qml header).

import QtQuick

import QtQml.Models

QtObject {
    id: root

    property ListModel frameListModel: ListModel {}
    property ListModel modelListModel: ListModel {}
    property int currentFrame: 0
    property int primaryModelIndex: -1
    property var selectedModels: []
    property int selectedModelCount: 0
    property bool hasDataset: false
    property bool hasCalibration: false
    property bool calibratedForBiplane: false
    property int frameCount: 0

    // Every setCurrentFrame/toggleModelSelected call, in order.
    property var setCurrentFrameLog: []
    property var toggleLog: []

    signal datasetChanged()
    signal selectionChanged()
    signal messageRequested(string title, string message)

    function setCurrentFrame(i) {
        setCurrentFrameLog.push(i)
        currentFrame = i
    }

    function toggleModelSelected(i) {
        toggleLog.push(i)
        const idx = selectedModels.indexOf(i)
        if (idx === -1) {
            selectedModels = selectedModels.concat([i])
        } else {
            const copy = selectedModels.slice()
            copy.splice(idx, 1)
            selectedModels = copy
        }
        selectedModelCount = selectedModels.length
        if (selectedModels.length > 0) {
            primaryModelIndex = selectedModels[0]
        } else {
            primaryModelIndex = -1
        }
        selectionChanged()
    }

    function loadCalibration(path) { hasCalibration = true }
    // Load logs (007 R3): the toolbar pins assert the flows reached the
    // bridge (the real bridge parses; the fake records + no-ops).
    property var loadImagesLog: []
    property var loadModelsLog: []
    function loadImages(paths) { loadImagesLog = loadImagesLog.concat(paths) }
    function loadModels(paths) { loadModelsLog = loadModelsLog.concat(paths) }
    function clearDataset() { }
    function applyViewerPose(index, x, y, z, xa, ya, za) { }
}
