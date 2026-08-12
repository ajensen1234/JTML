// Plan 007 U6 — fake bridge surfaces for the Qt Quick Test harness.
// Pure QML doubles of the C++ bridges' property/signal surface (the
// exact subset the view components read). No VTK, no backend.

import QtQuick

import QtQml.Models

QtObject {
    id: root

    property ListModel tableModel: ListModel {}
    property int rowCount: tableModel.count
    property bool dirty: false
    property string validationMessage: ""

    // [frame, model, axis, parsedValue] per successful commit.
    property var commitLog: []
    property var copyCalls: 0
    property var saveCalls: 0
    property var loadCalls: 0

    signal poseTableChanged()

    function setPoseValue(frame, model, axis, text) {
        // Mirror the real bridge's validation (review fix 2026-08-12):
        // full-string parse + finite check + axis/frame bounds — a
        // partial-numeric string or Infinity must NOT be accepted here or
        // the QML pins cannot catch over-acceptance regressions.
        const trimmed = String(text).trim()
        if (trimmed === "") {
            validationMessage = "Invalid numeric value"
            return false
        }
        const v = Number(trimmed)
        if (!isFinite(v)) {
            validationMessage = "Invalid numeric value"
            return false
        }
        if (axis < 0 || axis > 5 || frame < 0 || frame >= tableModel.count) {
            validationMessage = "Invalid numeric value"
            return false
        }
        commitLog.push([frame, model, axis, v])
        const keys = ["x", "y", "z", "xa", "ya", "za"]
        const key = keys[axis] !== undefined ? keys[axis] : "x"
        for (let i = 0; i < tableModel.count; ++i) {
            if (tableModel.get(i).frameIndex === frame) {
                tableModel.setProperty(i, key, v)
                break
            }
        }
        validationMessage = ""
        dirty = true
        return true
    }

    function copyPrevious() { copyCalls = copyCalls + 1 }
    function copyNext() { copyCalls = copyCalls + 1 }
    function savePoseFile(path) { saveCalls = saveCalls + 1 }
    function loadPoseFile(path) { loadCalls = loadCalls + 1 }
    function saveKinematics(path) { saveCalls = saveCalls + 1 }
    function loadKinematics(path) { loadCalls = loadCalls + 1 }
}
