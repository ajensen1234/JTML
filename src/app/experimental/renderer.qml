import QtQuick
import QtQuick.Window
import jtml.experimental 1.0

// 005 U3: the QML render smoke scene — one QmlVtkRenderer filling the
// window. The C++ driver (test/oracle/qml_render_smoke.cpp) binds the
// app-owned ExperimentalScene and drives the renderer's GUI-thread slots
// (applyScene/updatePose/updateBackground) through the dispatch_async
// render-thread contract.

Window {
    id: root
    visible: true
    width: 640
    height: 640
    title: qsTr("JTML QML renderer smoke (plan 005 U3)")
    color: "#14161a"

    QmlVtkRenderer {
        id: vtkItem
        anchors.fill: parent
    }
}
