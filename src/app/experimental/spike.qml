import QtQuick 2.15
import QtQuick.Window 2.15
import jtml.experimental 1.0

Window {
    id: root
    visible: true
    width: 640
    height: 640
    title: qsTr("JTML QML/VTK spike (plan 005 U1)")
    color: "#14161a"

    // A Qt Quick UI element composited OVER the VTK scene: verifies the new
    // QQuickVTKItem composites correctly in the scenegraph (the deprecated
    // QQuickVTKRenderWindow-era integration rendered below all QtQuick
    // elements).
    Rectangle {
        id: banner
        z: 1
        width: 220
        height: 28
        radius: 4
        color: "#c0392b"
        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: 8

        Text {
            anchors.centerIn: parent
            color: "white"
            text: qsTr("QQuickVTKItem spike")
            font.pixelSize: 14
        }
    }

    SpikeVtkItem {
        id: vtkItem
        anchors.fill: parent
    }
}
