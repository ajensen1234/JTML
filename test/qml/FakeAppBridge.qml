// Plan 007 U6 — fake app-bridge surface (see FakePoseBridge.qml header).

import QtQuick

QtObject {
    id: root

    property int frameCount: 0
    property int modelCount: 0

    signal sessionChanged()
}
