import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import Qt.labs.folderlistmodel
import "."  // qmldir: singleton Theme

// Multi-file picker (plan-005 feedback): the native/portal and Qt built-in
// file dialogs both failed to deliver multi-select on this box (GNOME portal
// backend + the QuickDialogs2 built-in, verified: the portal plugin passes
// multiple=true but the backend still single-selects in practice; the
// built-in's ExtendedSelection path was not reliably reachable either). This
// picker is pure QML with explicit checkboxes — identical behavior on every
// backend, no native dependency, no portal variance.
//
// Returns file:// URLs via filesSelected(); the bridge normalizes them to
// local paths (pinned by the "file:// URLs normalize to local" headless
// test). Folder navigation: double-click a directory to enter; the ↑ button
// goes up. The caller supplies the start folder (AppBridge::homeDir or a
// remembered folder).

Dialog {
    id: root

    property string pickerTitle: qsTr("Select Files")
    property var nameFilter: ["*"]
    property string startFolder: "file:///"
    property var selectedUrls: []

    title: pickerTitle
    modal: true
    width: 720
    height: 520
    closePolicy: Popup.CloseOnEscape

    signal filesSelected(var urls)

    FolderListModel {
        id: folderModel
        folder: root.startFolder
        nameFilters: root.nameFilter
        showDirs: true
        showFiles: true
        showDotAndDotDot: false
        sortField: FolderListModel.Name
        sortReversed: false
    }

    onOpened: {
        selectedUrls = []
        folderModel.folder = startFolder
    }

    contentItem: ColumnLayout {
        spacing: 6

        RowLayout {
            Layout.fillWidth: true
            spacing: 4

            Button {
                text: "\u2191"
                ToolTip.visible: hovered
                ToolTip.text: qsTr("Up")
                onClicked: folderModel.folder = folderModel.parentFolder
            }
            TextField {
                Layout.fillWidth: true
                readOnly: true
                text: folderModel.folder
                color: Theme.fg
            }
        }

        ListView {
            id: fileList
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: folderModel

            delegate: Rectangle {
                width: fileList.width
                height: 26
                color: "transparent"

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 4
                    spacing: 4

                    CheckBox {
                        id: pickBox
                        visible: !model.isDir
                        Layout.preferredWidth: 24
                        checked: root.selectedUrls.indexOf(model.fileURL) !== -1
                        onToggled: {
                            const u = model.fileURL
                            if (checked) {
                                if (root.selectedUrls.indexOf(u) === -1) {
                                    root.selectedUrls = root.selectedUrls.concat([u])
                                }
                            } else {
                                root.selectedUrls = root.selectedUrls.filter(
                                    function(v) { return v !== u })
                            }
                        }
                    }
                    // Label + its own double-click area (kept BELOW the
                    // checkbox in z so checkbox clicks are never swallowed).
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "transparent"

                        Label {
                            anchors.fill: parent
                            anchors.leftMargin: 2
                            verticalAlignment: Text.AlignVCenter
                            text: model.fileName + (model.isDir ? "/" : "")
                            color: model.isDir ? Theme.accent : Theme.fg
                            elide: Text.ElideRight
                        }
                        MouseArea {
                            anchors.fill: parent
                            onDoubleClicked: {
                                if (model.isDir) {
                                    folderModel.folder = model.fileURL
                                }
                            }
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 6

            Label {
                text: qsTr("%1 selected").arg(root.selectedUrls.length)
                color: Theme.fgMuted
            }
            Item { Layout.fillWidth: true }
            Button {
                text: qsTr("Cancel")
                onClicked: root.reject()
            }
            Button {
                text: qsTr("OK")
                enabled: root.selectedUrls.length > 0
                onClicked: {
                    root.filesSelected(root.selectedUrls)
                    root.accept()
                }
            }
        }
    }
}
