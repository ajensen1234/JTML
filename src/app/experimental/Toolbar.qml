import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import "."  // Theme

// 007 R3 (review round): the toolbar — study load actions + calibration
// status + interaction mode + dialog openers + the shell unsaved pill.
// Extracted from main.qml so the harness can pin the load-flow guards
// (calibration-first, replace-request, D10 merge) and the run lock on the
// Camera/Model toggles (review D-07 — they escaped the original lock
// inventory). Pure view glue: it calls the injected bridges + the picker
// directly and EMITS signals for the root-owned dialogs + message channel
// (single-owner rule — no Connections here).
Rectangle {
    id: root

    Layout.fillWidth: true
    // The fixed 40px height overflowed with the U4 label scale (the pills
    // drew over the viewport) — size from the row's implicit height.
    Layout.preferredHeight: toolbarRow.implicitHeight + 2 * Theme.spacingSm
    color: Theme.panel
    radius: 4
    clip: true  // belt-and-braces: nothing may draw over the viewport

    // 007 U6 (D1): injected bridge surface.
    required property var studyBridge
    required property var optimizerBridge
    // The native multi-select picker (FileDialogBridge in the app; a fake
    // exposing getOpenFileNames(title, filter, startDir, purpose) in tests).
    required property var fileDialogBridge
    // The QmlVtkRenderer for the interaction-mode toggles (a fake with
    // interactionMode + setInteractionMode in tests).
    required property var viewport
    // Shell-level unsaved indicator (input — the root owns the aggregation
    // over settings/pose dirty; injected so the harness can pin the pill).
    required property bool shellDirty

    // D5 (plan 007 U3): per-panel run lock — every lockable control binds
    // to this one property (the review found the spread raw bindings are
    // how controls keep escaping the lock).
    readonly property bool runLocked: optimizerBridge.running

    // Testability hooks (RunBar/StudyPanel pattern).
    readonly property string calibrationButtonObjectName: "toolbarCalibrationButton"
    readonly property string imagesButtonObjectName: "toolbarImagesButton"
    readonly property string modelsButtonObjectName: "toolbarModelsButton"
    readonly property string cameraModeButtonObjectName: "toolbarCameraModeButton"
    readonly property string modelModeButtonObjectName: "toolbarModelModeButton"
    readonly property string settingsButtonObjectName: "toolbarSettingsButton"
    readonly property string posesButtonObjectName: "toolbarPosesButton"
    readonly property string dirtyPillObjectName: "toolbarDirtyPill"
    readonly property string dirtyPillLabelObjectName: "toolbarDirtyPillLabel"

    // Load-flow requests the ROOT owns (dialogs + the message channel).
    signal showMessageRequested(string title, string message)
    signal replaceRequested(var paths)   // pendingPaths for the replace dialog
    signal calibrationRequested()        // open the calibration FileDialog
    signal settingsRequested()
    signal posesRequested()

    // Plan 007 U4: the dialog-close focus-return targets these openers
    // (the ids are component-local; the root reaches them via functions).
    function focusSettingsOpener() { settingsOpenButton.forceActiveFocus() }
    function focusPosesOpener() { posesOpenButton.forceActiveFocus() }

    // ---- Study load actions (moved from main.qml verbatim) -------------
    // Multi-select via the native Qt file dialog (FileDialogBridge — Qt's
    // in-process dialog, DontUseNativeDialog: immune to the portal backend
    // dispatch that breaks multi-select on this box; see FileDialogBridge.h).
    function pickImages() {
        const files = root.fileDialogBridge.getOpenFileNames(
                    qsTr("Load Images"),
                    "Image Files (*.tif *.tiff *.png *.TIF *.TIFF *.PNG)",
                    "",
                    "images")
        if (files.length === 0) return
        // A second image set is a new study: confirm, then replace the
        // dataset before loading. (D10: the replace rule keys on
        // frameCount — a models-only dataset merges, not replaces.)
        if (root.studyBridge.frameCount > 0) {
            root.replaceRequested(files)
        } else {
            root.studyBridge.loadImages(files)
        }
    }
    function pickModels() {
        const files = root.fileDialogBridge.getOpenFileNames(
                    qsTr("Load Implant Models"),
                    "CAD File (*.stl *.STL)",
                    "",
                    "models")
        if (files.length === 0) return
        root.studyBridge.loadModels(files)
    }

    RowLayout {
        id: toolbarRow
        anchors.fill: parent
        anchors.margins: Theme.spacingSm
        spacing: Theme.spacingXs

        // ---- Cluster 1: study load actions ----------------------------
        RowLayout {
            spacing: Theme.spacingXs
            Button {
                objectName: root.calibrationButtonObjectName
                text: qsTr("Calibration")
                enabled: !root.studyBridge.hasCalibration
                         && !root.runLocked
                onClicked: root.calibrationRequested()
            }
            Button {
                objectName: root.imagesButtonObjectName
                text: qsTr("Images")
                enabled: !root.runLocked
                onClicked: {
                    if (!root.studyBridge.hasCalibration) {
                        root.showMessageRequested(qsTr("Error!"),
                                                  qsTr("Load Calibration First!"))
                    } else {
                        pickImages()
                    }
                }
            }
            Button {
                objectName: root.modelsButtonObjectName
                text: qsTr("Models")
                enabled: !root.runLocked
                onClicked: {
                    if (!root.studyBridge.hasCalibration) {
                        root.showMessageRequested(qsTr("Error!"),
                                                  qsTr("Load Calibration First!"))
                    } else {
                        pickModels()
                    }
                }
            }
        }

        // ---- Cluster separators (visual grouping) ----------------------
        Rectangle {
            Layout.preferredWidth: 1
            Layout.preferredHeight: 18
            color: Theme.border
            Accessible.ignored: true
        }

        // ---- Cluster 2: calibration status ----------------------------
        Label {
            text: root.studyBridge.hasCalibration
                  ? (root.studyBridge.calibratedForBiplane
                     ? qsTr("Calibrated (biplane)")
                     : qsTr("Calibrated (monoplane)"))
                  : qsTr("No calibration")
            color: root.studyBridge.hasCalibration
                   ? Theme.ok : Theme.fgMuted
            font.pixelSize: Theme.caption
        }

        Rectangle {
            Layout.preferredWidth: 1
            Layout.preferredHeight: 18
            color: Theme.border
            Accessible.ignored: true
        }

        // ---- Cluster 3: interaction mode ------------------------------
        // Interaction mode (#1/#4): camera-centric (trackball camera,
        // pivots at the primary model) vs model-centric (rotates the
        // primary model about its center — the widgets app's
        // trackball-actor mode).
        RowLayout {
            spacing: Theme.spacingXs
            Label {
                text: qsTr("Interact:")
                color: Theme.fgMuted
                font.pixelSize: Theme.caption
            }
            ButtonGroup {
                id: interactGroup
                buttons: [cameraModeButton, modelModeButton]
            }
            Button {
                id: cameraModeButton
                objectName: root.cameraModeButtonObjectName
                text: qsTr("Camera")
                checkable: true
                // D5 (plan 007 U3, I4): locked during a run (review D-07 —
                // the mode toggles escaped the original inventory).
                enabled: !root.runLocked
                Accessible.name: qsTr("Camera interaction mode")
                onClicked: root.viewport.setInteractionMode(0)
            }
            // D-08 (plan 007 U3): an inline `checked:` binding dies on the
            // first click (AbstractButton + ButtonGroup write the property
            // imperatively), so programmatic bridge changes would leave the
            // toggle stale. A Binding object re-asserts from the viewport
            // value (the single source); the group keeps click exclusivity.
            Binding {
                target: cameraModeButton
                property: "checked"
                value: root.viewport.interactionMode === 0
            }
            Button {
                id: modelModeButton
                objectName: root.modelModeButtonObjectName
                text: qsTr("Model")
                checkable: true
                // D5 (plan 007 U3, I4): locked during a run.
                enabled: !root.runLocked
                Accessible.name: qsTr("Model interaction mode")
                onClicked: root.viewport.setInteractionMode(1)
            }
            Binding {
                target: modelModeButton
                property: "checked"
                value: root.viewport.interactionMode === 1
            }
        }

        Rectangle {
            Layout.preferredWidth: 1
            Layout.preferredHeight: 18
            color: Theme.border
            Accessible.ignored: true
        }

        // ---- Cluster 4: dialog openers --------------------------------
        RowLayout {
            spacing: Theme.spacingXs
            Button {
                id: settingsOpenButton
                objectName: root.settingsButtonObjectName
                text: qsTr("Optimizer Settings…")
                enabled: !root.runLocked
                onClicked: root.settingsRequested()
            }
            Button {
                id: posesOpenButton
                objectName: root.posesButtonObjectName
                text: qsTr("Poses…")
                enabled: !root.runLocked
                onClicked: root.posesRequested()
            }
        }

        Item { Layout.fillWidth: true }

        // ---- Shell unsaved indicator (plan 007 U4, M6) -----------------
        Rectangle {
            id: shellDirtyBadge
            objectName: root.dirtyPillObjectName
            Layout.preferredHeight: 14
            Layout.preferredWidth:
                Math.max(shellDirtyLabel.implicitWidth + 12, 28)
            radius: 7
            color: root.shellDirty ? Theme.badgeDirtyBg : Theme.badgeCleanBg
            Accessible.role: Accessible.StatusBar
            Label {
                id: shellDirtyLabel
                objectName: root.dirtyPillLabelObjectName
                anchors.centerIn: parent
                text: root.shellDirty ? qsTr("● unsaved") : qsTr("saved")
                color: root.shellDirty ? Theme.badgeDirtyFg : Theme.badgeCleanFg
                font.pixelSize: Theme.caption
            }
        }
    }
}
