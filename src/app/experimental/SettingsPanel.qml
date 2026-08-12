import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import "."  // Theme

// 005 U5: SettingsPanel — the experiment knobs (R4, R5, R10, R17).
// A scrollable form over the SettingsBridge session-local editor state:
//  - per stage (trunk/branch/leaf): the cost-variant combo (populated from
//    getAvailableCostFunctions via the bridge), the 6 range fields
//    (translations x/y/z + rotations xa/ya/za), the budget, and the
//    dilation field (the active cost function's "Dilation" int parameter —
//    disabled when the active variant has none, e.g. DIRECT_MAHFOUZ);
//  - branch: number-of-branches + enable toggle; leaf: enable toggle;
//  - Save / Reset buttons; the dirty badge shows "● unsaved" while the
//    session differs from the persisted registry (explicit save — review
//    fix), "saved" after Save/Load/Reset-free startup.
//
// The bridge does the work; this file is pure view glue (no logic): every
// field binds a property and commits through its setter. Range fields use a
// SpinBox scaled ×100 (2-decimal knob precision; the bridge keeps the full
// double — lossless persistence is pinned in experimental_settings_test.cpp).
// 007 U2: theme import + token re-point (colors + pinned type scale), the
// invisible-badge-pill fix (review D-01: the pill Rectangle collapsed to
// 0 width in its RowLayout), and the scrollbar-gutter width fix (review
// D-03: availableWidth instead of root.width - 18).

Item {
    id: root

    // 007 U6 (D1): injected bridge surface — the composition root passes
    // the real bridge; tests pass a fake. No context-property coupling.
    required property var settingsBridge

    // Testability (plan 007 U6): the form controls carry objectNames so
    // the Qt Quick Test can reach them via findChild (binding-mirror +
    // ×100-scale + dirty-badge + enablement pins). The inline component
    // instances pass their own controlName through to the inner SpinBox.
    readonly property string dirtyLabelObjectName: "settingsDirtyLabel"
    readonly property string saveButtonObjectName: "settingsSaveButton"
    readonly property string resetButtonObjectName: "settingsResetButton"
    readonly property string branchEnableObjectName: "settingsBranchEnable"
    readonly property string leafEnableObjectName: "settingsLeafEnable"

    ColumnLayout {
        id: formRoot
        anchors.fill: parent
        anchors.margins: Theme.spacingSm
        spacing: Theme.spacingSm

        // ---- Header: title + dirty/unsaved badge ------------------------
        RowLayout {
            Layout.fillWidth: true
            spacing: Theme.spacingXs
            Label {
                text: qsTr("Settings")
                color: Theme.fg
                font.bold: true
                font.pixelSize: Theme.label
            }
            Item { Layout.fillWidth: true }
            Rectangle {
                Layout.preferredHeight: 14
                Layout.preferredWidth: Math.max(dirtyLabel.implicitWidth + 12, 28)
                radius: 7
                color: root.settingsBridge.dirty ? Theme.badgeDirtyBg
                                            : Theme.badgeCleanBg
                Accessible.role: Accessible.StatusBar
                Label {
                    id: dirtyLabel
                    objectName: root.dirtyLabelObjectName
                    anchors.centerIn: parent
                    text: root.settingsBridge.dirty ? qsTr("● unsaved")
                                               : qsTr("saved")
                    color: root.settingsBridge.dirty ? Theme.badgeDirtyFg
                                                : Theme.badgeCleanFg
                    font.pixelSize: Theme.caption
                }
            }
        }


        // ---- The form (scrolls when the column is short) -----------------
        ScrollView {
            id: settingsScroll
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true

            ColumnLayout {
                id: formColumn
                width: settingsScroll.availableWidth
                spacing: Theme.spacingSm

                // ---- Reusable widgets ----------------------------------
                // 2-decimal range field over a ×100 SpinBox.
                component RangeField: ColumnLayout {
                    id: rangeField
                    property string fieldLabel: ""
                    property double fieldValue: 0
                    property var commit: null  // function(double) -> bridge setter
                    // 007 U6: testability passthrough — the inner SpinBox
                    // carries this as its objectName when set.
                    property string controlName: ""
                    Layout.fillWidth: true
                    spacing: 1
                    Label {
                        text: rangeField.fieldLabel
                        color: Theme.fgMuted
                        font.pixelSize: Theme.caption
                    }
                    SpinBox {
                        id: rangeSpin
                        objectName: rangeField.controlName
                        Layout.fillWidth: true
                        editable: true
                        from: -100000
                        to: 100000
                        stepSize: 50
                        value: Math.round(rangeField.fieldValue * 100)
                        textFromValue: function(v) { return (v / 100).toFixed(2) }
                        valueFromText: function(t) {
                            let n = parseFloat(t)
                            return isNaN(n) ? rangeSpin.value : Math.round(n * 100)
                        }
                        onValueModified: {
                            if (rangeField.commit) {
                                rangeField.commit(rangeSpin.value / 100)
                            }
                        }
                    }
                }

                // Integer field over a plain SpinBox.
                component IntField: ColumnLayout {
                    id: intField
                    property string fieldLabel: ""
                    property int fieldValue: 0
                    property var commit: null  // function(int) -> bridge setter
                    property int minValue: 0
                    property int maxValue: 100000
                    // 007 U6: testability passthrough.
                    property string controlName: ""
                    Layout.fillWidth: true
                    spacing: 1
                    Label {
                        text: intField.fieldLabel
                        color: Theme.fgMuted
                        font.pixelSize: Theme.caption
                    }
                    SpinBox {
                        objectName: intField.controlName
                        Layout.fillWidth: true
                        editable: true
                        from: intField.minValue
                        to: intField.maxValue
                        value: intField.fieldValue
                        onValueModified: {
                            if (intField.commit) {
                                intField.commit(value)
                            }
                        }
                    }
                }

                // Cost-variant combo (populated from the manager's
                // available cost functions through the bridge).
                component CostVariantCombo: ColumnLayout {
                    id: combo
                    property var model: []
                    property int current: 0
                    property var commit: null  // function(int) -> bridge setter
                    Layout.fillWidth: true
                    spacing: 1
                    Label {
                        text: qsTr("Cost variant")
                        color: Theme.fgMuted
                        font.pixelSize: Theme.caption
                    }
                    ComboBox {
                        Layout.fillWidth: true
                        model: combo.model
                        currentIndex: combo.current
                        onActivated: {
                            if (combo.commit) {
                                combo.commit(currentIndex)
                            }
                        }
                    }
                }

                // ---- Trunk -------------------------------------------------
                Label {
                    text: qsTr("Trunk")
                    color: Theme.fg
                    font.bold: true
                    font.pixelSize: Theme.label
                }
                CostVariantCombo {
                    model: root.settingsBridge.trunkCostFunctions
                    current: root.settingsBridge.trunkCostFunctionIndex
                    commit: function(i) { root.settingsBridge.trunkCostFunctionIndex = i }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    RangeField {
                        fieldLabel: "X"
                        controlName: "trunkRangeX"
                        fieldValue: root.settingsBridge.trunkRangeX
                        commit: function(v) { root.settingsBridge.trunkRangeX = v }
                    }
                    RangeField {
                        fieldLabel: "Y"
                        fieldValue: root.settingsBridge.trunkRangeY
                        commit: function(v) { root.settingsBridge.trunkRangeY = v }
                    }
                    RangeField {
                        fieldLabel: "Z"
                        fieldValue: root.settingsBridge.trunkRangeZ
                        commit: function(v) { root.settingsBridge.trunkRangeZ = v }
                    }
                    RangeField {
                        fieldLabel: "XA"
                        fieldValue: root.settingsBridge.trunkRangeXA
                        commit: function(v) { root.settingsBridge.trunkRangeXA = v }
                    }
                    RangeField {
                        fieldLabel: "YA"
                        fieldValue: root.settingsBridge.trunkRangeYA
                        commit: function(v) { root.settingsBridge.trunkRangeYA = v }
                    }
                    RangeField {
                        fieldLabel: "ZA"
                        fieldValue: root.settingsBridge.trunkRangeZA
                        commit: function(v) { root.settingsBridge.trunkRangeZA = v }
                    }
                }
                IntField {
                    fieldLabel: qsTr("Budget")
                    controlName: "trunkBudget"
                    fieldValue: root.settingsBridge.trunkBudget
                    commit: function(v) { root.settingsBridge.trunkBudget = v }
                }
                IntField {
                    fieldLabel: qsTr("Dilation")
                    controlName: "trunkDilation"
                    fieldValue: root.settingsBridge.trunkDilation
                    commit: function(v) { root.settingsBridge.trunkDilation = v }
                    enabled: root.settingsBridge.trunkHasDilation
                }

                // ---- Branch ------------------------------------------------
                Label {
                    text: qsTr("Branch")
                    color: Theme.fg
                    font.bold: true
                    font.pixelSize: Theme.label
                }
                CostVariantCombo {
                    model: root.settingsBridge.branchCostFunctions
                    current: root.settingsBridge.branchCostFunctionIndex
                    commit: function(i) { root.settingsBridge.branchCostFunctionIndex = i }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    RangeField {
                        fieldLabel: "X"
                        fieldValue: root.settingsBridge.branchRangeX
                        commit: function(v) { root.settingsBridge.branchRangeX = v }
                    }
                    RangeField {
                        fieldLabel: "Y"
                        fieldValue: root.settingsBridge.branchRangeY
                        commit: function(v) { root.settingsBridge.branchRangeY = v }
                    }
                    RangeField {
                        fieldLabel: "Z"
                        fieldValue: root.settingsBridge.branchRangeZ
                        commit: function(v) { root.settingsBridge.branchRangeZ = v }
                    }
                    RangeField {
                        fieldLabel: "XA"
                        fieldValue: root.settingsBridge.branchRangeXA
                        commit: function(v) { root.settingsBridge.branchRangeXA = v }
                    }
                    RangeField {
                        fieldLabel: "YA"
                        fieldValue: root.settingsBridge.branchRangeYA
                        commit: function(v) { root.settingsBridge.branchRangeYA = v }
                    }
                    RangeField {
                        fieldLabel: "ZA"
                        fieldValue: root.settingsBridge.branchRangeZA
                        commit: function(v) { root.settingsBridge.branchRangeZA = v }
                    }
                }
                IntField {
                    fieldLabel: qsTr("Budget")
                    fieldValue: root.settingsBridge.branchBudget
                    commit: function(v) { root.settingsBridge.branchBudget = v }
                }
                IntField {
                    fieldLabel: qsTr("Number of branches")
                    fieldValue: root.settingsBridge.numberBranches
                    minValue: 1
                    maxValue: 20
                    commit: function(v) { root.settingsBridge.numberBranches = v }
                }
                IntField {
                    fieldLabel: qsTr("Dilation")
                    fieldValue: root.settingsBridge.branchDilation
                    commit: function(v) { root.settingsBridge.branchDilation = v }
                    enabled: root.settingsBridge.branchHasDilation
                }
                CheckBox {
                    objectName: root.branchEnableObjectName
                    text: qsTr("Enable branch stage")
                    checked: root.settingsBridge.enableBranch
                    onToggled: root.settingsBridge.enableBranch = checked
                }

                // ---- Leaf ---------------------------------------------------
                Label {
                    text: qsTr("Leaf")
                    color: Theme.fg
                    font.bold: true
                    font.pixelSize: Theme.label
                }
                CostVariantCombo {
                    model: root.settingsBridge.leafCostFunctions
                    current: root.settingsBridge.leafCostFunctionIndex
                    commit: function(i) { root.settingsBridge.leafCostFunctionIndex = i }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    RangeField {
                        fieldLabel: "X"
                        fieldValue: root.settingsBridge.leafRangeX
                        commit: function(v) { root.settingsBridge.leafRangeX = v }
                    }
                    RangeField {
                        fieldLabel: "Y"
                        fieldValue: root.settingsBridge.leafRangeY
                        commit: function(v) { root.settingsBridge.leafRangeY = v }
                    }
                    RangeField {
                        fieldLabel: "Z"
                        fieldValue: root.settingsBridge.leafRangeZ
                        commit: function(v) { root.settingsBridge.leafRangeZ = v }
                    }
                    RangeField {
                        fieldLabel: "XA"
                        fieldValue: root.settingsBridge.leafRangeXA
                        commit: function(v) { root.settingsBridge.leafRangeXA = v }
                    }
                    RangeField {
                        fieldLabel: "YA"
                        fieldValue: root.settingsBridge.leafRangeYA
                        commit: function(v) { root.settingsBridge.leafRangeYA = v }
                    }
                    RangeField {
                        fieldLabel: "ZA"
                        fieldValue: root.settingsBridge.leafRangeZA
                        commit: function(v) { root.settingsBridge.leafRangeZA = v }
                    }
                }
                IntField {
                    fieldLabel: qsTr("Budget")
                    fieldValue: root.settingsBridge.leafBudget
                    commit: function(v) { root.settingsBridge.leafBudget = v }
                }
                IntField {
                    fieldLabel: qsTr("Dilation")
                    fieldValue: root.settingsBridge.leafDilation
                    commit: function(v) { root.settingsBridge.leafDilation = v }
                    enabled: root.settingsBridge.leafHasDilation
                }
                CheckBox {
                    objectName: root.leafEnableObjectName
                    text: qsTr("Enable leaf stage")
                    checked: root.settingsBridge.enableLeaf
                    onToggled: root.settingsBridge.enableLeaf = checked
                }

                // ---- Save / Reset --------------------------------------------
                RowLayout {
                    Layout.fillWidth: true
                    spacing: Theme.spacingXs
                    Button {
                        id: resetButton
                        objectName: root.resetButtonObjectName
                        text: qsTr("Reset")
                        onClicked: root.settingsBridge.reset()
                    }
                    Button {
                        id: saveButton
                        objectName: root.saveButtonObjectName
                        text: qsTr("Save")
                        Layout.fillWidth: true
                        highlighted: root.settingsBridge.dirty
                        onClicked: root.settingsBridge.save()
                    }
                }
            }
        }
    }

    // Plan 007 U4: the settings dialog focuses the first field on open.
    // The first focusable in the form is the trunk cost-variant combo.
    function focusFirstField() {
        const first = findFocusable(formColumn)
        if (first) first.forceActiveFocus()
    }
    function findFocusable(item) {
        if (!item) return null
        if (item instanceof SpinBox || item instanceof ComboBox) return item
        for (let i = 0; i < item.children.length; i++) {
            const hit = findFocusable(item.children[i])
            if (hit) return hit
        }
        return null
    }
}
