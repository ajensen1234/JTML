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

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 6
        spacing: 6

        // ---- Header: title + dirty/unsaved badge ------------------------
        RowLayout {
            Layout.fillWidth: true
            spacing: 6
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
                color: settingsBridge.dirty ? Theme.badgeDirtyBg
                                            : Theme.badgeCleanBg
                Label {
                    id: dirtyLabel
                    anchors.centerIn: parent
                    text: settingsBridge.dirty ? qsTr("● unsaved")
                                               : qsTr("saved")
                    color: settingsBridge.dirty ? Theme.badgeDirtyFg
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
                width: settingsScroll.availableWidth
                spacing: 8

                // ---- Reusable widgets ----------------------------------
                // 2-decimal range field over a ×100 SpinBox.
                component RangeField: ColumnLayout {
                    id: rangeField
                    property string fieldLabel: ""
                    property double fieldValue: 0
                    property var commit: null  // function(double) -> bridge setter
                    Layout.fillWidth: true
                    spacing: 1
                    Label {
                        text: rangeField.fieldLabel
                        color: Theme.fgMuted
                        font.pixelSize: Theme.caption
                    }
                    SpinBox {
                        id: rangeSpin
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
                    Layout.fillWidth: true
                    spacing: 1
                    Label {
                        text: intField.fieldLabel
                        color: Theme.fgMuted
                        font.pixelSize: Theme.caption
                    }
                    SpinBox {
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
                    model: settingsBridge.trunkCostFunctions
                    current: settingsBridge.trunkCostFunctionIndex
                    commit: function(i) { settingsBridge.trunkCostFunctionIndex = i }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    RangeField {
                        fieldLabel: "X"
                        fieldValue: settingsBridge.trunkRangeX
                        commit: function(v) { settingsBridge.trunkRangeX = v }
                    }
                    RangeField {
                        fieldLabel: "Y"
                        fieldValue: settingsBridge.trunkRangeY
                        commit: function(v) { settingsBridge.trunkRangeY = v }
                    }
                    RangeField {
                        fieldLabel: "Z"
                        fieldValue: settingsBridge.trunkRangeZ
                        commit: function(v) { settingsBridge.trunkRangeZ = v }
                    }
                    RangeField {
                        fieldLabel: "XA"
                        fieldValue: settingsBridge.trunkRangeXA
                        commit: function(v) { settingsBridge.trunkRangeXA = v }
                    }
                    RangeField {
                        fieldLabel: "YA"
                        fieldValue: settingsBridge.trunkRangeYA
                        commit: function(v) { settingsBridge.trunkRangeYA = v }
                    }
                    RangeField {
                        fieldLabel: "ZA"
                        fieldValue: settingsBridge.trunkRangeZA
                        commit: function(v) { settingsBridge.trunkRangeZA = v }
                    }
                }
                IntField {
                    fieldLabel: qsTr("Budget")
                    fieldValue: settingsBridge.trunkBudget
                    commit: function(v) { settingsBridge.trunkBudget = v }
                }
                IntField {
                    fieldLabel: qsTr("Dilation")
                    fieldValue: settingsBridge.trunkDilation
                    commit: function(v) { settingsBridge.trunkDilation = v }
                    enabled: settingsBridge.trunkHasDilation
                }

                // ---- Branch ------------------------------------------------
                Label {
                    text: qsTr("Branch")
                    color: Theme.fg
                    font.bold: true
                    font.pixelSize: Theme.label
                }
                CostVariantCombo {
                    model: settingsBridge.branchCostFunctions
                    current: settingsBridge.branchCostFunctionIndex
                    commit: function(i) { settingsBridge.branchCostFunctionIndex = i }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    RangeField {
                        fieldLabel: "X"
                        fieldValue: settingsBridge.branchRangeX
                        commit: function(v) { settingsBridge.branchRangeX = v }
                    }
                    RangeField {
                        fieldLabel: "Y"
                        fieldValue: settingsBridge.branchRangeY
                        commit: function(v) { settingsBridge.branchRangeY = v }
                    }
                    RangeField {
                        fieldLabel: "Z"
                        fieldValue: settingsBridge.branchRangeZ
                        commit: function(v) { settingsBridge.branchRangeZ = v }
                    }
                    RangeField {
                        fieldLabel: "XA"
                        fieldValue: settingsBridge.branchRangeXA
                        commit: function(v) { settingsBridge.branchRangeXA = v }
                    }
                    RangeField {
                        fieldLabel: "YA"
                        fieldValue: settingsBridge.branchRangeYA
                        commit: function(v) { settingsBridge.branchRangeYA = v }
                    }
                    RangeField {
                        fieldLabel: "ZA"
                        fieldValue: settingsBridge.branchRangeZA
                        commit: function(v) { settingsBridge.branchRangeZA = v }
                    }
                }
                IntField {
                    fieldLabel: qsTr("Budget")
                    fieldValue: settingsBridge.branchBudget
                    commit: function(v) { settingsBridge.branchBudget = v }
                }
                IntField {
                    fieldLabel: qsTr("Number of branches")
                    fieldValue: settingsBridge.numberBranches
                    minValue: 1
                    maxValue: 20
                    commit: function(v) { settingsBridge.numberBranches = v }
                }
                IntField {
                    fieldLabel: qsTr("Dilation")
                    fieldValue: settingsBridge.branchDilation
                    commit: function(v) { settingsBridge.branchDilation = v }
                    enabled: settingsBridge.branchHasDilation
                }
                CheckBox {
                    text: qsTr("Enable branch stage")
                    checked: settingsBridge.enableBranch
                    onToggled: settingsBridge.enableBranch = checked
                }

                // ---- Leaf ---------------------------------------------------
                Label {
                    text: qsTr("Leaf")
                    color: Theme.fg
                    font.bold: true
                    font.pixelSize: Theme.label
                }
                CostVariantCombo {
                    model: settingsBridge.leafCostFunctions
                    current: settingsBridge.leafCostFunctionIndex
                    commit: function(i) { settingsBridge.leafCostFunctionIndex = i }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    RangeField {
                        fieldLabel: "X"
                        fieldValue: settingsBridge.leafRangeX
                        commit: function(v) { settingsBridge.leafRangeX = v }
                    }
                    RangeField {
                        fieldLabel: "Y"
                        fieldValue: settingsBridge.leafRangeY
                        commit: function(v) { settingsBridge.leafRangeY = v }
                    }
                    RangeField {
                        fieldLabel: "Z"
                        fieldValue: settingsBridge.leafRangeZ
                        commit: function(v) { settingsBridge.leafRangeZ = v }
                    }
                    RangeField {
                        fieldLabel: "XA"
                        fieldValue: settingsBridge.leafRangeXA
                        commit: function(v) { settingsBridge.leafRangeXA = v }
                    }
                    RangeField {
                        fieldLabel: "YA"
                        fieldValue: settingsBridge.leafRangeYA
                        commit: function(v) { settingsBridge.leafRangeYA = v }
                    }
                    RangeField {
                        fieldLabel: "ZA"
                        fieldValue: settingsBridge.leafRangeZA
                        commit: function(v) { settingsBridge.leafRangeZA = v }
                    }
                }
                IntField {
                    fieldLabel: qsTr("Budget")
                    fieldValue: settingsBridge.leafBudget
                    commit: function(v) { settingsBridge.leafBudget = v }
                }
                IntField {
                    fieldLabel: qsTr("Dilation")
                    fieldValue: settingsBridge.leafDilation
                    commit: function(v) { settingsBridge.leafDilation = v }
                    enabled: settingsBridge.leafHasDilation
                }
                CheckBox {
                    text: qsTr("Enable leaf stage")
                    checked: settingsBridge.enableLeaf
                    onToggled: settingsBridge.enableLeaf = checked
                }

                // ---- Save / Reset --------------------------------------------
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6
                    Button {
                        text: qsTr("Reset")
                        onClicked: settingsBridge.reset()
                    }
                    Button {
                        text: qsTr("Save")
                        Layout.fillWidth: true
                        highlighted: settingsBridge.dirty
                        onClicked: settingsBridge.save()
                    }
                }
            }
        }
    }
}
