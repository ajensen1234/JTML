// Plan 007 U6 — SettingsPanel pins: field bindings mirror the fake
// bridge, the ×100 SpinBox scale commits 2-decimal doubles, the dirty
// badge reflects the bridge state, Save/Reset reach the bridge, and the
// dilation field follows the active cost variant's enablement.
import QtQuick
import QtTest
import "qrc:/components"

Item {
    id: root
    width: 560
    height: 2400

    FakeSettingsBridge { id: fakeSettings }

    Component {
        id: panelComp
        SettingsPanel {}
    }

    TestCase {
        id: testCase
        name: "SettingsPanel"
        when: windowShown

        function init() {
            fakeSettings.saveCalls = 0
            fakeSettings.resetCalls = 0
            fakeSettings.dirty = false
            fakeSettings.trunkRangeX = 0
        }

        // Qt 6 QML TestCase has no keyClicks — type char by char.
        // Punctuation chars need keycodes (keyClick(".") produces nothing).
        function typeChars(text) {
            for (let i = 0; i < text.length; ++i) {
                const ch = text.charAt(i)
                const key = ch === '.' ? Qt.Key_Period
                          : ch === '-' ? Qt.Key_Minus
                          : ch
                keyClick(key)
            }
        }
        function test_bindingMirrorsBridge() {
            const panel = createTemporaryObject(panelComp, root, {
                settingsBridge: fakeSettings, width: 400, height: 600 })
            verify(!!panel, "Component exists")
            const spin = findChild(panel, "trunkRangeX")
            verify(!!spin, "Object exists")
            compare(spin.value, 0)
            fakeSettings.trunkRangeX = 1.234
            compare(spin.value, 123)   // Math.round(1.234 * 100)
        }

        function test_scaleCommitsTwoDecimals() {
            const panel = createTemporaryObject(panelComp, root, {
                settingsBridge: fakeSettings, width:400, height: 600 })
            verify(!!panel, "Component exists")
            const spin = findChild(panel, "trunkRangeX")
            verify(!!spin, "Object exists")
            compare(spin.value, 0)
            // Click the up indicator (stepSize 50): value 50, valueModified
            // fires, and the ×100 commit closure writes 50/100 = 0.5 to
            // the bridge — pinning the 2-decimal scale end to end. (The
            // indicator is the top-right quadrant; spin.up itself fails
            // the TestCase instanceof-Item check.)
            mouseClick(spin, spin.width - 5, spin.height / 4)
            tryCompare(fakeSettings, "trunkRangeX", 0.5)
            // A second click lands at 1.00 (still the /100 scale).
            mouseClick(spin, spin.width - 5, spin.height / 4)
            tryCompare(fakeSettings, "trunkRangeX", 1.0)
        }

        function test_dirtyBadge() {
            const panel = createTemporaryObject(panelComp, root, {
                settingsBridge: fakeSettings, width: 400, height: 600 })
            verify(!!panel, "Component exists")
            const label = findChild(panel, "settingsDirtyLabel")
            verify(!!label, "Object exists")
            fakeSettings.dirty = true
            compare(label.text, "● unsaved")
            fakeSettings.dirty = false
            compare(label.text, "saved")
        }

        function test_saveAndResetReachBridge() {
            const panel = createTemporaryObject(panelComp, root, {
                settingsBridge: fakeSettings, width: 400, height: 600 })
            verify(!!panel, "Component exists")
            const save = findChild(panel, "settingsSaveButton")
            const reset = findChild(panel, "settingsResetButton")
            verify(!!save, "Object exists")
            verify(!!reset, "Object exists")
            // The form scrolls: a tall panel keeps the bottom row visible.
            panel.height = 1400
            wait(50)
            // The form scrolls (~1700px of content): a tall panel keeps
            // the bottom action row inside the window.
            panel.height = 2000
            wait(50)
            mouseClick(save)
            tryCompare(fakeSettings, "saveCalls", 1)
            mouseClick(reset)
            tryCompare(fakeSettings, "resetCalls", 1)
        }

        function test_dilationEnablementPerStage() {
            const panel = createTemporaryObject(panelComp, root, {
                settingsBridge: fakeSettings, width: 400, height: 600 })
            verify(!!panel, "Component exists")
            const dilation = findChild(panel, "trunkDilation")
            verify(!!dilation, "Object exists")
            fakeSettings.trunkHasDilation = false
            compare(dilation.enabled, false)
            fakeSettings.trunkHasDilation = true
            compare(dilation.enabled, true)
        }
    }
}
