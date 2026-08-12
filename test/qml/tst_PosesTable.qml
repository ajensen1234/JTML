// Plan 007 U6 — PosesTable pins (plan 007 U5 virtualization):
//  - a 500-row model instantiates only the visible rows (O(visible), not
//    O(frames));
//  - commit-on-pool: editing a cell and scrolling away mid-edit commits
//    the typed value to the captured row (no cross-row write, no lost
//    text);
//  - a rejected commit on a recycled cell reverts and stays editable;
//  - the PosesDialog Loader gate: the table does not exist while the
//    dialog is closed.
import QtQuick
import QtTest
import "qrc:/components"

Item {
    id: root
    width: 660
    height: 460

    FakePoseBridge { id: fakePose }
    FakeStudyBridge { id: fakeStudy }
    FakeOptimizerBridge { id: fakeOpt }

    Component {
        id: tableComp
        PosesTable {}
    }

    Component {
        id: dialogComp
        PosesDialog {}
    }

    function fillModel(count) {
        fakePose.tableModel.clear()
        for (let i = 0; i < count; ++i) {
            fakePose.tableModel.append({
                frameIndex: i, x: i, y: i * 2, z: 0, xa: 0, ya: 0, za: 0 })
        }
        fakePose.dirty = false
        fakePose.validationMessage = ""
        fakePose.commitLog = []
    }

    TestCase {
        id: testCase
        name: "PosesTable"
        when: windowShown

        function init() {
            fillModel(500)
            fakeStudy.primaryModelIndex = 0
            fakeOpt.running = false
        }

        function test_virtualizedInstantiation() {
            const table = createTemporaryObject(tableComp, root, {
                poseBridge: fakePose, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 660, height: 460 })
            verify(!!table, "Component exists")
            const list = findChild(table, "poseTableList")
            verify(!!list, "Object exists")
            compare(fakePose.rowCount, 500)
            wait(50)
            const visible = list.contentItem.children.length
            verify(visible < 100)   // ~20 rows for a 460px view, never 500
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
        function test_commitOnPoolMidEdit() {
            const table = createTemporaryObject(tableComp, root, {
                poseBridge: fakePose, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 660, height: 460 })
            verify(!!table, "Component exists")
            const list = findChild(table, "poseTableList")
            verify(!!list, "Object exists")
            wait(50)
            const row3 = list.itemAtIndex(3)
            verify(!!row3, "Object exists")
            const cell = row3.children[1]   // [Frame label, X cell, ...]
            verify(!!cell, "Object exists")
            cell.forceActiveFocus()
            keyClick(Qt.Key_A, Qt.ControlModifier)
            typeChars("12.5")
            // Scroll away gradually: row 3 leaves the viewport -> pooled
            // -> the attached ListView.onPooled flushes the live edit via
            // commitIfEditing (commit-on-pool, D8 — the focus-loss
            // ordering does NOT hold on Qt 6.7, see PosesTable.qml header).
            list.positionViewAtIndex(100, ListView.Center)
            wait(100)
            list.positionViewAtIndex(200, ListView.Center)
            wait(100)
            list.positionViewAtIndex(300, ListView.Center)
            wait(100)
            compare(fakePose.commitLog.length, 1)
            compare(fakePose.commitLog[0][0], 3)      // row 3, not 400
            compare(fakePose.commitLog[0][3], 12.5)
        }

        function test_rejectedCommitSurvivesRecycle() {
            const table = createTemporaryObject(tableComp, root, {
                poseBridge: fakePose, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 660, height: 460 })
            verify(!!table, "Component exists")
            const list = findChild(table, "poseTableList")
            verify(!!list, "Object exists")
            wait(50)
            const row3 = list.itemAtIndex(3)
            verify(!!row3, "Object exists")
            const cell = row3.children[1]
            verify(!!cell, "Object exists")
            cell.forceActiveFocus()
            keyClick(Qt.Key_A, Qt.ControlModifier)
            typeChars("abc")                          // rejected
            list.positionViewAtIndex(100, ListView.Center)
            wait(100)
            list.positionViewAtIndex(200, ListView.Center)
            wait(100)
            list.positionViewAtIndex(300, ListView.Center)
            wait(100)
            compare(fakePose.commitLog.length, 0)
            verify(fakePose.validationMessage.length > 0)
            // Back to row 3: the reused cell shows the stored value and
            // remains editable.
            list.positionViewAtIndex(0, ListView.Beginning)
            wait(100)
            const row3b = list.itemAtIndex(3)
            verify(!!row3b, "Object exists")
            const cellB = row3b.children[1]
            verify(!!cellB, "Object exists")
            tryCompare(cellB, "text", "3.000")        // model.x == 3
            cellB.forceActiveFocus()
            keyClick(Qt.Key_A, Qt.ControlModifier)
            typeChars("9")
            root.forceActiveFocus()
            wait(50)
            compare(fakePose.commitLog.length, 1)
            compare(fakePose.commitLog[0][3], 9)
        }

        function test_instantJumpFlushesEdit() {
            // R2 (review round): an INSTANT jump re-binds the delegate in
            // place (or destroys it) without pooling — the live edit must
            // still commit to the row it was typed on. The flush runs via
            // the delegate's onFrameIndexChanged (in-place re-bind) or
            // the cell's Component.onDestruction (destroyed) — either
            // way the commit lands exactly once.
            const table = createTemporaryObject(tableComp, root, {
                poseBridge: fakePose, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 660, height: 460 })
            verify(!!table, "Component exists")
            const list = findChild(table, "poseTableList")
            verify(!!list, "Object exists")
            wait(50)
            const row3 = list.itemAtIndex(3)
            verify(!!row3, "Object exists")
            const cell = row3.children[1]
            verify(!!cell, "Object exists")
            cell.forceActiveFocus()
            keyClick(Qt.Key_A, Qt.ControlModifier)
            typeChars("12.5")
            list.positionViewAtIndex(400, ListView.Center)   // instant
            wait(100)
            compare(fakePose.commitLog.length, 1)
            compare(fakePose.commitLog[0][0], 3)      // row 3, not 400
            compare(fakePose.commitLog[0][1], 0)
            compare(fakePose.commitLog[0][2], 0)
            compare(fakePose.commitLog[0][3], 12.5)
        }

        function test_keyboardLeftRightMovesCell() {
            // D7 (review round): the table is keyboard navigable —
            // Left/Right moves between the six cells of a row.
            const table = createTemporaryObject(tableComp, root, {
                poseBridge: fakePose, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 660, height: 460 })
            verify(!!table, "Component exists")
            const list = findChild(table, "poseTableList")
            verify(!!list, "Object exists")
            wait(50)
            const row0 = list.itemAtIndex(0)
            verify(!!row0, "Object exists")
            const cellX = row0.children[1]
            const cellY = row0.children[2]
            verify(!!cellX && !!cellY, "Cells exist")
            cellX.forceActiveFocus()
            keyClick(Qt.Key_Right)
            wait(50)
            verify(cellY.activeFocus, "Right moved focus to the Y cell")
            keyClick(Qt.Key_Left)
            wait(50)
            verify(cellX.activeFocus, "Left moved focus back to the X cell")
        }

        function test_keyboardUpDownMovesRow() {
            // D7 (review round): Up/Down moves to the same axis in the
            // adjacent row (the moveToRow position + deferred focus).
            const table = createTemporaryObject(tableComp, root, {
                poseBridge: fakePose, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 660, height: 460 })
            verify(!!table, "Component exists")
            const list = findChild(table, "poseTableList")
            verify(!!list, "Object exists")
            wait(50)
            const row1 = list.itemAtIndex(1)
            verify(!!row1, "Object exists")
            const cell1 = row1.children[1]
            cell1.forceActiveFocus()
            keyClick(Qt.Key_Up)
            wait(150)   // positionViewAtIndex + Qt.callLater focus
            const row0 = list.itemAtIndex(0)
            verify(!!row0, "Object exists")
            verify(row0.children[1].activeFocus,
                   "Up moved focus to row 0's X cell")
            // And back down.
            keyClick(Qt.Key_Down)
            wait(150)
            verify(row1.children[1].activeFocus,
                   "Down moved focus back to row 1's X cell")
        }

        function test_loaderGate() {
            const dialog = createTemporaryObject(dialogComp, root, {
                poseBridge: fakePose, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt })
            verify(!!dialog, "Component exists")
            // Closed: no table exists (deferred construction).
            verify(!findChild(dialog, "poseTableList"))
            dialog.open()
            wait(300)
            verify(!!findChild(dialog, "poseTableList"))
            dialog.close()
            wait(400)
            verify(!findChild(dialog, "poseTableList"))
        }
    }
}
