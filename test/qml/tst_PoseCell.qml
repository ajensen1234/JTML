// Plan 007 U6 — PoseCell data-integrity pins (plan 007 U3 D2, C1/C2):
//  - the commit lands on the (frame, model, axis) tuple CAPTURED at
//    edit start, never the values at commit time (a mid-edit selection
//    change or delegate recycle cannot misroute the write);
//  - a rejected commit reverts the display AND keeps the storedValue
//    binding alive (editing again works);
//  - a storedValue change re-renders the text while the binding is alive.
import QtQuick
import QtTest
import "qrc:/components"

Item {
    id: root
    width: 400
    height: 400

    FakePoseBridge { id: fakePose }
    FakeStudyBridge { id: fakeStudy }

    // Populate the fake's table model (the real bridge bounds-checks
    // frame/model against storage; the fake mirrors it since the review).
    function seedTable(n) {
        fakePose.tableModel.clear()
        for (let i = 0; i < n; ++i) {
            fakePose.tableModel.append({ x: 1, y: 2, z: 3, xa: 0, ya: 0, za: 0 })
        }
    }

    Component {
        id: cellComp
        PoseCell {}
    }

    TestCase {
        id: testCase
        name: "PoseCell"
        when: windowShown

        function init() {
            fakePose.commitLog = []
            fakePose.validationMessage = ""
            fakePose.dirty = false
            fakeStudy.primaryModelIndex = 0
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
        // Type into the focused cell, then move focus out to fire
        // editingFinished (QQC2: focus loss with modified contents).
        function typeAndCommit(cell, text) {
            cell.forceActiveFocus()
            keyClick(Qt.Key_A, Qt.ControlModifier)
            typeChars(text)
            root.forceActiveFocus()
        }

        function test_commitUsesCapturedTuple() {
            seedTable(100)
            const cell = createTemporaryObject(cellComp, root, {
                frameRow: 3, axisIndex: 1, storedValue: 5.0,
                poseBridge: fakePose, studyBridge: fakeStudy })
            verify(!!cell, "Component exists")
            // Review fix (2026-08-12): the mid-edit mutation must happen
            // WHILE the cell still has focus — a live-read implementation
            // would commit the mutated values and fail this pin. (The old
            // version mutated after the commit had already fired, so the
            // test could not fail.)
            cell.forceActiveFocus()
            keyClick(Qt.Key_A, Qt.ControlModifier)
            typeChars("12.5")
            fakeStudy.primaryModelIndex = 1
            cell.frameRow = 99
            wait(50)
            root.forceActiveFocus()  // focus out -> editingFinished
            wait(50)
            compare(fakePose.commitLog.length, 1)
            compare(fakePose.commitLog[0][0], 3)   // captured frame
            compare(fakePose.commitLog[0][1], 0)   // captured model
            compare(fakePose.commitLog[0][2], 1)   // captured axis
            compare(fakePose.commitLog[0][3], 12.5)
        }

        function test_recycleRetainedFocusRecapturesTuple() {
            // Review fix (2026-08-12, C1): a pool round-trip can retain
            // focus (Qt 6.7). resetDisplay must re-capture the tuple for
            // the NEW row — a stale (-1,-1,-1) tuple would silently
            // reject (and lose) the next typed value.
            seedTable(100)
            const cell = createTemporaryObject(cellComp, root, {
                frameRow: 3, axisIndex: 1, storedValue: 5.0,
                poseBridge: fakePose, studyBridge: fakeStudy })
            verify(!!cell, "Component exists")
            cell.forceActiveFocus()
            keyClick(Qt.Key_A, Qt.ControlModifier)
            typeChars("12.5")
            // Simulate the pool round-trip: the delegate re-binds to a
            // new row (frameRow re-bound) while focus is retained, then
            // resetDisplay runs (the onReused path).
            cell.frameRow = 7
            cell.storedValue = 2.0
            cell.resetDisplay()
            verify(cell.activeFocus, "Focus retained through recycle")
            // Type again without clicking: the re-captured tuple must be
            // the NEW row, and the commit must land.
            keyClick(Qt.Key_A, Qt.ControlModifier)
            typeChars("9.5")
            root.forceActiveFocus()
            wait(50)
            compare(fakePose.commitLog.length, 1)
            compare(fakePose.commitLog[0][0], 7)   // re-captured frame
            compare(fakePose.commitLog[0][2], 1)   // axis
            compare(fakePose.commitLog[0][3], 9.5)
        }

        function test_failedCommitKeepsBindingAlive() {
            seedTable(100)
            const cell = createTemporaryObject(cellComp, root, {
                frameRow: 0, axisIndex: 0, storedValue: 5.0,
                poseBridge: fakePose, studyBridge: fakeStudy })
            verify(!!cell, "Component exists")
            typeAndCommit(cell, "abc")            // non-numeric -> reject
            wait(50)
            compare(fakePose.commitLog.length, 0)
            verify(fakePose.validationMessage.length > 0)
            compare(cell.text, "5.000")           // display reverted
            // The binding is alive again: editing works on the second try.
            typeAndCommit(cell, "7")
            wait(50)
            compare(fakePose.commitLog.length, 1)
            compare(fakePose.commitLog[0][3], 7)
        }

        function test_storedValueChangeRerenders() {
            const cell = createTemporaryObject(cellComp, root, {
                frameRow: 0, axisIndex: 0, storedValue: 1.5,
                poseBridge: fakePose, studyBridge: fakeStudy })
            verify(!!cell, "Component exists")
            compare(cell.text, "1.500")
            cell.storedValue = 9.25
            tryCompare(cell, "text", "9.250")
        }

        function test_escapeRevertsCell() {
            // D7 (review round): Escape reverts the cell to the stored
            // value WITHOUT committing, and the storedValue binding is
            // alive again (typing a second time works).
            seedTable(100)
            const cell = createTemporaryObject(cellComp, root, {
                frameRow: 4, axisIndex: 2, storedValue: 5.0,
                poseBridge: fakePose, studyBridge: fakeStudy })
            verify(!!cell, "Component exists")
            cell.forceActiveFocus()
            keyClick(Qt.Key_A, Qt.ControlModifier)
            typeChars("12.5")
            keyClick(Qt.Key_Escape)
            wait(50)
            tryCompare(cell, "text", "5.000")     // reverted to stored
            compare(fakePose.commitLog.length, 0)  // nothing committed
            // The binding is alive: select-all + type, then commit.
            keyClick(Qt.Key_A, Qt.ControlModifier)
            typeChars("7")
            root.forceActiveFocus()
            wait(50)
            compare(fakePose.commitLog.length, 1)
            compare(fakePose.commitLog[0][0], 4)
            compare(fakePose.commitLog[0][2], 2)
            compare(fakePose.commitLog[0][3], 7)
        }

        function test_destructionCommitsLiveEdit() {
            // R2 (review round): destroying a cell mid-edit commits the
            // live edit (a model reset / dialog teardown must not lose
            // the typed value silently).
            seedTable(100)
            const cell = createTemporaryObject(cellComp, root, {
                frameRow: 6, axisIndex: 1, storedValue: 1.0,
                poseBridge: fakePose, studyBridge: fakeStudy })
            verify(!!cell, "Component exists")
            cell.forceActiveFocus()
            keyClick(Qt.Key_A, Qt.ControlModifier)
            typeChars("8.75")
            cell.destroy()
            wait(100)
            compare(fakePose.commitLog.length, 1)
            compare(fakePose.commitLog[0][0], 6)
            compare(fakePose.commitLog[0][1], 0)
            compare(fakePose.commitLog[0][2], 1)
            compare(fakePose.commitLog[0][3], 8.75)
        }

        function test_destructionSuppressedDoesNotCommit() {
            // R2 (review round): the discard-close path sets
            // suppressDestructionCommit before teardown — the in-flight
            // edit is dropped, not committed.
            seedTable(100)
            const cell = createTemporaryObject(cellComp, root, {
                frameRow: 6, axisIndex: 1, storedValue: 1.0,
                poseBridge: fakePose, studyBridge: fakeStudy })
            verify(!!cell, "Component exists")
            cell.suppressDestructionCommit = true
            cell.forceActiveFocus()
            keyClick(Qt.Key_A, Qt.ControlModifier)
            typeChars("8.75")
            cell.destroy()
            wait(100)
            compare(fakePose.commitLog.length, 0)
        }
    }
}
