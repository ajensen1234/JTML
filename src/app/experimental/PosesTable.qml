import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import "."  // Theme + PoseCell

// 007 U2: the editable pose table (extracted from main.qml, U8 surface):
// the column header + empty states + the table body.
//
// 007 U5: the Repeater-in-Column-in-ScrollView is replaced by a virtualized
// ListView (reuseItems) — O(visible rows) delegate instances instead of
// O(frames) (6 PoseCell TextFields × every frame were built eagerly, even
// while the dialog was closed; the PosesDialog now Loader-gates this
// component, see D-04/D-05).
//
// Data-integrity contract (D2/D8, owner-confirmed):
//  - capture-at-edit-start (U3) makes recycling safe — a pooled cell
//    commits to the (frame, model, axis) tuple captured when editing
//    started, never the row it now displays;
//  - commit-on-pool (corrected in U6 — the focus-loss ordering does NOT
//    hold on Qt 6.7: pooling retains focus, so editingFinished never
//    fires on recycle): a live edit is flushed explicitly via
//    commitIfEditing() from the attached ListView.onPooled handler;
//    onReused re-arms each cell's storedValue binding (the text binding
//    is dead after editing) and re-captures the tuple when focus was
//    retained (PoseCell.resetDisplay);
//  - I-02 note: PoseTableModel::refresh() full-resets on copy/load paths —
//    acceptable for now (pooled delegates flush with the reset); U7 may
//    move single-row operations to role-filtered notify.
//
// Width authority (I-06): the header and every row delegate span the full
// available width with fixed column preferredWidths (44 + 6x78) — the
// trailing gutter (>= overlay-scrollbar width) stays empty, so the
// scrollbar covers gutter, not the ZA column. I-09 (availableWidth jitter)
// is moot: no Column-sized-by-availableWidth remains.

ColumnLayout {
    id: root
    Layout.fillWidth: true
    Layout.fillHeight: true
    spacing: Theme.spacingXs

    // 007 U6 (D1): injected bridge surface — the PosesDialog passes its
    // own injected props through the Loader instantiation site.
    required property var poseBridge
    required property var studyBridge
    required property var optimizerBridge

    // Testability (plan 007 U6): the virtualized list is reachable from
    // the Qt Quick Test via findChild (row instantiation + recycling pins).
    readonly property string tableObjectName: "poseTableList"

    // R2 (review round): the discard-close path sets this BEFORE closing
    // so in-flight cell edits are dropped instead of committed during
    // teardown (Component.onDestruction on the cells honors it).
    property bool suppressDestructionCommit: false

    // ---- Column header (fixed widths match the cell fields) -------------
    RowLayout {
        Layout.fillWidth: true
        spacing: Theme.spacingXs
        Label {
            text: qsTr("Frame")
            Layout.preferredWidth: 44
            horizontalAlignment: Text.AlignRight
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("X")
            Layout.preferredWidth: 78
            horizontalAlignment: Text.AlignHCenter
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("Y")
            Layout.preferredWidth: 78
            horizontalAlignment: Text.AlignHCenter
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("Z")
            Layout.preferredWidth: 78
            horizontalAlignment: Text.AlignHCenter
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("XA")
            Layout.preferredWidth: 78
            horizontalAlignment: Text.AlignHCenter
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("YA")
            Layout.preferredWidth: 78
            horizontalAlignment: Text.AlignHCenter
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
        Label {
            text: qsTr("ZA")
            Layout.preferredWidth: 78
            horizontalAlignment: Text.AlignHCenter
            color: Theme.fgMuted
            font.pixelSize: Theme.caption
        }
    }

    // ---- Empty states -----------------------------------------------------
    Label {
        Layout.fillWidth: true
        Layout.topMargin: Theme.spacingSm
        visible: root.poseBridge.rowCount === 0
        horizontalAlignment: Text.AlignHCenter
        color: Theme.fgDim
        wrapMode: Text.Wrap
        text: qsTr("No frames loaded — load a study first.")
    }
    Label {
        Layout.fillWidth: true
        Layout.topMargin: Theme.spacingSm
        visible: root.poseBridge.rowCount > 0
                 && root.studyBridge.primaryModelIndex < 0
        horizontalAlignment: Text.AlignHCenter
        color: Theme.fgDim
        wrapMode: Text.Wrap
        text: qsTr("Select a model in the model list to edit its "
                   + "poses (v1: pose ops edit the primary model).")
    }

    // ---- The editable table (virtualized, U5) -----------------------------
    // Hidden only without a dataset + primary model; during an optimizer
    // run it stays visible but disabled (the U6 locking — the Run button
    // also closes this dialog). The dialog Loader-gates this whole
    // component, so the ListView exists only while the dialog is open.
    ListView {
        id: tableList
        objectName: root.tableObjectName
        Layout.fillWidth: true
        Layout.fillHeight: true
        clip: true
        spacing: 2
        visible: root.poseBridge.rowCount > 0
                 && root.studyBridge.primaryModelIndex >= 0
        enabled: !root.optimizerBridge.running
        model: root.poseBridge.tableModel
        reuseItems: true
        ScrollBar.vertical: ScrollBar {}

        delegate: RowLayout {
            // 007 U6 fix: the pose roles are accessed as model.roleName —
            // required-property declarations for x/y/z collide with Item's
            // FINAL geometry properties (Qt 6: "Cannot override FINAL
            // property" at load; the app has been unloadable since U5).
            // frameIndex does NOT collide, so it is declared required to
            // enable onFrameIndexChanged (the R2 in-place re-bind flush).
            // `model` is declared required too: with required properties
            // present, the IMPLICIT model context object goes stale on
            // pool reuse (test-proven: frameIndex re-bound to 3 while
            // model.x stayed 0) — the required model re-binds per row.
            required property int frameIndex
            required property var model

            // ListView pooling hooks (007 U6 fix): the pool manager emits
            // these by signal name — the delegate must DECLARE them or the
            // onPooled/onReused handlers fail at load ("Cannot assign to
            // non-existent property").

            // The delegate root is a ListView child, not a layout child —
            // width here is the view width (stretch; D-02/I-08: the old
            // Layout.fillWidth inside a plain Column was a no-op).
            width: ListView.view.width
            spacing: Theme.spacingXs

            Label {
                text: qsTr("F%1").arg(frameIndex)
                Layout.preferredWidth: 44
                horizontalAlignment: Text.AlignRight
                color: Theme.fgMuted
                font.pixelSize: Theme.caption
            }
            PoseCell {
                id: cellX
                frameRow: frameIndex
                axisIndex: 0
                storedValue: model.x
                // 007 U6 (D1): pass the injected bridges down to the cells.
                poseBridge: root.poseBridge
                studyBridge: root.studyBridge
                // R2: the discard-close path suppresses destruction commits.
                suppressDestructionCommit: root.suppressDestructionCommit
                // D7: cell navigation (left/right within the row).
                onNavRequested: (dir) => {
                    if (dir === -1 || dir === 1) {
                        focusCell(0 + dir)
                    } else {
                        moveToRow(frameIndex + (dir === -2 ? -1 : 1), 0)
                    }
                }
            }
            PoseCell {
                id: cellY
                frameRow: frameIndex
                axisIndex: 1
                storedValue: model.y
                poseBridge: root.poseBridge
                studyBridge: root.studyBridge
                suppressDestructionCommit: root.suppressDestructionCommit
                onNavRequested: (dir) => {
                    if (dir === -1 || dir === 1) {
                        focusCell(1 + dir)
                    } else {
                        moveToRow(frameIndex + (dir === -2 ? -1 : 1), 1)
                    }
                }
            }
            PoseCell {
                id: cellZ
                frameRow: frameIndex
                axisIndex: 2
                storedValue: model.z
                poseBridge: root.poseBridge
                studyBridge: root.studyBridge
                suppressDestructionCommit: root.suppressDestructionCommit
                onNavRequested: (dir) => {
                    if (dir === -1 || dir === 1) {
                        focusCell(2 + dir)
                    } else {
                        moveToRow(frameIndex + (dir === -2 ? -1 : 1), 2)
                    }
                }
            }
            PoseCell {
                id: cellXA
                frameRow: frameIndex
                axisIndex: 3
                storedValue: model.xa
                poseBridge: root.poseBridge
                studyBridge: root.studyBridge
                suppressDestructionCommit: root.suppressDestructionCommit
                onNavRequested: (dir) => {
                    if (dir === -1 || dir === 1) {
                        focusCell(3 + dir)
                    } else {
                        moveToRow(frameIndex + (dir === -2 ? -1 : 1), 3)
                    }
                }
            }
            PoseCell {
                id: cellYA
                frameRow: model.frameIndex
                axisIndex: 4
                storedValue: model.ya
                poseBridge: root.poseBridge
                studyBridge: root.studyBridge
                suppressDestructionCommit: root.suppressDestructionCommit
                onNavRequested: (dir) => {
                    if (dir === -1 || dir === 1) {
                        focusCell(4 + dir)
                    } else {
                        moveToRow(frameIndex + (dir === -2 ? -1 : 1), 4)
                    }
                }
            }
            PoseCell {
                id: cellZA
                frameRow: frameIndex
                axisIndex: 5
                storedValue: model.za
                poseBridge: root.poseBridge
                studyBridge: root.studyBridge
                suppressDestructionCommit: root.suppressDestructionCommit
                onNavRequested: (dir) => {
                    if (dir === -1 || dir === 1) {
                        focusCell(5 + dir)
                    } else {
                        moveToRow(frameIndex + (dir === -2 ? -1 : 1), 5)
                    }
                }
            }

            // D7: focus the sibling cell in this row (Left/Right).
            function focusCell(axis) {
                if (axis < 0 || axis > 5) return
                const cells = [cellX, cellY, cellZ, cellXA, cellYA, cellZA]
                cells[axis].forceActiveFocus()
            }

            // D7: move to the same axis in an adjacent row (Up/Down).
            // The target row may not be instantiated (virtualized) —
            // position first, then focus the cell once it exists.
            function moveToRow(frame, axis) {
                if (frame < 0 || frame >= root.poseBridge.rowCount) return
                tableList.positionViewAtIndex(frame, ListView.Contain)
                Qt.callLater(() => {
                    const row = tableList.itemAtIndex(frame)
                    if (row && row.focusCell) row.focusCell(axis)
                })
            }

            // R2: an in-place re-bind (instant jump / fast scroll — the
            // same delegate instance re-binds without pooling): flush any
            // live edit to the OLD row (captured tuple + live text). The
            // display re-arm stays with the pool path (ListView.onReused)
            // — a second resetDisplay here double-armed the bindings and
            // left a stale value (test-proven). In-place stale-text after
            // a flush is the documented residual edge (the pool path is
            // the normal one).
            onFrameIndexChanged: {
                cellX.commitIfEditing()
                cellY.commitIfEditing()
                cellZ.commitIfEditing()
                cellXA.commitIfEditing()
                cellYA.commitIfEditing()
                cellZA.commitIfEditing()
            }

            // Commit-on-pool, made real (007 U6): Qt 6.7 pooling does not
            // drop focus (hidden/reparented items keep activeFocus), so
            // focus-loss editingFinished cannot be relied on — the typed
            // value would be silently clobbered by resetDisplay without a
            // commit. The pool notifications are ListView ATTACHED signals
            // (delegate-root `signal pooled()` + onPooled: never fire).
            // Flush any live edit explicitly BEFORE the cell is re-bound
            // to a new row — the commit reads the live text against the
            // captured tuple; resetDisplay then re-arms the bindings.
            ListView.onPooled: {
                cellX.commitIfEditing()
                cellY.commitIfEditing()
                cellZ.commitIfEditing()
                cellXA.commitIfEditing()
                cellYA.commitIfEditing()
                cellZA.commitIfEditing()
            }
            ListView.onReused: {
                // The storedValue text bindings are dead after editing —
                // re-arm all six for the NEW row (never stale pooled text).
                cellX.resetDisplay()
                cellY.resetDisplay()
                cellZ.resetDisplay()
                cellXA.resetDisplay()
                cellYA.resetDisplay()
                cellZA.resetDisplay()
            }
        }
    }

    // Plan 007 U4: the Poses dialog focuses the first editable cell on
    // open. Row delegates are [Frame label, 6 PoseCells] — the first cell
    // is child 1 of the first row. Virtualized (U5): itemAtIndex(0)
    // returns the delegate for row 0 (visible at the top; recycling never
    // evicts it while it is on screen).
    function focusFirstCell() {
        const row = tableList.itemAtIndex(0)
        if (row && row.children.length > 1) {
            const cell = row.children[1]
            if (cell) cell.forceActiveFocus()
        }
    }
}
