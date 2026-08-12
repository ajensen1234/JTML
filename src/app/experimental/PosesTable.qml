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
//  - commit-on-pool ordering: focus loss (the delegate is removed for
//    pooling) fires editingFinished BEFORE onPooled — the commit handler
//    reads the live text against the captured tuple first. onPooled
//    therefore has no text to protect (documented below); onReused re-arms
//    each cell's storedValue binding (the text binding is dead after
//    editing — a plain re-bind, never a stale one-shot value);
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
        visible: poseBridge.rowCount === 0
        horizontalAlignment: Text.AlignHCenter
        color: Theme.fgDim
        wrapMode: Text.Wrap
        text: qsTr("No frames loaded — load a study first.")
    }
    Label {
        Layout.fillWidth: true
        Layout.topMargin: Theme.spacingSm
        visible: poseBridge.rowCount > 0
                 && studyBridge.primaryModelIndex < 0
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
        Layout.fillWidth: true
        Layout.fillHeight: true
        clip: true
        spacing: 2
        visible: poseBridge.rowCount > 0
                 && studyBridge.primaryModelIndex >= 0
        enabled: !optimizerBridge.running
        model: poseBridge.tableModel
        reuseItems: true
        ScrollBar.vertical: ScrollBar {}

        delegate: RowLayout {
            required property int frameIndex
            required property double x
            required property double y
            required property double z
            required property double xa
            required property double ya
            required property double za

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
                storedValue: x
            }
            PoseCell {
                id: cellY
                frameRow: frameIndex
                axisIndex: 1
                storedValue: y
            }
            PoseCell {
                id: cellZ
                frameRow: frameIndex
                axisIndex: 2
                storedValue: z
            }
            PoseCell {
                id: cellXA
                frameRow: frameIndex
                axisIndex: 3
                storedValue: xa
            }
            PoseCell {
                id: cellYA
                frameRow: frameIndex
                axisIndex: 4
                storedValue: ya
            }
            PoseCell {
                id: cellZA
                frameRow: frameIndex
                axisIndex: 5
                storedValue: za
            }

            // Commit-on-pool ordering (D8): when this row scrolls out of
            // view mid-edit, the delegate is removed from the scene —
            // focus out fires editingFinished FIRST (the commit reads the
            // live text against the captured tuple), only then is the
            // item pooled and onPooled called. There is nothing to reset
            // here by design: any reset that ran before the commit would
            // corrupt the typed value.
            onPooled: {
                // No text to drop — the commit already ran on focus loss.
            }
            onReused: {
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
