All facts traced. Compiling the Agent 4 report:

## Agent 4: ListView & Delegate Correctness — findings

**Trace basis:** pose-table delegate (main.qml:363–399) binds `poseBridge.tableModel` (a `QAbstractListModel` `PoseTableModel`, roles `frameIndex/x/y/z/xa/ya/za` — PoseBridge.h:67–94); frame/model list delegates (main.qml:652–711) bind `studyBridge.frameListModel`/`modelListModel` (`display` role) and `selectedModels` (`QVariantList`, StudyBridge.h:89). Zero focus/keyboard constructs anywhere in the app's QML (grep confirmed).

### Findings (confidence > 80)

**F1 (90) — Pose-table O(n) instantiation with a 7-items-per-row multiplier.** `Repeater` in `Column` in `ScrollView` materializes every frame row at dialog open: each row = 1 Label + 6 PoseCell TextFields. 500 frames → 3,000+ TextFields + 500 RowLayouts live simultaneously; scrolling pans only, never instantiates. The plan's U5 (ListView) is the right fix; its justification should carry the per-row multiplier and dialog-open latency, not just "O(n) rows".

**F2 (88) — Pooled-reuse display corruption mechanism.** PoseCell's `text: storedValue.toFixed(3)` binding is killed by the first keystroke (QQC2 text editing assigns `text`; the file's own comment confirms). Under `reuseItems: true`, the `required property double storedValue` re-binds to the new row on reuse, but `text` stays at the stale typed string — the binding is dead, so no declarative re-sync occurs. Mitigation that must land with U5: `onReused` must imperatively re-sync `text` (assign or re-apply `Qt.binding`), and `onPooled` must not clobber a mid-edit commit (ordering pin). PoseCell has no per-edit JS state (only required props), so pooling is otherwise safe.

**F3 (85) — Commit-time tuple reads defeat recycling correctness.** The delegate feeds `frameRow: model.frameIndex` fresh from the model; PoseCell commits `setPoseValue(frameRow, studyBridge.primaryModelIndex, ...)` reading both `frameRow` (re-binds on reuse) and `primaryModelIndex` at commit time. A recycle or selection change between typing and `editingFinished` writes the new row/primary with the old text — the C1/C2 class, confirmed at delegate level. D2's capture-at-edit-start is the hard precondition for U5.

**F4 (90) — Lists and the pose table are entirely mouse-only.** Zero `activeFocus`/`KeyNavigation`/`Keys`/`focus`/`TapHandler` in the app. Delegate Rectangles + MouseArea: Tab skips both lists and the 500-row table; ListView's built-in Up/Down currentIndex movement would diverge from the bridge because `setCurrentFrame` only fires in `onClicked`. Delegate-level requirement (feeds plan D7): rows need `activeFocusOnTab` (or the list-level key contract), a visible focus state, and the bridge sync must fire on any currentIndex change, not just clicks.

### Investigation targets (60–79)

**I1 (75) — Full model reset vs pooled delegates.** `PoseTableModel::refresh()` does a full begin/endResetModel on copy/load/selection changes (header: "small tables, the write-once list models' full-reset style"); per-cell commits use `notifyCellChanged` (delegate-preserving). After virtualization, a full reset flushes the pool and rebuilds visible rows — acceptable, but the "small tables" assumption and reset-vs-notify call sites should be re-checked in U5.

**I2 (70) — Delegate id references block component extraction.** List delegates use `width: frameList.width`, `frameList.currentIndex === index`, `modelList.width` — root ids reachable only while delegates stay in the same file (default Unbound ComponentBehavior). If U2 moves delegates into separate files (or adds `pragma ComponentBehavior: Bound`), these break; convert to `ListView.view` + required properties at that point. Keeping delegates inline in StudyPanel.qml is safe.

**I3 (65) — `Layout.fillWidth: true` on the pose-row RowLayout is a no-op** (parent is a plain Column, not a layout); row width comes from Column cross-axis sizing. Harmless today; in U5 the row delegate must size via `ListView.view.width`/cellWidth instead.

**I4 (60) — `selectedModels.indexOf(index)` is O(n) per delegate evaluation** on a QVariantList→JS-array conversion; fine at current model counts, cosmetic.

**Residual risks:** line refs will drift with the in-flight plan-006 edits; F2/F3/I1 rely on Qt-standard ListView/QQC2 semantics + the header contract, not runtime observation (no headless view-layer run exists yet — the U6 harness is the first).