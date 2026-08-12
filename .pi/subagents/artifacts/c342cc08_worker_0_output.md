All line refs verified. My analysis is complete — here is Agent 3's report.

**Agent 3: Component Loading & Lifecycle — findings**

**Verified clean (checklist items that don't apply):**
- No `Loader`, no `Image` elements, no `Component.createObject`/`Qt.createComponent`/`createQmlObject` anywhere in the 5 QML files (grep over all files) — the dynamic-creation and image rules are N/A. The fluoro background is VTK-side, not QML.
- Teardown order in `main.cpp` is safe: `QQmlApplicationEngine` is declared last → destroyed first; `ExperimentalScene` first → destroyed last; the QML-created renderer (JavaScriptOwnership) is destroyed with the engine, before the scene it points at.
- Context-property bridges are C++-stack-owned (`AppBridge`, `FileDialogBridge`) and the engine dies before them — no QML-side ownership transfer, no dangling access. `Qt.callLater` callbacks are dropped safely if the engine dies first.
- `Qt.callLater` coalesces multiple schedules per event-loop iteration — the datasetChanged deferral is correct.
- `findChild<QmlVtkRenderer*>` result is null-guarded in main.cpp; empty-rootObjects is handled with `return -1`.

**Findings:**

#### [D-301] Pose table built eagerly at dataset load while the dialog is closed — Loader rule violation
- **File**: `src/app/experimental/main.qml:179,201,359-360`
- **Category**: Component Loading & Lifecycle
- **Confidence**: 90/100
- **Finding**: The `poseDialog`'s `contentItem: ColumnLayout` (201) is created when the `Dialog` is instantiated — at app startup — and the `Repeater { model: poseBridge.tableModel }` (359-360) instantiates 6 `PoseCell` TextFields × every frame **whenever the model populates, regardless of dialog visibility**. A 500-frame study builds 3,000 TextField delegates at load time, not on first open. `tableModel` is a CONSTANT `Q_PROPERTY` (PoseBridge.h:110), so the Repeater tracks row changes directly.
- **Trace**: `contentItem:` with an inline object is constructed at parent creation (standard QML semantics); Repeater builds delegates on model populate; the dialog is closed by default (`Dialog` never `open()`ed at startup). No `Loader` exists anywhere in the app (grep confirms zero).
- **Mitigation**: wrap the pose-dialog content in a `Loader` whose `active` is bound to the dialog's open state (destroys the table when closed, defers construction to first open), plus the plan's virtualization (U5) to cap instantiated rows. Same mechanism affects the settings panel (main.qml:173) but at trivial cost.

#### [D-302] Dialog content persists across open/close — stale-table premise confirmed structurally
- **File**: `src/app/experimental/main.qml:179,201,344,359`
- **Category**: Component Loading & Lifecycle
- **Confidence**: 85/100
- **Finding**: QQC2 `Dialog` never destroys its `contentItem` on close — the pose table's cells and bindings live for the app's lifetime. A table opened after an optimizer run or viewer drag therefore shows whatever the model last notified, not fresh storage state; the only refresh paths are the bridge's `notifyCellChanged`/`refresh()` calls. This structurally confirms the plan's I1/I2 stale-on-reopen premise (the flow analysis asked to verify it empirically — the lifecycle semantics make it a certainty, not a hypothesis).
- **Trace**: `poseDialog` is a Window child; closing hides the popup window, the object tree (ScrollView 344 + Column + Repeater 359) is untouched; `text: storedValue.toFixed(3)` re-reads only when the model notifies.
- **Mitigation**: the plan's U3 refresh-owner fix (PoseBridge hooks runStateChanged + viewerPoseApplied → notify) is the correct repair; a `Loader`-based content (D-301) would also make open-time recreation a natural re-sync.

#### [D-303] Whole-table delegate teardown/rebuild is the app's re-sync mechanism — fragile coupling
- **File**: `src/app/experimental/PoseCell.qml:33,37-49` + `main.qml:359-401`
- **Category**: Component Loading & Lifecycle
- **Confidence**: 85/100
- **Finding**: The app relies on full delegate recreation to re-sync display state: PoseCell.qml:33 documents "full table refreshes recreate this delegate" as the recovery path for the imperative-text binding kill (BND-2, the known C1). Every `refresh()` therefore tears down and rebuilds all cells (destroy → recreate), destroying any editing state and churning 6×N objects. This lifecycle coupling is precisely why the C1 binding kill is dangerous and why virtualization must come with the U3 commit contract (capture-at-edit-start), not after.
- **Trace**: `notifyCellChanged` → model `dataChanged`/`reset` → Repeater rebuilds affected/all delegates; PoseCell's `onEditingFinished` does `text = storedValue.toFixed(3)` (imperative, kills the binding) and relies on recreation to restore.
- **Mitigation**: U3 (commit contract, binding-alive restore) + U5 (ListView reuse with onPooled/onReused) together remove the recreation dependency; no lifecycle work needed beyond that.

**Investigation targets:**

#### [I-301] Silent no-renderer failure mode — viewport glue dead without an error surface
- **File**: `src/app/experimental/main.qml:461` (+ 419-510 glue blocks reference `viewport` declared at ~910)
- **Category**: Component Loading & Lifecycle
- **Confidence**: 70/100
- **Finding**: Six glue blocks bind `target:`/calls to the `viewport` id declared ~450 lines later. The bindings re-evaluate when the id materializes (correct null→object handling by Connections), so normal operation is fine — but if the `QmlVtkRenderer` ever fails to instantiate (GL/xcb failure), every glue block silently no-ops: no warning, no error dialog, and the app runs with a dead viewport behind the placeholder.
- **Unverified because**: the render smoke proves construction works on this box, so the failure branch has no observed trigger; the id-later pattern itself is correct QML.
- **How to verify**: temporarily break the renderer registration and observe whether the app surfaces anything; or check for a `Component.onCompleted` guard on the viewport in the extraction pass (U2's ViewportPanel).

**Residual risks / manual notes:** The `Import`-level Qt5+Qt6 module ambiguity noise in the env (QtQuick.Dialogs) is environmental; `QmlVtkRenderer was not found` in qmllint is the C++-registration/import-path gap (no qmltypes file), not a runtime defect. Teardown ordering was verified safe and needs no plan change.