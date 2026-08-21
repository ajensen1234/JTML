# JTML View Layer + App Layer Scout Findings

## Per-file one-liners

### include/view + src/view (QWidgets view, STATIC lib `jtml_view`)
- `mainscreen.h` (547) / `mainscreen.cpp` (5045) — The **view + composition root**. QMainWindow; owns two `Viewer`, all VTK actors/mappers, `loaded_frames/_B`/`loaded_models`, `LocationStorage`, plus the extracted seams it delegates to. Emits only `UpdateDisplayText(bool)` (dead — no `emit` in cpp) plus plain methods `VTKEscapeSignal()`/`VTKMakePrincipalSignal()`. ~60 `on_*` slots.
- `viewer.h` (176) / `viewer.cpp` (525) — Pure VTK render surface (renderer, camera, actors, image importers, model actors/mappers). Holds `Frame`/`Model`/`Calibration` + `render_pipeline_builder`. No signals.
- `drr_tool.h` (84) / `drr_tool.cpp` (243) — QDialog, DRR preview. Renders VTK DRR from a `Model` + CUDA `gpu_cost_function::GPUModel`, `CameraCalibration`.
- `controls.h` (36) / `controls.cpp` (32) — QDialog, keyboard/mouse controls. Holds `QGraphicsScene` pixmap; no backend.
- `settings_control.h` (96) / `settings_control.cpp` (1260) — QDialog optimizer-settings editor. Holds 3 `jta_cost_function::CostFunctionManager` + `OptimizerSettings`. **Only view widget with real signal emission**: emits `SaveSettings(...)` and `Done()`.
- `about.h` (26) / `about.cpp` (156) — QDialog version-label popup. No backend.
- `frame_list_model.h` (35) / `frame_list_model.cpp` (26) — `QAbstractListModel` view-model (frame display names). QtCore-only, widget-free, headless-testable.
- `model_list_model.h` (32) / `model_list_model.cpp` (56) — `QAbstractListModel` view-model; reuses `jta::ModelListBuilder::UniquifyModelNames`.

### src/app — composition roots
- `main.cpp` (21) — Thin widgets-app root: sets QVTK GL surface, `MainScreen w; w.show()`. No other wiring; everything lives inside MainScreen.
- `experimental/` — **Parallel QML Qt-Quick front-end executable** (`jtml_experimental`, plans 005/006/007). QML: `main.qml` (347, composition-root Window + FileDialogs + panels), `StudyPanel.qml`, `ViewportPanel.qml`, `RunBar.qml`, `MlStrip.qml`, `SettingsPanel.qml`, `PosesDialog.qml`, `PosesTable.qml`, `PoseCell.qml`, `Toolbar.qml`, `Theme.qml`, `renderer.qml`.
  - C++ bridge seams (thin pass-through `Q_OBJECT`, `Q_PROPERTY`/`Q_INVOKABLE`): `AppBridge.h` (hub owns `ExperimentalSession` + adapters), `StudyBridge.h` (load/selection/scene), `SettingsBridge.h`, `OptimizerBridge.h` (delegates to coordinator `OptimizerRunController`), `MlBridge.h`, `PoseBridge.h`, `FileDialogBridge.h`, `DelegateSelection.*` (multi-select). `ExperimentalSession.*` = app-owned dataset. `ExperimentalScene.*` + `QmlVtkRenderer.*` = QML viewport render seam (`QQuickVTKItem`).
- `Study2Grid/` — self-contained legacy tool (own `main.cpp`, `Study`, parsers), unrelated to view.

## Directionality answers

### 1. Does the view import/link domain/services/coordinator/compute? Which edges?
**Yes — the view is the top of the stack and links everything below.** Concrete edges:
- `mainscreen.h` links all four layers: `compute/curvature_utilities.h`, `domain/data_structures_6D.h`, `services/calibration.h|location_storage.h|model.h|optimizer_settings.h|settings_service.h|session_controller.h|study_load_controller.h|ml_orchestrator.h|segmentation_controller.h`, `coordinator/optimizer_run_controller.h|session_state_controller.h`, `compute/CostFunctionManager.h|camera_calibration.h|machine_learning_tools.h|frame.h`, `domain/session_state.h`.
- `mainscreen.cpp` adds: `compute/curvature_utilities.h`, `domain/settings_constants.h|pose_file_io.h|optimize_intent_controller.h|model_list_builder.h|pose_copy.h|ambiguous_pose_processing.h`, `services/STLReader.h|cost_function_registry.h|edge_processor.h|save_last_pose.h|segmentation_controller.h`.
- `viewer.cpp`: `services/render_pipeline_builder.h`. `model_list_model.cpp`: `domain/model_list_builder.h`. `settings_control.cpp`: `domain/settings_constants.h`. `drr_tool.h`: `compute/gpu_model.cuh`.
- `src/view/CMakeLists.txt` links PUBLIC `jtml_coordinator` (which PUBLIC-pulls services→domain→compute) + JTA/VTK/Torch/OpenCV/Eigen → the view dominates.

### 2. Is the view QML-swappable, or QWidgets-only today?
**QWidgets-only in production; QML is a live parallel experimental target, NOT a swap of the same view.** Two separate executables (`${PROJECT_NAME}` widgets vs `jtml_experimental`); no runtime switch. Not a shared interface — the seam is "experimental does NOT link `jtml_view` at all" (R1). Reuse is only `FrameListModel`/`ModelListModel` **direct-compiled** from include/view+src/view; everything else reaches the stack bottom-up via jtml_coordinator. The `test/qml/Fake*.qml` bridges (FakeApp/Study/Optimizer/Ml/Pose/Settings) replace the C++ bridges so `test/qml/main.cpp` is headless + VTK-free; `tests.qrc` aliases the REAL experimental QML sources (no drift). src/view/CMakeLists.txt lines 10-15 QML DECISION RECORD: full QML front-end **DEFERRED**; experimental is the sanctioned parallel vehicle; revisit on a concrete widget-unreachable need or Qt 6.8+/VTK 9.4+ maturity.

### 3. What is the source of truth the view binds to?
- **Widgets view:** services/domain are the fact store but the widget wires manually — `MainScreen` owns `loaded_frames`/`loaded_models`/`LocationStorage` and pulls into `jta::SessionState session_state_` via `SyncSessionState()`. List view-models hold only write-once display names; selection lives in `QItemSelectionModel`. `SettingsService` owns QSettings; `OptimizerRunController` owns run state; `SessionStateController` diffs+emits over `session_state`.
- **QML view-modified:** bridge adapters are the source of truth — QML binds Q_PROPERTY (`optimizerBridge.runState`/`progress`, `settingsBridge.dirty`, `poseBridge.dirty`, `studyBridge.frameListModel`) + Q_INVOKABLE mutators hitting shared services (SessionController, OptimizerRunController, SettingsService, PoseFileIO). AppBridge owns `ExperimentalSession` (the dataset).

### 4. Where does the MVVM-with-QWidgets-view-vs-QML-view tension sit — MVVM or layered MVC?
**Layered MVC with a light unbound "session_state"-intent controller, NOT MVVM.** `mainscreen.h` lines 141-148 / 251-252 explicitly: "View + composition root… no binding framework", `session_state_` is "NOT an observable ViewModel". Widget-free services/domain + re-emission seams (`OptimizerRunController`/`SessionStateController` re-emit on own thread so `QSignalSpy` observes on main thread — QTBUG-2842). The closest thing to a VM is the diffless `SessionStateController` over the singleton `session_state_` (a controller, not a bound VM). Actual Q_PROPERTY+bindings MVVM exists only in `src/app/experimental`, excluded from `jtml_view`. Tension also in two divergent selection idioms: QML `DelegateSelection` row-set vs widget `QItemSelectionModel`.

## Architecture
```
WIDGETS app                          QML app (jtml_experimental)
 src/app/main.cpp ──→ MainScreen ─┐   src/app/experimental/main.cpp ─┐
 jtml_view (STATIC) = MainScreen  │   bridge QML adapter shells ─────┘
  └ mainscreen.h/.cpp + viewer/   └──► jtml_coordinator (PUBLIC-pulls
      drr_tool/controls/settings/        jtml_services → jtml_domain
      about + list models                 + jtml_compute)
 Shared bottom half = whole layered stack; only the top surface splits
 (QWidgets MainScreen vs QML bridge adapters).

## Start here
`include/view/mainscreen.h` — both the view AND the widget composition root: all include edges, the pure-seam vs view-only slice, and the `SyncSessionState`/`LaunchOptimizer` wires a QML swap must mirror. QML side: `src/app/experimental/AppBridge.h` + `main.qml` (adapter surface + root-context property injection).

## Clarification questions (to the requesting engineer)
1. Is an actual single-view-swappable design intended (one shared view interface, QML _or_ QWidgets at build/link time), or are the two front-ends deliberately parallel forever? The current code is the latter — reverting would change R1/DECISION RECORD.
2. Should the "MVVM" framing be corrected to layered-MVC in docs, or is there a real plan to introduce a true binding VM/selection-model unification? The two selection idioms (`DelegateSelection` vs `QItemSelectionModel`) are the actionable seam, but may be out of scope.
3. The only dead signal in the whole view layer is `MainScreen.UpdateDisplayText(bool)` (never emitted) — worth removing? Severity: low/cosmetic.
4. `Verify whether `FrameListModel`/`ModelListModel` direct-compile (the only jtml_view reuse) is a deliberate stabite-keeping, and if tests should pin behavior instead of including the header pixels.