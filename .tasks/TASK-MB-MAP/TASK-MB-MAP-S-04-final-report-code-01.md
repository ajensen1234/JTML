# TASK-MB-MAP S-04: GUI Subsystem Report

**Stage:** S-04  
**Scope:** Qt5 GUI layer, main application, user interaction  
**Date:** 2026-04-18  
**Analyst:** Claude Sonnet 4.6 (subagent)

---

## 1. Executive Summary

JointTrack Auto GPU (JTML) is a Qt5/VTK5-era biplanar fluoroscopy implant tracking application. The entire GUI lives in a single monolithic `MainScreen` class (~5,806 lines) that acts as both view and controller. Three `Viewer` objects encapsulate VTK scene management; the main viewer (`vw`) and a coronal-plane viewer (`coronal_vw`) are `std::shared_ptr<Viewer>`. Optimization runs in a `QThread` (`optimizer_thread`) owned as a raw pointer alongside `OptimizerManager*`. The DRR (Digitally Reconstructed Radiograph) tool is a modal `QDialog` that directly invokes CUDA via `gpu_cost_function::GPUModel::RenderDRRPrimaryCamera` and copies the result back to the CPU for display. Settings persist via `QSettings` under the registry/config key `"JointTrackAutoGPU"`. The UI is laid out programmatically (not via Qt Designer layouts) by measuring font metrics at construction time, which makes it DPI-sensitive but fragile. There are no custom smart-pointer deleters for Qt objects — the parent-child ownership tree is used throughout for dialog lifetime but several optimization-related pointers (`optimizer_manager`, `optimizer_thread`, `gpu_mod` inside ML routines) are naked and managed manually.

---

## 2. Main Window Structure and Qt Widget Hierarchy

### 2.1 Top-Level Window

**Fact** (`src/gui/main.cpp:11-13`): Entry point constructs a bare `QApplication` + `MainScreen w` (subclass of `QMainWindow`) and calls `w.show()`.

**Fact** (`src/gui/mainscreen.cpp:116`): `MainScreen` inherits `QMainWindow`.

**Fact** (`include/core/mainscreen_size_constants.h:12-13`): Minimum window size is 1600 x 918 px. The application calls `showMaximized()` if the screen is at least this large (`mainscreen.cpp:234-236`).

### 2.2 Left Panel Group Boxes (Preprocessor and Optimization Controls)

**Fact** (`mainscreen.cpp:452-605`): `ArrangeMainScreenLayout()` manually calls `setGeometry()` on every child widget. The left column contains four vertically stacked `QGroupBox` instances, all children of `MainScreen`:

| Widget name | Type | Purpose |
|---|---|---|
| `ui.preprocessor_box` | QGroupBox | Container for load buttons |
| `ui.load_calibration_button` | QPushButton | Load `.txt` calibration file |
| `ui.load_image_button` | QPushButton | Load TIFF/PNG fluoroscopy images |
| `ui.load_model_button` | QPushButton | Load STL implant CAD model(s) |
| `ui.optimization_box` | QGroupBox | Four optimizer trigger buttons |
| `ui.optimize_button` | QPushButton | Optimize current frame/model (Single) |
| `ui.optimize_all_button` | QPushButton | Optimize all frames |
| `ui.optimize_each_button` | QPushButton | Optimize each frame independently |
| `ui.optimize_from_button` | QPushButton | Optimize forward from current frame |
| `ui.image_view_box` | QGroupBox | Image display mode radio buttons |
| `ui.original_image_radio_button` | QRadioButton | Show raw fluoroscopy |
| `ui.inverted_image_radio_button` | QRadioButton | Show inverted/segmented image |
| `ui.edges_image_radio_button` | QRadioButton | Show Canny edge image |
| `ui.dilation_image_radio_button` | QRadioButton | Show morphologically dilated image |
| `ui.image_selection_box` | QGroupBox | Camera A/B selector + frame list |
| `ui.camera_A_radio_button` | QRadioButton | Select primary camera view |
| `ui.camera_B_radio_button` | QRadioButton | Select secondary camera (biplane only) |
| `ui.image_list_widget` | QListWidget | Scrollable list of loaded frames |

### 2.3 Right Panel Group Boxes (Edge Detection and Model Controls)

**Fact** (`mainscreen.cpp:706-914`): Symmetric right column group boxes:

| Widget name | Type | Purpose |
|---|---|---|
| `ui.edge_detection_box` | QGroupBox | Canny edge parameters |
| `ui.aperture_spin_box` | QSpinBox | Canny kernel size (odd 3/5/7) |
| `ui.low_threshold_slider` | QSlider | Low Canny threshold (0–800) |
| `ui.high_threshold_slider` | QSlider | High Canny threshold (0–800) |
| `ui.low_threshold_value` | QLabel | Live numeric display for low slider |
| `ui.high_threshold_value` | QLabel | Live numeric display for high slider |
| `ui.apply_all_edge_button` | QPushButton | Apply current edge settings to all frames |
| `ui.reset_edge_button` | QPushButton | Reset edge settings to defaults |
| `ui.model_view_box` | QGroupBox | Model rendering style radio buttons |
| `ui.original_model_radio_button` | QRadioButton | Surface with diffuse lighting |
| `ui.solid_model_radio_button` | QRadioButton | Solid surface, ambient-only |
| `ui.transparent_model_radio_button` | QRadioButton | 20% opacity surface |
| `ui.wireframe_model_radio_button` | QRadioButton | Wireframe overlay |
| `ui.model_selection_box` | QGroupBox | Single/multiple model selector + list |
| `ui.single_model_radio_button` | QRadioButton | Single-selection mode |
| `ui.multiple_model_radio_button` | QRadioButton | Multi-selection mode |
| `ui.model_list_widget` | QListWidget | Scrollable list of loaded models |

### 2.4 Central VTK Widgets

**Fact** (`mainscreen.cpp:944-949`): Two `QVTKOpenGLNativeWidget` instances fill the central area:

| Widget name | Purpose |
|---|---|
| `ui.qvtk_widget` | Primary fluoroscopy + 3D model rendering viewport |
| `ui.qvtk_cpv` | Coronal plane viewer (CPV) for secondary perspective |

**Fact** (`mainscreen.cpp:932-948`): Both widgets are laid out as squares. The main viewer is sized as the minimum of available width and height between the two side columns.

### 2.5 Progress Overlay

**Fact** (`mainscreen.cpp:301-303`): `ui.pose_progress` (QProgressBar) and `ui.pose_label` (QLabel) are initially hidden and revealed only during ML pose estimation or segmentation runs.

### 2.6 Menu Bar

**Fact** (`mainscreen.cpp:171-197`): Six menus with icons loaded from Qt resources (`:Menu_Icons/Resources/...`):

- **File**: Load Pose, Save Pose, Load/Save Kinematics, Copy Prev/Next Pose, Quit
- **View**: Reset View, Reset Normal Up, Model Interaction Mode, Camera Interaction Mode
- **Segment**: Segment FemHR, Segment TibHR, Estimate Femoral Implant, Estimate Tibial Implant, Ambiguous Pose Processing, Reset/Remove All Segmentation
- **Options**: Optimizer Settings, Region Selection, Center Placement, Stop Optimizer, DRR Settings
- **Help**: About JointTrack Auto, Controls

### 2.7 Modal Dialogs (Child Windows)

**Fact** (`mainscreen.cpp:142`): `SettingsControl* settings_control` is created as a `new SettingsControl(this)` child in the constructor. It is `show()`-ed modeless.

**Fact** (`mainscreen.cpp:1683-1685`): `About` is constructed as a local stack variable and executed with `abt.exec()` (modal).

**Fact** (`mainscreen.cpp:2568-2569`): `Controls` is a local variable with `.exec()` (modal).

**Fact** (`mainscreen.cpp:2591-2595`): `DRRTool` is a local variable with `.exec()` (modal). It receives a `Model`, `CameraCalibration`, and z-depth from the current state.

---

## 3. User Workflow (Primary Flow)

**Fact** (derived from button disable/enable logic and slot implementations):

```
1. START
   └─ Load Calibration (.txt, format JT_INTCALIB / JTA_INTCALIB / JTA_INTCALIB_BIPLANE)
      └─ VTK renderers configured; load_calibration_button disabled permanently
         └─ Load Image(s) (.tif/.tiff/.png)
            └─ Canny edge + dilation images computed; frames added to image_list_widget
               └─ Load Model(s) (.stl STL files)
                  └─ VTK actors/mappers created; models added to model_list_widget
                     ├─ [MANUAL] Select frame in image_list_widget
                     │  └─ Select model in model_list_widget
                     │     └─ Drag/keyboard-adjust model pose in VTK viewport
                     │        └─ Save Pose / Save Kinematics
                     │
                     └─ [AUTOMATED] Optimizer Settings (SettingsControl dialog)
                        └─ Click Optimize / Optimize All / Optimize Each / Optimize From
                           └─ OptimizerManager created; moved to QThread
                              └─ Signals update VTK actors in real-time
                                 └─ Final pose stored in model_locations_ (LocationStorage)
                                    └─ Save Kinematics (.jtak)
```

**Fact** (`mainscreen.cpp:4613-4779`): Optimizer thread launch packages calibration, all frames, selected models, `OptimizerSettings`, and three `CostFunctionManager` instances (trunk/branch/leaf), then starts `optimizer_thread`.

**Fact** (`mainscreen.cpp:1688-1834`): Neural network pose estimation (Segment Fem/TibHR + Estimate Fem/Tib) path: loads a TorchScript `.pt` model onto CUDA, runs inference per frame, computes Z from projected vs. segmented area, gets X/Y from phase correlation, converts to ZXY Euler angles, stores in `model_locations_`.

---

## 4. Key UI Controls and Their Purposes

| Control | Signal / Slot | Effect |
|---|---|---|
| `load_calibration_button` | `on_load_calibration_button_clicked` | Parses calibration TXT; configures both Viewer objects; permanently disables button (**Fact** `mainscreen.cpp:2768`) |
| `load_image_button` | `on_load_image_button_clicked` | Opens multi-file dialog; creates `Frame` objects (with edge/dilation images); populates `image_list_widget` (**Fact** `mainscreen.cpp:2807`) |
| `load_model_button` | `on_load_model_button_clicked` | Opens multi-file STL dialog; creates `Model` and VTK actor/mapper pairs via `vw->load_models()` (**Fact** `mainscreen.cpp:2982`) |
| `optimize_button` | calls `LaunchOptimizer("Single")` | Optimize current frame only (**Fact** `mainscreen.cpp:4505`) |
| `optimize_all_button` | calls `LaunchOptimizer("All")` | Optimize all frames, uses current pose as seed for next (**Fact** `mainscreen.cpp:4510`) |
| `optimize_each_button` | calls `LaunchOptimizer("Each")` | Optimize each frame independently (**Fact** `mainscreen.cpp:4515`) |
| `optimize_from_button` | calls `LaunchOptimizer("From")` | Optimize all frames starting from current frame forward (**Fact** `mainscreen.cpp:4520`) |
| `apply_all_edge_button` | `on_apply_all_edge_button_clicked` | Re-runs Canny + dilation on all loaded frames with current slider values (**Fact** `mainscreen.cpp:4430`) |
| `actionStop_Optimizer` | `on_actionStop_Optimizer_triggered` | Emits `StopOptimizer()` signal; connected with `Qt::DirectConnection` to optimizer (**Fact** `mainscreen.cpp:1539`) |
| Image view radio buttons | click handlers | Call `vw->update_display_background_to_*_image()` (**Fact** `mainscreen.cpp:3957-4005`) |
| Model view radio buttons | click handlers | Call `vw->change_model_opacity_to_*()` (**Fact** `mainscreen.cpp:4007-4074`) |
| `image_list_widget` selection | `on_image_list_widget_currentRowChanged` | Calls `SaveLastPose()` then restores model pose for new frame selection |
| Keyboard in VTK | `KeyPressInteractorStyle::OnKeyPress` | Arrow: translate/rotate; Shift+Arrow: X/Y rotate; Ctrl+Arrow: Z-translate/Z-rotate; `+`/`-`: speed; `i`/`I`: toggle info overlay; `p`/`P`: set principal model (**Fact** `interactor.cpp:48-287`) |

---

## 5. VTK Integration for Rendering

**Fact** (`viewer.h:42`, `viewer.cpp:17-50`): `Viewer` is a plain C++ class (not a Qt object) wrapping all VTK smart pointers. It has two `vtkSmartPointer<vtkRenderer>` instances per viewport:

- `background_renderer_` (Layer 0, Interactive Off): renders the fluoroscopy image via `vtkImageImport` -> `vtkDataSetMapper` -> `vtkActor`
- `scene_renderer_` (Layer 1, Interactive On): renders all 3D implant STL models as `vtkPolyDataMapper` -> `vtkActor` pairs

**Fact** (`viewer.cpp:365-380`): `load_renderers_into_render_window()` calls `qvtk_render_window_->SetNumberOfLayers(2)` and adds both renderers; the background is non-interactive and the scene layer handles mouse/key events.

**Fact** (`viewer.cpp:418-498`): Camera parameters are computed from intrinsic calibration (`fx`, `fy`, `cx`, `cy`) using: viewing angle from `atan2(h, 2*fy)`, window center shift from principal point offset, and aspect from `fx/fy` via a `vtkMatrix4x4` user transform.

**Fact** (`mainscreen.cpp:268-298`): Two `Viewer` instances exist: `vw` (main viewport, `ui.qvtk_widget`) and `coronal_vw` (secondary viewport, `ui.qvtk_cpv`). Both share the same `calibration_file_` but `coronal_vw` has its camera set to `(-1, 1, 0)` for a coronal view (**Fact** `viewer.cpp:167-169`).

**Fact** (`interactor.cpp:31-288`): `KeyPressInteractorStyle` (subclasses `vtkInteractorStyleTrackballActor`) holds a raw back-pointer to `MainScreen*` and a `shared_ptr<Viewer>`. A second style `CameraInteractorStyle` (subclasses `vtkInteractorStyleTrackballCamera`) is used in Camera Interaction Mode. The active style is swapped at runtime.

**Fact** (`mainscreen.cpp:199-202`): View mode (Model vs Camera Interaction) is managed by a `QActionGroup` (`alignmentGroup`) so only one can be checked at a time.

**Fact** (`mainscreen.cpp:68-83`): `matToVTK()` converts `cv::Mat` to `vtkImageData` via `vtkImageImport` (direct pointer passthrough, no copy for grayscale; the caller must keep the `cv::Mat` alive).

---

## 6. How the GUI Connects to Core/GPU Layers

### 6.1 Calibration Layer

**Fact** (`mainscreen.cpp:2601-2804`): `on_load_calibration_button_clicked()` parses the calibration file and constructs `Calibration calibration_file_` (from `include/core/calibration.h`). This object is passed by value into both `Viewer::setup_camera_calibration()` and later into `OptimizerManager::Initialize()`.

### 6.2 Frame / Image Processing Layer

**Fact** (`mainscreen.cpp:2843-2979`): `Frame` objects (from `include/core/frame.h`) are constructed with aperture and threshold values from the edge detection spinboxes. `Frame` internally runs Canny edge detection and dilation during construction, caching all four image variants (original, inverted, edges, dilated) as `cv::Mat`. Distance maps and curvature heatmaps are also computed here.

### 6.3 Optimizer Thread Layer

**Fact** (`mainscreen.cpp:4652-4779`): `LaunchOptimizer()` constructs `OptimizerManager* optimizer_manager` (from `include/core/optimizer_manager.h`) and `QThread* optimizer_thread`, moves the manager to the thread, then connects eight signals:

| Signal (OptimizerManager) | Slot (MainScreen) | Purpose |
|---|---|---|
| `UpdateDisplay(double,int,double,uint)` | `onUpdateDisplay` | Progress during evaluation |
| `OptimizerError(QString)` | `onOptimizerError` | Error message display |
| `UpdateOptimum(6x double, uint)` | `onUpdateOptimum` | Move blue "current best" actor |
| `OptimizedFrame(6x double, bool, uint, bool, QString)` | `onOptimizedFrame` | Final pose for frame; advance list |
| `UpdateDilationBackground()` | `onUpdateDilationBackground` | Refresh dilated image display |
| `onUpdateOrientationSymTrap(...)` | `updateOrientationSymTrap_MS` | Sym-Trap optimizer update |
| `StopOptimizer()` (from MainScreen) | `onStopOptimizer` | Stop signal (DirectConnection) |

**Fact** (`mainscreen.cpp:4776`): `currently_optimizing_` flag prevents certain keyboard actions during optimization.

### 6.4 GPU Layer (CUDA / PyTorch TorchScript)

**Fact** (`drr_tool.cpp:65-83`): `DRRTool` directly instantiates `gpu_cost_function::GPUModel` (raw pointer) in its constructor with CUDA context. `DrawDRR()` calls `gpu_model_->RenderDRRPrimaryCamera(Pose(...))` then `cudaMemcpy` to `host_image_` and assigns to `QLabel` as a `QPixmap`.

**Fact** (`mainscreen.cpp:54-58`): `mainscreen.cpp` includes `<torch/script.h>`, `<torch/cuda.h>`, and `<c10/cuda/CUDACachingAllocator.h>`. Neural network inference is done inline in the GUI thread (blocking) via `torch::jit::load(path, torch::kCUDA)` and `model->forward(inputs)`.

**Fact** (`mainscreen.cpp:1789`): `c10::cuda::CUDACachingAllocator::emptyCache()` is called after each frame's segmentation to manage VRAM on limited GPUs.

*Inference* (no explicit threading found for ML): The ML pose estimation (`on_actionEstimate_Femoral_Implant_s_triggered`, ~lines 1847-2208) runs entirely on the GUI thread with periodic `qApp->processEvents()` calls for UI responsiveness. This is a UI-freeze risk.

---

## 7. Settings Management

**Fact** (`mainscreen.cpp:4235-4240`): All edge detection settings persist to `QSettings("JointTrackAutoGPU", Version)` under the group `"EdgeDetectionSettings"` with keys `APERTURE`, `LOW_THRESH`, `HIGH_THRESH`. This is done on every slider/spinbox change.

**Fact** (`mainscreen.cpp:5540-5725`): `onSaveSettings()` (triggered when SettingsControl dialog saves) persists all three `CostFunctionManager` states (trunk/branch/leaf) plus `OptimizerSettings` to `QSettings`. Keys use an `@`-delimited scheme: `"TRUNK@<CostFunctionName>@<ParameterName>@<ParameterType>"`.

**Fact** (`mainscreen.cpp:129`): Constructor calls `LoadSettingsBetweenSessions()` first (before any widget setup), which reads from `QSettings` and populates `trunk_manager_`, `branch_manager_`, `leaf_manager_`, and `optimizer_settings_`.

**Fact** (`include/core/settings_constants.h:18-20`): Version string is built from constants `VER_FIRST_NUM=3`, `VER_MIDDLE_NUM=4`, `VER_LAST_NUM=0`, yielding settings key `"Version340"`.

**Fact** (`include/core/settings_constants.h:23-52`): Default optimizer values: Trunk range ±35mm/deg, budget 20,000; Branch range varies (15-25), budget 5,000, 2 branches; Leaf not shown with defaults. Edge defaults: aperture 3, low 40, high 120.

*Inference*: There is no import/export of settings as a file; the only persistence mechanism is the platform `QSettings` registry/config store.

---

## 8. DRR Tool Functionality

**Fact** (`drr_tool.cpp:21-116`): `DRRTool` is a `QDialog` opened modally from `on_actionDRR_Settings_triggered()` (`mainscreen.cpp:2585-2596`). It receives the currently selected `Model`, the primary `CameraCalibration`, and the model's current Z depth.

**Fact** (`drr_tool.cpp:65-76`): `gpu_cost_function::GPUModel* gpu_model_` is a raw-pointer GPU model initialized in the constructor. If initialization fails, a `QMessageBox::critical` is shown and the dialog closes.

**Fact** (`drr_tool.cpp:126-157`): `DrawDRR()`:
1. Calls `gpu_model_->RenderDRRPrimaryCamera(Pose(...), min_val, max_val)` with slider-derived Hounsfield-like threshold range.
2. `cudaMemcpy` from device to `host_image_` (malloc'd `unsigned char*`).
3. Wraps in `QImage(host_image_, w, h, w, Format_Grayscale8)` mirrored vertically.
4. Sets `QLabel::setPixmap`.

**Fact** (`drr_tool.cpp:160-243`): Eight slot handlers (`on_minLowerSpinBox_valueChanged`, etc.) all call `DrawDRR()` after updating derived display values. The tool has two slider-spinbox pairs: `minSlider`/`minLowerSpinBox`/`minUpperSpinBox` and matching `max*` controls, mapping a [0,1000] slider position to a [lower, upper] Hounsfield range.

**Fact** (`drr_tool.cpp:118-124`): Destructor calls `delete gpu_model_` and `free(host_image_)`. No RAII wrapper.

**Fact** (`drr_tool.cpp:18`): `vtkSmartPointer<DRRInteractorStyle> drr_interactor` is a file-scope global (not a member), creating a potential lifetime issue if multiple DRRTool instances were ever opened.

---

## 9. Raw Pointer Qt Patterns

**Fact** (`mainscreen.cpp:142`): `SettingsControl* settings_control = new SettingsControl(this)` — Qt parent-child ownership, destructor handled by Qt. Correct pattern.

**Fact** (`mainscreen.cpp:199`): `QActionGroup* alignmentGroup = new QActionGroup(this)` — Qt parent-child ownership. Correctly deleted in destructor (`mainscreen.cpp:320`), which is redundant but harmless.

**Fact** (`mainscreen.cpp:4652-4653`):
```cpp
optimizer_manager = new OptimizerManager();
optimizer_thread = new QThread();
```
These are raw pointers stored as member variables. `optimizer_manager->moveToThread(optimizer_thread)` means Qt owns cleanup when the thread finishes if `deleteLater()` is connected. *Inference*: It is unclear whether `deleteLater()` is wired up; a memory leak or double-delete is possible if the optimizer is launched multiple times.

**Fact** (`drr_tool.cpp:65`): `gpu_model_ = new gpu_cost_function::GPUModel(...)` — raw pointer, manually deleted in destructor. Acceptable but fragile.

**Fact** (`mainscreen.cpp:1907`, `2271`): `auto gpu_mod = new GPUModel(...)` inside ML estimation routines — raw pointer, manually `delete gpu_mod` called before return. Risk: if an early return is added later, leak will occur.

**Fact** (`mainscreen.cpp:1966`, `2328`): `auto orientation = new float[3]` — raw pointer array, never `delete[]`'d. **This is a confirmed memory leak.**

**Fact** (`controls.cpp:12-15`): `Controls` constructor does `center_scene = new QGraphicsScene` and `center_graph = new QGraphicsView()` with `center_graph->setParent(this)`. The scene is manually deleted in the destructor (line 30-31). However, `setParent(this)` means Qt will also delete `center_graph` when the dialog closes, and the destructor then calls `delete center_graph` again — **potential double-free**.

**Fact** (`drr_tool.cpp:18`): File-scope global `vtkSmartPointer<DRRInteractorStyle> drr_interactor` is initialized in the `DRRTool` constructor. If `DRRTool` is opened more than once, the previous smart pointer is overwritten, potentially deleting the old interactor while the old dialog is still alive.

---

## 10. C4 L2-L3: GUI Subsystem Structure

```
[C4 Level 2 - Container View]

┌──────────────────────────────────────────────────────────────────────┐
│  JointTrack Auto GPU Desktop App                                     │
│                                                                      │
│  ┌─────────────────────┐   ┌──────────────────────────────────────┐ │
│  │   Qt GUI Layer      │   │         Core / GPU Layer             │ │
│  │                     │   │                                      │ │
│  │  MainScreen         │──>│  OptimizerManager (QThread)          │ │
│  │  (QMainWindow)      │   │  CostFunctionManager (x3)            │ │
│  │                     │   │  GPUModel (CUDA)                     │ │
│  │  Viewer (x2, C++)   │   │  Frame (OpenCV processing)           │ │
│  │  SettingsControl    │   │  Calibration / Model                 │ │
│  │  DRRTool            │   │  LocationStorage                     │ │
│  │  About / Controls   │   │  TorchScript (PyTorch CUDA)          │ │
│  └─────────────────────┘   └──────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘

[C4 Level 3 - GUI Component View]

MainScreen (QMainWindow, 5806 lines)
├── Left Column Group Boxes (preprocessor, optimizer directives, image view, image selection)
├── Central VTK Widgets
│   ├── ui.qvtk_widget  ─── shared_ptr<Viewer> vw  (background + scene renderers)
│   └── ui.qvtk_cpv     ─── shared_ptr<Viewer> coronal_vw  (coronal plane)
├── Right Column Group Boxes (edge detection, model view, model selection)
├── Menu Bar (File / View / Segment / Options / Help)
├── Progress overlay (pose_progress, pose_label)
├── SettingsControl* settings_control  [owned child, modeless]
│   └── CostFunctionManager editors (trunk/branch/leaf stages)
├── [modal, transient] About (QDialog)
├── [modal, transient] Controls (QDialog, QGraphicsView with keybindings PNG)
├── [modal, transient] DRRTool (QDialog + VTK + CUDA GPUModel)
└── Optimization subsystem
    ├── OptimizerManager* optimizer_manager  [raw ptr, moved to thread]
    └── QThread* optimizer_thread  [raw ptr]
```

---

## 11. Primary User Flow Description

1. **Setup phase** (one-time per session): User loads a calibration file, then one or more fluoroscopy image sequences, then one or more STL implant models. These three steps must occur in order; the buttons are conditionally enabled to enforce this.

2. **Manual alignment** (optional): User selects a frame from the image list and a model from the model list. The 3D STL model overlays the fluoroscopy background in the VTK viewport. The user drags (mouse) or nudges (keyboard arrows) the model to roughly align it with the implant silhouette in the X-ray image.

3. **Automated optimization**: User clicks an optimize button. The optimizer runs in a background thread using a Trunk/Branch/Leaf three-stage stochastic search (similar to CMA-ES or grid search structure) against a GPU-computed cost function. The VTK display updates in real time as better poses are found.

4. **ML-assisted initialization** (optional, Segment menu): User loads PyTorch TorchScript `.pt` files for segmentation and pose regression. The app runs NN inference on the GPU to produce an initial pose estimate, which is then refined by the optimizer.

5. **Export**: User saves per-frame pose as `.jtap` or all-frames kinematics as `.jtak`.

---

## 12. Risks

| ID | Risk | Severity | Evidence |
|---|---|---|---|
| R-01 | **ML inference on GUI thread**: `on_actionEstimate_Femoral/Tibial_Implant_s_triggered` runs blocking CUDA+TorchScript inference in the main event loop with only `qApp->processEvents()` as relief. UI is effectively frozen for the duration (can be minutes). | High | `mainscreen.cpp:1847-2208, 2210-2543` |
| R-02 | **Memory leak — `orientation` array**: `new float[3]` allocated in two ML routines, never `delete[]`'d | Medium | `mainscreen.cpp:1966, 2328` |
| R-03 | **Potential double-free in Controls dialog**: `center_graph` is parented to `this` (Qt will delete) AND explicitly `delete`'d in destructor | Medium | `controls.cpp:15-31` |
| R-04 | **File-scope global VTK interactor in DRRTool**: Opening DRRTool a second time overwrites the global `drr_interactor` smart pointer, potentially destroying the interactor of a still-open dialog | Medium | `drr_tool.cpp:18` |
| R-05 | **Raw optimizer thread not cleaned up**: `optimizer_manager` and `optimizer_thread` are re-allocated on each optimize call without deleting or `wait()`-ing the previous thread | High | `mainscreen.cpp:4652-4653` |
| R-06 | **`goto` in image loading**: `goto stop` and `goto stop_biplane` jump out of for-loops in `on_load_image_button_clicked`. While functional, they make control flow analysis difficult and may skip cleanup | Low | `mainscreen.cpp:2873, 2934` |
| R-07 | **Monolithic MainScreen**: At 5,806 lines with direct member access to VTK objects, UI widgets, CUDA code, and file I/O, the class is untestable in isolation and requires full Qt + VTK + CUDA environment for any modification | High | Architecture-wide |
| R-08 | **`mainscreen_size_constants.h` defines non-const globals**: `MINIMUM_WIDTH` and `MINIMUM_HEIGHT` are non-const `int` (not `const int`) which means including this header in multiple TUs causes ODR violations | Medium | `include/core/mainscreen_size_constants.h:12-13` |
| R-09 | **`matToVTK` does not copy pixel data**: `vtkImageImport::SetImportVoidPointer` stores a raw pointer to `cv::Mat::data`. If the `cv::Mat` is destroyed or reallocated, VTK has a dangling pointer | High | `mainscreen.cpp:68-83`, `viewer.cpp:105` |
| R-10 | **Settings persistence uses a version-keyed store**: If the version number changes, all user settings are silently lost with no migration path | Low | `mainscreen.cpp:4235`, `settings_constants.h:18-20` |

---

## 13. File Inventory

| File | Lines | Role |
|---|---|---|
| `src/gui/main.cpp` | 14 | Entry point: `QApplication` + `MainScreen` |
| `src/gui/mainscreen.cpp` | 5,806 | Main window: all UI logic, optimizer launch, ML inference |
| `src/gui/viewer.cpp` | 509 | VTK scene wrapper (background + scene renderers) |
| `src/gui/interactor.cpp` | 291 | Custom VTK interactor styles (keyboard/mouse) |
| `src/gui/drr_tool.cpp` | 243 | DRR modal dialog with embedded CUDA GPUModel |
| `src/gui/about.cpp` | 156 | About dialog: CUDA device detection, version display |
| `src/gui/settings_control.cpp` | 1,260 | Optimizer settings dialog: 3-stage cost function editor |
| `src/gui/controls.cpp` | 32 | Controls reference dialog: QGraphicsView displaying a PNG |
| `include/gui/viewer.h` | 177 | Viewer class declaration |
| `include/core/mainscreen_size_constants.h` | 47 | Layout pixel constants + font size |
| `include/core/settings_constants.h` | 60 | Default optimizer/edge/version constants |
| `src/gui/mainscreen.ui` | (Qt Designer XML) | Widget tree definition |
| `src/gui/drr_tool.ui` | (Qt Designer XML) | DRR dialog widget tree |
| `src/gui/settings_control.ui` | (Qt Designer XML) | Settings dialog widget tree |
| `src/gui/about.ui` | (Qt Designer XML) | About dialog widget tree |
| `src/gui/controls.ui` | (Qt Designer XML) | Controls dialog widget tree |
