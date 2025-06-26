# TASK-MB-MAP Stage S-02: Core & Cost Functions Layer — Code Report

**Repo:** /home/ajj/repo/uf/JTML
**Scope:** src/core/**, src/cost_functions/**, include/core/**, include/cost_functions/**
**Date:** 2026-04-18

---

## Executive Summary

JTML is a 6-DOF fluoroscopic bone/implant pose estimation system (JTA-GPU variant) built in C++/CUDA/Qt. The core layer implements the DIRECT global optimizer using a 6D hyperrectangle search space; cost function evaluation is delegated to a plugin-style `CostFunctionManager` compiled as a shared library (`libJTA_Cost_Functions.so`). All GPU resources (frames, models, metrics) are managed as raw pointers owned by `OptimizerManager`, allocated in `Initialize()` and deleted in the destructor — there are no smart pointers in GPU resource management. A three-stage optimization pipeline (Trunk → Branch → Leaf) with per-stage cost function selection drives convergence. Machine learning segmentation (`segment_image`) uses LibTorch JIT on CUDA. There is a known bug in `Point6D`'s `Pose` constructor where `ya` is never set. Several global mutable variables are declared in header files via `#include` inside class definitions — a significant ODR/linkage risk. The `Parameter<double>` class stores its value as `int` (type mismatch bug). The `CostFunctionManager::getActiveCostFunctionClass()` leaks a `new CostFunction()` on the fallback path.

---

## 1. Key Classes and Their Responsibilities

### 1.1 `Point6D` (struct)
**Fact** (`include/core/data_structures_6D.h:27`)

Represents a 6-degree-of-freedom pose: x, y, z (translations, mm) and xa, ya, za (Euler angles, degrees, 3-1-2 order). Used pervasively as the unit of pose exchange between all subsystems.

Methods: `GetDistanceFrom`, `GetLargestDirection`, `GetDirection`, `UpdateDirection`.

**Bug (Fact)** (`src/core/data_structures_6D.cpp:40`): The `Point6D(gpu_cost_function::Pose p)` constructor sets `this->xa = p.z_angle_` twice and never sets `ya`. The `ya` field will be uninitialized/zero after construction from a `Pose`.

### 1.2 `HyperBox6D` (struct)
**Fact** (`include/core/data_structures_6D.h:56`)

Stores a hyperrectangle in the 6D search space for the DIRECT algorithm. Contains:
- `value_` — cost function value at the center
- `size_` — L2 norm of the side lengths vector
- `center_` — `Point6D` center
- `sides_` — `Point6D` half-side lengths

Methods: `TrisectSide`, `containsPoint`, `PrintCenter`. Used internally by `DirectDataStorage` and `OptimizerManager`.

### 1.3 `DirectDataStorage`
**Fact** (`include/core/direct_data_storage.h:19`, `src/core/direct_data_storage.cpp`)

Implements the hyperrectangle storage matrix for the DIRECT optimizer. Structure: `std::vector<std::vector<HyperBox6D*>*> storage_matrix_` — a vector of columns, each column being a sorted vector of pointers to `HyperBox6D` objects with the same L2 size. Columns are sorted by ascending size (smallest first); within each column, hyperboxes are sorted by descending value (minimum value at back).

Companion vectors `minimum_value_columns_` and `size_columns_` cache the minimum value and size per column for fast convex hull access.

**Risk:** The destructor is commented out (`src/core/direct_data_storage.cpp:45–48`). Callers must call `DeleteAllStoredHyperboxes()` explicitly, or the memory leaks. This is done in `OptimizerManager::Optimize()` at frame boundaries.

**Risk:** Comments in `AddHyperBox` and `DeleteHyperBoxes` note "(NOT SAFE)" for minimum container manipulation — these could go out of sync if invariants are violated.

### 1.4 `LocationStorage`
**Fact** (`include/core/location_storage.h:18`)

A 2D matrix (`std::vector<std::vector<Point6D>>`) indexed by [frame_index][model_index]. Stores the current best pose for each model at each frame. Also maintains `no_image_location_storage_vector_` for the case where no frames are loaded. The default initialization pose for each new model is `(0, 0, -0.25 * principal_distance / pixel_pitch, 0, 0, 0)`.

Interface: `LoadNewModel`, `LoadNewFrame`, `GetPose`, `SavePose`, `GetFrameCount`, `GetModelCount`.

### 1.5 `Frame`
**Fact** (`include/core/frame.h:28`, `src/core/frame.cu`)

Manages one X-ray image and all its derived representations, all stored as `cv::Mat`:
- `original_image_` — raw grayscale (flipped vertically on load)
- `inverted_image_` — 255 - original
- `edge_image_` — Canny edge detection result
- `dilation_image_` — morphological dilation of the edge image
- `distance_map_` — distance transform of the inverted edge image (CV_DIST_L1)
- `curvature_heatmaps_` — vector of heatmaps from curvature keypoints (flattened to `curvature_heatmap_chars_` for GPU upload)

`SetDistanceMap()` inverts the edge image before running the distance transform, because OpenCV's distanceTransform computes distance to nearest black pixel.

`setCurvatureHeatmaps()` must be called explicitly after construction; it is NOT called in the constructor.

### 1.6 `Model`
**Fact** (`include/core/model.h:24`, `src/core/model.cpp`)

Loads an STL file and stores:
- `cad_reader_` — `vtkSmartPointer<vtkSTLReader>` (one use of smart pointer — VTK's own)
- `triangle_vertices_` — `std::vector<float>` (flat array, 9 floats per triangle)
- `triangle_normals_` — `std::vector<float>` (flat array, 9 floats per triangle)

**Note (Fact):** `vtkSmartPointer` is used here for the VTK reader — the only smart pointer in the core layer. All GPU objects use raw pointers.

### 1.7 `Calibration` (struct)
**Fact** (`include/core/calibration.h:83`)

Supports monoplane (one camera) and biplane (two cameras). Contains `CameraCalibration camera_A_principal_`, `CameraCalibration camera_B_principal_`, `Vect_3 origin_B_`, `Matrix_3_3 axes_B_`. Provides `convert_Pose_A_to_Pose_B` and `convert_Pose_B_to_Pose_A` using Z-X-Y Euler angle recovery with gimbal lock handling. Supports calibration `type_` ("UF" or "Denver") for Z-axis sign convention in default initialization.

### 1.8 `OptimizerSettings` (struct)
**Fact** (`include/core/optimizer_settings.h:16`, `src/core/optimizer_settings.cpp`)

Plain data struct carrying optimizer configuration. Declared `Q_DECLARE_METATYPE` for Qt signal/slot transport. Default values from `settings_constants.h`:
- Trunk: range ±35mm/deg all axes, budget 20,000 evaluations
- Branch: range ±15mm/±25deg, budget 5,000, 2 branches
- Leaf ("Z search"): range ±3mm/deg except ±15mm Z, budget 5,000
- `enable_branch_` and `enable_leaf_` default true

### 1.9 `OptimizerManager` (QObject)
**Fact** (`include/core/optimizer_manager.h:51`, `src/core/optimizer_manager.cpp`)

The central orchestration class. Runs on a dedicated `QThread`. Implements the full DIRECT algorithm inline:
- `Initialize(...)` — allocates all GPU resources, validates inputs, sets up cost function managers
- `Optimize()` (slot) — the main optimization loop; runs Trunk → Branch (×N) → Leaf for each frame
- `ConvexHull()` — Jarvis's Gift Wrapping on `DirectDataStorage`
- `TrisectPotentiallyOptimal()` — trisects each potentially optimal hyperbox and evaluates cost at two new centers (allocates new `HyperBox6D*` via `new`)
- `EvaluateCostFunction()` — dispatches to the active cost function manager based on `search_stage_flag_`
- `CalculateSymTrap()` — sweeps a list of poses through the leaf cost function, outputs Results.csv/.xyz/.xy to the CWD
- Destructor — deletes all GPU objects manually

**Optimization directives:** `"Single"`, `"All"`, `"Each"`, `"From"`, `"Backward"`, `"Sym_Trap"`.

### 1.10 `CostFunctionManager` (namespace `jta_cost_function`)
**Fact** (`include/cost_functions/CostFunctionManager.h`, `src/cost_functions/CostFunctionManager.cpp`)

Plugin-style manager. Each instance is bound to one `Stage` (Trunk/Branch/Leaf). Holds:
- A vector of registered `CostFunction` objects
- The name of the active cost function
- Raw pointers to GPU data (frames, models, metrics) — not owned, set by `UploadData()`/`UploadDistanceMap()`
- Per-cost-function state variables injected via `#include` of `*CustomVariables.h` files directly inside the class definition

The dispatch is a large if/else chain in `callActiveCostFunction()`, `InitializeActiveCostFunction()`, and `DestructActiveCostFunction()`.

**Memory leak (Fact, `src/cost_functions/CostFunctionManager.cpp:247,261`):** `getActiveCostFunctionClass()` and `getCostFunctionClass()` return `new CostFunction()` on the fallback/not-found path — these are never freed by callers.

### 1.11 `CostFunction` (namespace `jta_cost_function`)
**Fact** (`include/cost_functions/CostFunction.h:18`)

Stores the name and typed parameter lists (double, int, bool) for one cost function variant. Parameters use `Parameter<T>` template with `JTML_DLL` export macros. Parameters are looked up by name string at runtime.

### 1.12 `Parameter<T>` (template, namespace `jta_cost_function`)
**Fact** (`include/cost_functions/Parameter.h:26`)

Specializations for `double`, `int`, `bool`. Only `double`, `int`, `bool` are allowed (enforced by `static_assert`).

**Bug (Fact, `include/cost_functions/Parameter.h:73`):** The `Parameter<double>` specialization stores `parameter_value_` as `int parameter_value_` — the member declaration uses `int` instead of `double`. This means all double parameter values are silently truncated to integers.

---

## 2. Data Structures

| Structure | Location | Purpose |
|---|---|---|
| `Point6D` | `include/core/data_structures_6D.h:27` | 6-DOF pose (x,y,z,xa,ya,za) |
| `HyperBox6D` | `include/core/data_structures_6D.h:56` | DIRECT search hyperrectangle |
| `DirectDataStorage` | `include/core/direct_data_storage.h:19` | Sorted matrix of HyperBox6D* for DIRECT |
| `LocationStorage` | `include/core/location_storage.h:18` | 2D pose matrix [frame][model] |
| `Calibration` | `include/core/calibration.h:83` | Camera calibration + coordinate transforms |
| `OptimizerSettings` | `include/core/optimizer_settings.h:16` | DIRECT stage budgets, ranges, flags |
| `Vect_3`, `Matrix_3_3` | `include/core/calibration.h:14,33` | Small linear algebra helpers for biplane transforms |

---

## 3. Frame/Model System

**Fact:** `Frame` objects (`src/core/frame.cu`) are value-typed and passed by `std::vector<Frame>` copy into `OptimizerManager::Initialize`. The GPU frame objects (`GPUEdgeFrame*`, `GPUDilatedFrame*`, `GPUIntensityFrame*`, `GPUFrame*`) are separate raw-pointer GPU counterparts created from the CPU `Frame` data.

**Fact:** Per-stage dilation levels require separate GPU frame sets — `OptimizerManager` allocates 3 separate `GPUIntensityFrame*` vectors (trunk/branch/leaf × A/B) and 3 `GPUDilatedFrame*` vectors per camera. The dilation is reapplied to the CPU `Frame::dilation_image_` on-the-fly before each stage transition.

**Fact:** `Model` holds triangle vertex/normal data as flat `std::vector<float>` (9 floats per triangle). `GPUModel*` is the GPU counterpart, also raw pointer.

*Inference:* `setCurvatureHeatmaps()` on `Frame` must be called externally before the frame is passed to `OptimizerManager`, or `GetNumCurvatureKeypoints()` returns an uninitialized value and the GPU heatmap upload will silently upload 0 bytes.

---

## 4. Optimization Subsystem

### 4.1 Algorithm: DIRECT in 6D

The optimizer implements the DIRECT (DIviding RECTangles) algorithm in 6-dimensional pose space, normalized to [0,1]^6. The algorithm:

1. Initialize with one unit hyperrectangle at (0.5)^6; evaluate cost there
2. `ConvexHull()` — find potentially optimal hyperrectangles via Jarvis's Gift Wrapping on the (size, min_value) frontier
3. `TrisectPotentiallyOptimal()` — for each potentially optimal box, split the longest denormalized dimension into thirds; evaluate at the two new centers; insert three boxes back into storage
4. Repeat until budget exhausted

**Fact** (`src/core/optimizer_manager.cpp:883`): `Optimize()` runs stages sequentially: Trunk always executes; Branch runs `number_branches` times if `enable_branch_`; Leaf runs once if `enable_leaf_`. Each stage re-initializes storage, resets the starting point to the current optimum, and adds its own budget to the cumulative counter.

### 4.2 `OptimizerSettings` Stages

| Stage | Default Range | Default Budget | Notes |
|---|---|---|---|
| Trunk | ±35 all axes | 20,000 | Always active |
| Branch | ±15mm / ±25deg | 5,000 | ×2 iterations |
| Leaf | ±3mm/deg, ±15mm Z | 5,000 | Optional |

**Fact** (`src/core/optimizer_settings.cpp:27`): Leaf is called "Z search" internally and its range constant is `Z_SEARCH_RANGE`.

### 4.3 Search Stage Flags

**Fact** (`include/core/metric_enum.h:9`): `SearchStageFlag { Trunk=0, Branch=1, Leaf=2 }` — used in `OptimizerManager::search_stage_flag_` to dispatch cost function calls.

*Note:* There is also `enum class Stage { Trunk, Branch, Leaf }` in `include/cost_functions/Stage.h` — a parallel enum for `CostFunctionManager`'s stage context. `DIRECT_DILATION_T1` checks `stage_` to switch dilation level in trunk vs. other stages.

---

## 5. Cost Function Types

All cost functions live in namespace `jta_cost_function` and are implemented as methods of `CostFunctionManager`.

### 5.1 `DIRECT_DILATION` (default)
**Fact** (`src/cost_functions/DIRECT_DILATION.cpp`)

Score = `sum_white_pixels_dilated_comparison_A + FastImplantDilationMetric(render_A, dilated_frame_A, dilation)` + `DistanceMapMetric(render_A, distance_map_A, dilation)`
In biplane mode: + `(score_B)^2`

Parameters: `Dilation` (int, default 6). **Special case:** if cost function name is `DIRECT_MAHFOUZ`, `Initialize()` hard-codes `dilation=3` regardless of parameter.

### 5.2 `DIRECT_MAHFOUZ`
**Fact** (`src/cost_functions/DIRECT_MAHFOUZ.cpp`)

Uses `GPUMetrics::ImplantMahfouzMetric` which takes both a dilated frame and an intensity frame. Intended for implant tracking using intensity-weighted metric. Parameters: `Black_Silhouette` (bool, default true). Forces dilation=3 in `OptimizerManager::Initialize`.

### 5.3 `sym_trap_function`
**Fact** (`src/cost_functions/sym_trap_function.cpp`)

Requires exactly 2 models (principal = tibia, non-principal = femur). Computes standard DIRECT_DILATION score + pole constraint (shortest distance from tibia to femur's sagittal plane axis) × `PoleWeight` + varus/valgus angular cost × `VVWeight`. Parameters: `Dilation` (int, default 3), `PoleWeight` (double, default 75), `VVWeight` (double, default 500).

**Bug:** `create_312_transform` call in `sym_trap_function.cpp:107` passes `p.z_location_` as the first argument instead of `p.x_location_` — x translation will be wrong.

**Constraint (Fact):** `initializesym_trap_function` checks `gpu_non_principal_models_->size() == 1` and returns error if not; model lookup uses hardcoded string `"tibia"` for pose retrieval.

### 5.4 `DD_NEW_POLE_CONSTRAINT`
**Fact** (`src/cost_functions/DD_NEW_POLE_CONSTRAINT.cpp`)

Like `sym_trap_function` but the pole constraint allows selective axis weighting via bool parameters `X_TRANS`, `Y_TRANS`, `Z_TRANS`. Computes X, Y, Z distance components separately. Requires 2 models. Parameters: `Dilation` (int, default 3), `PoleWeight` (double, default 75), `X_TRANS`, `Y_TRANS`, `Z_TRANS` (bools, default false).

**Note:** `min_dist` variable is used without initialization before conditional += operations — undefined behavior if all bools are false.

### 5.5 `DIRECT_DILATION_POLE_CONSTRAINT`
**Fact** (`src/cost_functions/DIRECT_DILATION_POLE_CONSTRAINT.cpp`)

DIRECT_DILATION + single scalar `pole_weight * shortest_distance`. Parameters: `PoleWeight` (double, default 1), `Pole_Weight` (double, default 1, duplicate), `Dilation` (int, default 6). Requires 2 models.

### 5.6 `DIRECT_DILATION_SAME_Z`
**Fact** (`src/cost_functions/DIRECT_DILATION_SAME_Z.cpp`)

DIRECT_DILATION + `Z_Weight * |principal_z - non_principal_z|`. Requires exactly 2 models. Parameters: `Z_Weight` (double, default 1), `Dilation` (int, default 6). Biplane-aware.

### 5.7 `DIRECT_DILATION_T1`
**Fact** (`src/cost_functions/DIRECT_DILATION_T1.cpp`)

Variant of DIRECT_DILATION that uses dilation=1 during Trunk stage (closer to original JTA paper) and the configured dilation during Branch/Leaf stages. Parameters: `Dilation` (int, default 6). Biplane-aware.

### Cost Function Parameter Summary

| Name | Params (key ones) | Models Req'd | Biplane |
|---|---|---|---|
| DIRECT_DILATION | Dilation(int,6) | 1+ | Yes |
| DIRECT_MAHFOUZ | Black_Silhouette(bool,true) | 1+ | Yes |
| sym_trap_function | Dilation(int,3), PoleWeight(dbl,75), VVWeight(dbl,500) | 2 (tibia+femur) | No |
| DD_NEW_POLE_CONSTRAINT | Dilation(int,3), PoleWeight(dbl,75), X/Y/Z_TRANS(bool) | 2 | No |
| DIRECT_DILATION_POLE_CONSTRAINT | PoleWeight(dbl,1), Dilation(int,6) | 2 | No |
| DIRECT_DILATION_SAME_Z | Z_Weight(dbl,1), Dilation(int,6) | 2 | Yes |
| DIRECT_DILATION_T1 | Dilation(int,6) | 1+ | Yes |

---

## 6. Machine Learning Tools

**Fact** (`include/core/machine_learning_tools.h`, `src/core/machine_learning_tools.cpp`)

Single function: `segment_image(cv::Mat orig, bool black_sil, torch::jit::Module* model, uint w, uint h) -> cv::Mat`

Flow:
1. Invert image based on `black_sil_used`
2. Pad to square (larger dimension)
3. Resize to `(input_width, input_height)` using `cv::resize`
4. `cudaMemcpy` to a pre-allocated `torch::Tensor` on GPU (kByte)
5. Cast to float, flip vertical, run `model->forward(inputs)`
6. Threshold at >0, convert to byte, flip back, `cudaMemcpy` to host
7. Resize back to original padded size, crop to original dimensions

**Note:** There are two copies of this file — `include/core/machine_learning_tools.cpp` and `src/core/machine_learning_tools.cpp`. The `include/` version has a slightly different implementation (uses a single `cudaMemcpy` for the output, skipping the intermediate `processed_tensor` variable). Both declare the same function — this is a **duplicate definition risk** if both are compiled.

**Fact:** The model pointer (`torch::jit::Module*`) is a raw pointer; ownership/lifetime is the caller's responsibility.

---

## 7. Smart Pointer Usage vs. Raw Pointers

### Smart Pointers (limited use):

| Location | Type | Usage |
|---|---|---|
| `include/core/model.h:15` | `vtkSmartPointer<vtkSTLReader>` | VTK CAD reader in Model |

### Raw Pointers (extensive use in OptimizerManager):

**Fact** (`include/core/optimizer_manager.h:186–212`):
- `GPUMetrics* gpu_metrics_` — owned, deleted in destructor
- `GPUModel* gpu_principal_model_` — owned, deleted in destructor
- `std::vector<GPUModel*> gpu_non_principal_models_` — owned, elements deleted in destructor
- `std::vector<GPUIntensityFrame*> gpu_intensity_frames_trunk_A_` (×6 similar vectors for trunk/branch/leaf × A/B) — owned, elements deleted in destructor
- `std::vector<GPUEdgeFrame*> gpu_edge_frames_A_/B_` — owned, deleted in destructor
- `std::vector<GPUDilatedFrame*>` (×6) — owned, deleted in destructor
- `std::vector<GPUFrame*> gpu_distance_maps_` — owned, deleted in destructor
- `std::vector<GPUHeatmap*> gpu_heatmaps_` — **NOT deleted in destructor** (omission)

**Fact** (`src/core/direct_data_storage.cpp:18–23`): `DirectDataStorage` allocates `HyperBox6D*` and `std::vector<HyperBox6D*>*` via `new`, relies on `DeleteAllStoredHyperboxes()` for cleanup.

**Fact** (`src/core/optimizer_manager.cpp:1579,1588,1605`): `TrisectPotentiallyOptimal()` allocates three `HyperBox6D*` per iteration via `new`; these are transferred to `DirectDataStorage`.

### Raw Pointers in CostFunctionManager (non-owning):

**Fact** (`include/cost_functions/CostFunctionManager.h:155–175`): All GPU data pointers in `CostFunctionManager` are non-owning borrowed pointers set via `UploadData()`; `OptimizerManager` retains ownership.

---

## 8. Public Interface Boundaries

### jtml_core (static library)
Public headers via `include/core/`:
- `data_structures_6D.h` — `Point6D`, `HyperBox6D`, `Direction` enum
- `direct_data_storage.h` — `DirectDataStorage`
- `location_storage.h` — `LocationStorage`
- `calibration.h` — `Calibration`, `Vect_3`, `Matrix_3_3`
- `frame.h` — `Frame`
- `model.h` — `Model`
- `optimizer_manager.h` — `OptimizerManager` (QObject)
- `optimizer_settings.h` — `OptimizerSettings`
- `metric_enum.h` — `SearchStageFlag`
- `sym_trap_functions.h` — math utilities for symmetry trap
- `machine_learning_tools.h` — `segment_image`
- `ambiguous_pose_processing.h` — `tibial_pose_selector`, `varus_valgus_calculation`
- `curvature_utilities.h` — curvature heatmap generation

### JTA_Cost_Functions (shared library)
**Fact** (`src/cost_functions/CMakeLists.txt`): Built as a `.so`. Public interface via `include/cost_functions/`:
- `CostFunction.h` — `jta_cost_function::CostFunction`
- `CostFunctionManager.h` — `jta_cost_function::CostFunctionManager`
- `Parameter.h` — `jta_cost_function::Parameter<T>`
- `Stage.h` — `Stage` enum class

All exported functions marked `JTML_DLL` (expands to `__declspec(dllexport)` on Windows, empty on Linux).

---

## 9. C4 L2-L3: Subsystems and Modules

```
[Core Layer: jtml_core]
  ├── Pose Representation
  │     ├── Point6D / HyperBox6D    (data_structures_6D)
  │     └── LocationStorage         (location_storage)
  ├── Input Data Management
  │     ├── Frame                   (frame.cu - OpenCV + CUDA)
  │     ├── Model                   (model.cpp - STL/VTK)
  │     └── Calibration             (calibration.h - inline)
  ├── Optimization Engine
  │     ├── OptimizerManager        (optimizer_manager.cpp - Qt thread)
  │     ├── OptimizerSettings       (optimizer_settings.cpp)
  │     └── DirectDataStorage       (direct_data_storage.cpp)
  ├── Geometry & Math Utilities
  │     ├── sym_trap_functions      (sym_trap_functions.cpp)
  │     ├── ambiguous_pose_processing (ambiguous_pose_processing.cpp)
  │     └── curvature_utilities     (curvature_utilities.cpp)
  ├── Machine Learning
  │     └── machine_learning_tools  (LibTorch JIT, CUDA)
  └── I/O
        ├── stl_reader / STLReader  (stl_reader.cpp, STLReader.cpp)
        └── DRRInteractorStyle      (drr_interactor.h - VTK, GUI boundary)

[Cost Functions Layer: JTA_Cost_Functions.so]
  ├── CostFunctionManager           (one per stage: Trunk/Branch/Leaf)
  │     ├── listCostFunctions()     (registration)
  │     ├── callActiveCostFunction()
  │     ├── InitializeActiveCostFunction()
  │     └── DestructActiveCostFunction()
  ├── Cost Function Implementations
  │     ├── DIRECT_DILATION         (primary, edge+dilation+distance map)
  │     ├── DIRECT_MAHFOUZ          (intensity-based)
  │     ├── DIRECT_DILATION_T1      (stage-aware dilation)
  │     ├── DIRECT_DILATION_SAME_Z  (Z-coupling constraint)
  │     ├── DIRECT_DILATION_POLE_CONSTRAINT (bone axis constraint)
  │     ├── DD_NEW_POLE_CONSTRAINT  (axis-selective constraint)
  │     └── sym_trap_function       (full symmetry trap: DD + pole + VV)
  └── Parameter System
        ├── CostFunction            (named parameter container)
        └── Parameter<T>            (double/int/bool typed parameter)
```

---

## 10. Key Invariants / MUST / NEVER Rules

1. **MUST** call `OptimizerManager::Initialize()` and check return value before calling `Optimize()` — `Optimize()` checks `succesfull_initialization_` but emits signals rather than throwing.

2. **MUST** call `CostFunctionManager::UploadData()` before `callActiveCostFunction()` — all GPU pointers start as `0` (null); calling cost functions without upload will segfault.

3. **MUST** call `InitializeActiveCostFunction()` before the first cost function evaluation in each stage, and `DestructActiveCostFunction()` after — these pre-compute the white pixel sum used in all DIRECT_DILATION variants.

4. **MUST** ensure `selected_models[0].row() == primary_model_index` — `Initialize()` checks this and returns false if violated.

5. **MUST NOT** mix `Stage::Trunk/Branch/Leaf` (from `Stage.h`) with `SearchStageFlag::Trunk/Branch/Leaf` (from `metric_enum.h`) — they are separate enums and not interchangeable.

6. **MUST** have exactly 2 models loaded when using `sym_trap_function`, `DD_NEW_POLE_CONSTRAINT`, or `DIRECT_DILATION_POLE_CONSTRAINT` — these cost functions explicitly check and error.

7. **MUST NOT** edit `CostFunctionManager.cpp` or the `DO NOT EDIT` sections — the code comments explicitly warn this is generated/wizard-managed code.

8. **NEVER** call `getActiveCostFunctionClass()` or `getCostFunctionClass()` when the named function does not exist — it returns a heap-allocated `new CostFunction()` that leaks.

9. **MUST** call `DirectDataStorage::DeleteAllStoredHyperboxes()` explicitly — the destructor is commented out.

10. *Inference:* `Frame::setCurvatureHeatmaps()` must be called before `Frame` objects are passed to `OptimizerManager::Initialize()`, otherwise `GetNumCurvatureKeypoints()` returns 0 and the heatmap GPU upload allocates 0-byte data.

---

## 11. Risks and Memory Management Issues

### Risk 1: `gpu_heatmaps_` not freed in destructor
**Fact** (`src/core/optimizer_manager.cpp:1664–1722`): The destructor iterates and deletes all GPU frame and model vectors, but does NOT delete `gpu_heatmaps_` or `gpu_distance_maps_`. This is a memory leak per optimization session.

### Risk 2: `Parameter<double>` stores value as `int`
**Fact** (`include/cost_functions/Parameter.h:73`): `int parameter_value_` instead of `double parameter_value_`. All double parameters (PoleWeight=75, VVWeight=500, Z_Weight, etc.) are silently truncated. The getter returns `double` but the stored value is `int`.

### Risk 3: `Point6D(Pose p)` constructor bug — `ya` never set
**Fact** (`src/core/data_structures_6D.cpp:40`): `this->xa = p.z_angle_` is written twice; `this->ya = p.y_angle_` is never written. Any `Point6D` constructed from a `Pose` object will have incorrect `xa` (z_angle) and `ya=0` always.

### Risk 4: Global mutable variables in `*CustomVariables.h` included inside class definition
**Fact** (`include/cost_functions/CostFunctionManager.h:190–197`): Custom variable headers are `#include`d inside the class body. These headers define global (non-member) variables. This is syntactically legal (they expand to member declarations inside the class when included inside `{}`), but creates confusion and coupling. Variables like `DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_A_` become private class members — which is correct — but the mechanism is fragile and non-obvious.

**Exception:** `sym_trap_functionCustomVariables.h` and `DD_NEW_POLE_CONSTRAINTCustomVariables.h` define `double x_loc_non` and `void invert_transformation(...)`/`void matmult(...)` etc. as file-scope definitions. If these headers are included more than once across translation units, this creates ODR violations and linker errors.

### Risk 5: `DirectDataStorage` destructor commented out
**Fact** (`src/core/direct_data_storage.cpp:45–48`): Explicit memory management required. If `DeleteAllStoredHyperboxes()` is not called (e.g., early return path in `Optimize()`), memory leaks. The early-exit path on `!succesfull_initialization_` does not call this.

### Risk 6: `sym_trap_function.cpp` — wrong x translation passed to `create_312_transform`
**Fact** (`src/cost_functions/sym_trap_function.cpp:107`): `create_312_transform(x2tib, p.z_location_, p.y_location_, p.z_location_, ...)` — `p.z_location_` is used for both the first and third arguments; `p.x_location_` is never used. This is inconsistent with the `create_312_transform` signature which takes `(transform, xt, yt, zt, zr, xr, yr)`.

### Risk 7: `DD_NEW_POLE_CONSTRAINT` — `min_dist` uninitialized
**Fact** (`src/cost_functions/DD_NEW_POLE_CONSTRAINT.cpp:133`): `double min_dist;` — uninitialized; if all `x_tran`, `y_tran`, `z_tran` are false (all default), `min_dist` is uninitialized garbage.

### Risk 8: Duplicate `machine_learning_tools.cpp`
**Fact:** Two files exist: `include/core/machine_learning_tools.cpp` and `src/core/machine_learning_tools.cpp`. Only `src/core/machine_learning_tools.cpp` is listed in `src/core/CMakeLists.txt:23`. The `include/` version has a slightly different implementation. This is confusing and could lead to divergence.

### Risk 9: Biplane distance map upload missing for Camera B
*Inference:* `OptimizerManager::Initialize()` uploads `gpu_distance_maps_` only from `frames_A_`, and there is no `gpu_distance_maps_B_` vector. In biplane mode, the distance map metric in `DIRECT_DILATION` is only applied to Camera A. This may be intentional but is not documented.

### Risk 10: `CalculateSymTrap` hardcodes `iter_val = 60`
**Fact** (`src/core/optimizer_manager.cpp:1348`): `int iter_val = 60; // iter_count * 3;` — the commented-out formula referencing `iter_count` suggests this was previously user-configurable. The `iter_count` parameter is passed to `Initialize()` and stored, but never used.

---

## 12. Files Covered

| File | Role |
|---|---|
| `include/core/data_structures_6D.h` | Point6D, HyperBox6D, Direction |
| `include/core/direct_data_storage.h` | DirectDataStorage |
| `include/core/location_storage.h` | LocationStorage |
| `include/core/frame.h` | Frame |
| `include/core/model.h` | Model |
| `include/core/calibration.h` | Calibration, Vect_3, Matrix_3_3 |
| `include/core/optimizer_manager.h` | OptimizerManager |
| `include/core/optimizer_settings.h` | OptimizerSettings |
| `include/core/machine_learning_tools.h` | segment_image |
| `include/core/sym_trap_functions.h` | Math utilities |
| `include/core/ambiguous_pose_processing.h` | Tibial pose selection |
| `include/core/curvature_utilities.h` | Curvature heatmap generation |
| `include/core/metric_enum.h` | SearchStageFlag |
| `include/core/settings_constants.h` | Default optimizer constants |
| `include/core/preprocessor-defs.h` | JTML_DLL macro |
| `include/core/drr_interactor.h` | DRR VTK interactor (GUI boundary) |
| `include/core/stl_reader.h` | STL reader interface |
| `include/cost_functions/CostFunction.h` | CostFunction |
| `include/cost_functions/CostFunctionManager.h` | CostFunctionManager |
| `include/cost_functions/Parameter.h` | Parameter<T> |
| `include/cost_functions/Stage.h` | Stage enum |
| `include/cost_functions/*CustomVariables.h` | Per-CF state variable injection |
| `src/core/data_structures_6D.cpp` | Point6D, HyperBox6D impl |
| `src/core/direct_data_storage.cpp` | DirectDataStorage impl |
| `src/core/location_storage.cpp` | LocationStorage impl |
| `src/core/frame.cu` | Frame impl (CUDA compilation unit) |
| `src/core/model.cpp` | Model impl |
| `src/core/optimizer_manager.cpp` | OptimizerManager impl (1723 lines) |
| `src/core/optimizer_settings.cpp` | OptimizerSettings defaults |
| `src/core/machine_learning_tools.cpp` | segment_image impl |
| `src/core/sym_trap_functions.cpp` | Symmetry trap math |
| `src/core/ambiguous_pose_processing.cpp` | Tibial pose selector |
| `src/cost_functions/CostFunctionManager.cpp` | CostFunctionManager impl |
| `src/cost_functions/DIRECT_DILATION.cpp` | Primary cost function |
| `src/cost_functions/DIRECT_MAHFOUZ.cpp` | Mahfouz metric |
| `src/cost_functions/DIRECT_DILATION_T1.cpp` | Stage-aware dilation variant |
| `src/cost_functions/DIRECT_DILATION_SAME_Z.cpp` | Z-coupling variant |
| `src/cost_functions/DIRECT_DILATION_POLE_CONSTRAINT.cpp` | Pole constraint variant |
| `src/cost_functions/DD_NEW_POLE_CONSTRAINT.cpp` | Axis-selective pole constraint |
| `src/cost_functions/sym_trap_function.cpp` | Full sym trap cost function |
