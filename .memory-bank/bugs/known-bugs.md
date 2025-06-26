---
description: Known bugs found during codebase mapping (2026-04-18). Severity rated High/Medium/Low.
status: active
---
# Known Bugs

Discovered during brownfield mapping (2026-04-18). Evidence references subagent reports in `.tasks/TASK-MB-MAP/`.

## High Severity

### BUG-001: `cudaFree` called on pinned host memory
- **File**: `src/gpu/gpu_image.cu` — `GPUImage::GetDeviceImagePointer`
- **Problem**: `cudaFree` called on memory allocated with `cudaMallocHost` (pinned). Should be `cudaFreeHost`. Results in CUDA error / undefined behavior.
- **Source**: TASK-MB-MAP-S-03

### BUG-002: Grid-index arithmetic typo in `DistanceMapMetric_Kernel`
- **File**: `src/gpu/distance_map_metric.cu`
- **Problem**: Multiply/add expression in grid index computation is wrong, silently processing incorrect pixel coordinates.
- **Source**: TASK-MB-MAP-S-03

### BUG-003: Integer divide-by-zero in `ImplantMahfouzMetric`
- **File**: `src/gpu/implant_mahfouz_metric.cu`
- **Problem**: Null-pointer check uses value comparison instead of pointer check, allowing integer divide-by-zero when denominator is 0.
- **Source**: TASK-MB-MAP-S-03

### BUG-004: `Point6D(Pose)` constructor assigns `xa` twice, `ya` never set
- **File**: `src/core/data_structures_6D.cpp:40`
- **Problem**: `ya` (y-angle) is never assigned; `xa` is assigned the value intended for `ya`. Corrupts all pose conversions.
- **Source**: TASK-MB-MAP-S-02

### BUG-005: `Parameter<double>` stores value as `int` — silent truncation
- **File**: `include/cost_functions/Parameter.h:73`
- **Problem**: Template parameter stores value as `int` regardless of template type. All float cost function parameters (PoleWeight=75, VVWeight=500) silently truncated.
- **Source**: TASK-MB-MAP-S-02

### BUG-006: `sym_trap_function` passes wrong argument
- **File**: `src/cost_functions/sym_trap_function.cpp:107`
- **Problem**: Passes `p.z_location_` as x-translation argument to `create_312_transform` instead of `p.x_location_`.
- **Source**: TASK-MB-MAP-S-02

## Medium Severity

### BUG-007: `OptimizerManager` leaks `gpu_heatmaps_` and `gpu_distance_maps_`
- **File**: `src/core/optimizer_manager.cpp` — destructor
- **Problem**: ~20+ GPU resource raw pointers are freed in destructor, but `gpu_heatmaps_` and `gpu_distance_maps_` vectors are omitted — never freed.
- **Source**: TASK-MB-MAP-S-02

### BUG-008: `optimizer_manager*` / `optimizer_thread*` re-allocated without cleanup
- **File**: `src/gui/mainscreen.cpp`
- **Problem**: On each optimization run, new pointers allocated but old ones not freed first. Memory leak + thread safety risk.
- **Source**: TASK-MB-MAP-S-04

### BUG-009: `CostFunctionManager::getActiveCostFunctionClass()` leaks on not-found path
- **File**: `src/cost_functions/CostFunctionManager.cpp`
- **Problem**: `new CostFunction()` allocated on the not-found code path and never freed.
- **Source**: TASK-MB-MAP-S-02

### BUG-010: `DD_NEW_POLE_CONSTRAINT` uninitialized `min_dist`
- **File**: `src/cost_functions/DD_NEW_POLE_CONSTRAINT.cpp`
- **Problem**: `min_dist` variable uninitialized; UB when all axis flags are false.
- **Source**: TASK-MB-MAP-S-02

### BUG-011: `orientation = new float[3]` never freed
- **File**: `src/gui/mainscreen.cpp`
- **Problem**: Heap array allocated, no corresponding delete[]. Memory leak.
- **Source**: TASK-MB-MAP-S-04

### BUG-012: `QGraphicsView` double-free in `Controls`
- **File**: `src/gui/controls.cpp`
- **Problem**: `QGraphicsView` created with Qt parent (auto-managed) AND manually deleted — likely double-free.
- **Source**: TASK-MB-MAP-S-04

### BUG-013: `DirectDataStorage` destructor commented out
- **File**: `src/core/direct_data_storage.cpp`
- **Problem**: Destructor body commented out; requires explicit `DeleteAllStoredHyperboxes()` call or resources leak.
- **Source**: TASK-MB-MAP-S-02

## Low Severity / Quality

### BUG-014: `CurvatureHausdorffMetric` is a stub (always returns 0)
- **File**: `src/gpu/curvature_hausdorf_metric.cu`
- **Problem**: Metric is referenced but always returns zero — silently produces wrong results if selected.
- **Source**: TASK-MB-MAP-S-03

### BUG-015: Duplicate `machine_learning_tools.cpp`
- **File**: `include/core/machine_learning_tools.cpp` and `src/core/machine_learning_tools.cpp`
- **Problem**: Two slightly different implementations. One may be stale/wrong.
- **Source**: TASK-MB-MAP-S-02

### BUG-016: PyTorch NN inference blocks GUI thread
- **File**: `src/gui/mainscreen.cpp`
- **Problem**: TorchScript model runs synchronously on the Qt main thread, freezing UI.
- **Source**: TASK-MB-MAP-S-04
