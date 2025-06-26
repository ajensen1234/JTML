---
description: System architecture overview (C4 L1–L3) — subsystems, data flow, key invariants.
status: active
---
# Architecture Overview

See also: [guides/build.md](../guides/build.md) for HOW to build each layer.

## C4 L1 — System context
JTML is a standalone desktop application. External inputs: fluoroscopy TIFF/image files, STL implant models, calibration .jts files, kinematic .jts/.jtak files.

## C4 L2 — Subsystems

```
┌─────────────────────────────────────────────────────┐
│                  JTML Desktop App                    │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────┐  │
│  │  Qt GUI  │──▶│  Core    │──▶│   GPU / CUDA    │  │
│  │ MainScrn │   │ Optim.   │   │ Render + Metrics│  │
│  │ Viewer   │   │ CostFns  │   │ libjtml_gpu.so  │  │
│  └──────────┘   └──────────┘   └─────────────────┘  │
│       │              │                               │
│       ▼              ▼                               │
│  ┌──────────┐   ┌──────────┐                         │
│  │ VTK 9.3  │   │ PyTorch  │                         │
│  │ Rendering│   │ NN Pose  │                         │
│  └──────────┘   └──────────┘                         │
└─────────────────────────────────────────────────────┘
```

## C4 L3 — Module breakdown

### GUI layer (`src/gui/`, Qt5)
- `MainScreen` (5806 lines) — monolithic view+controller; owns optimizer_manager and optimizer_thread
- `Viewer` — VTK dual-renderer wrapper (background fluoroscopy + 3D implant scene)
- `Controls` — UI controls dialog
- `DRR Tool` — modal dialog for DRR rendering via CUDA
- `Settings Control` — QSettings persistence
- `Interactor` / `About` — misc

### Core layer (`src/core/`, `include/core/`)
- `Frame` — fluoroscopy image frame (host-side)
- `Model` — 3D implant model (STL data)
- `OptimizerManager` — owns all GPU resources, drives DIRECT optimizer across stages (Trunk→Branch→Leaf)
- `OptimizerSettings` — 6D bounds, stage config
- `DirectDataStorage` — hyperbox storage for DIRECT algorithm
- `LocationStorage` — pose location tracking
- `DataStructures6D` — Point6D, Joint6D, Pose types
- `CurvatureUtilities` — curvature computation helpers
- `AmbiguousPoseProcessing` — handles ambiguous pose cases
- `MachineLearningTools` — PyTorch/TorchScript NN inference

### Cost functions layer (`src/cost_functions/`, `include/cost_functions/`)
- `CostFunctionManager` — selects and instantiates active cost function
- `CostFunction` / `Stage` / `Parameter` — base types
- Implementations: `DIRECT_DILATION`, `DIRECT_DILATION_T1`, `DIRECT_DILATION_SAME_Z`, `DIRECT_DILATION_POLE_CONSTRAINT`, `DD_NEW_POLE_CONSTRAINT`, `DIRECT_MAHFOUZ`, `sym_trap_function`

### GPU layer (`src/gpu/`, `include/gpu/`, `libjtml_gpu.so`)
- `RenderEngine` — CUDA triangle rasterizer (ZXY Euler / rotation matrix → silhouette or DRR)
- `GPUImage` / `GPUFrame` — device image management
- `GPUEdgeFrame` / `GPUDilatedFrame` / `GPUIntensityFrame` — derived frame types
- `GPUModel` — device-side STL model
- `PoseMatrix` — 4×4 SE(3) pose
- Metrics: `FastImplantDilationMetric`, `ImplantMahfouzMetric`, `DistanceMapMetric`, `IOU`, `L1_1MatrixDiffNorm`
- `CurvatureHausdorffMetric` — **stub, always returns 0**
- `GPUHeatmaps` — curvature/metric heatmap generation

### Auxiliary modules
- `shape_sensitivity` — standalone research binary; sweeps bone models through rotation space, outputs IARTD/Hu CSVs
- `Study2Grid` — data pipeline converting fluoroscopy study dirs to 1024×1024 TIFF+label grids for ML training (marked "old way" in source)

## Key invariants
- MUST: Build Release only (`-DCMAKE_BUILD_TYPE=Release`)
- MUST: All GPU device pointers use raw `cudaMalloc`/`cudaFree` (smart pointer CUDA support in progress via `cuda_deleters.cuh`)
- MUST NOT: Convert Qt widget pointers with parent-child relationships to smart pointers
- MUST: VTK built from source before CMake configure

## Data formats
- Calibration: `.jts` files (5-line text, `JT_INTCALIB` type)
- Kinematics: `.jts` (tab/comma-delimited 6-DOF pose) or `.jtak` (biplane)
- Images: TIFF fluoroscopy frames
- Models: STL binary/ASCII
