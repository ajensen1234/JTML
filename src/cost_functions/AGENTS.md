# Agent Operating Guide — Cost Functions

## OVERVIEW
- **Domain**: Optimization metrics for 2D/3D medical image registration.
- **Core Logic**: GPU-accelerated cost calculations (Dilation, Distance Maps, Pole Constraints).
- **Integration**: Managed by `CostFunctionManager` via method injection (partial class pattern).

## STRUCTURE
- `CostFunctionManager.cpp`: Central registry and data uploader. DO NOT edit the `listCostFunctions` section manually unless adding a new metric.
- `DIRECT_DILATION.cpp`: Standard edge-based registration metric.
- `DD_NEW_POLE_CONSTRAINT.cpp`: Metric with geometric constraints (Poles).
- `DIRECT_MAHFOUZ.cpp`: Silhouette-based registration.
- `sym_trap_function.cpp`: Symmetric trapezoidal error function.

## WHERE TO LOOK
- **Metric Implementation**: Look for `costFunction[NAME]()` in the corresponding `.cpp` file.
- **Initialization**: `initialize[NAME]()` handles pre-optimization GPU setup (e.g., white pixel sums).
- **Parameters**: Defined in `CostFunctionManager::listCostFunctions()`.
- **GPU Interaction**: Calls `gpu_metrics_` (GPUMetrics) for heavy lifting (FastImplantDilationMetric, DistanceMapMetric).

## CONVENTIONS
- **Method Naming**: `initialize[NAME]`, `destruct[NAME]`, `costFunction[NAME]`.
- **Data Access**: Use `gpu_principal_model_`, `gpu_dilated_frames_A_`, and `gpu_distance_maps_`.
- **Biplane Support**: Always check `biplane_mode_` and handle `Camera B` / `SecondaryCamera`.
- **Error Handling**: Return `false` from init/destruct and set `error_message`.

## ANTI-PATTERNS
- **Manual Registry**: Avoid editing `CostFunctionManager.cpp` outside the designated "WIZARD" blocks.
- **CPU Bottlenecks**: Do not perform heavy image processing on the CPU inside the cost function loop.
- **Unchecked CUDA**: Ensure `cudaStatus` is checked during initialization.
- **State Leakage**: Always clean up temporary GPU allocations in `destruct[NAME]`.
