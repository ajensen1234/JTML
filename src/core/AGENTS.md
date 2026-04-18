# Agent Operating Guide — JTML Core

## OVERVIEW
Optimization orchestration, data persistence, and 6D spatial structures. Bridges high-level UI requests to low-level GPU kernels.

## STRUCTURE
- `optimizer_manager.cpp`: Orchestrates optimization threads and state.
- `direct_data_storage.cpp`: Implements DIRECT optimization storage (HyperBox6D).
- `data_structures/`: Core POD and utility types.
- `optimization/`: Specific algorithm implementations.
- `frame.cu`: GPU-accelerated frame processing.

## WHERE TO LOOK
- **Optimization Flow**: Start in `OptimizerManager::Initialize` and `Optimize()`.
- **Data Persistence**: Check `DirectDataStorage` for how search space is partitioned.
- **GPU Integration**: See `frame.cu` and `gpu_model.cuh` includes.
- **Geometry**: `data_structures_6D.cpp` for spatial primitives.

## CONVENTIONS
- **Threading**: `OptimizerManager` runs in a dedicated `QThread`. Avoid blocking calls.
- **Memory**: Manual `new`/`delete` used in `DirectDataStorage`. Use `DeleteAllStoredHyperboxes` for cleanup.
- **GPU**: CUDA kernels invoked via `.cu` files. Keep host-side logic in `.cpp`.
- **Error Handling**: Use `error_message` strings and boolean return flags for initialization.
