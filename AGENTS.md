# Agent Operating Guide — Joint Track Machine Learning (JTML)

## OVERVIEW
- **Core Stack**: C++20, CUDA 13.1, Qt5, VTK, Eigen, PyTorch (LibTorch)
- **Build System**: Pixi (conda-based package management) + CMake
- **Primary Goal**: High-performance medical image registration and optimization

## STRUCTURE
- `src/core`: Math, optimizer logic, data structures
- `src/gpu`: CUDA kernels, image processing, distance maps
- `src/gui`: Qt UI components (MainScreen, Viewer)
- `src/cost_functions`: Optimization criteria and parameter management
- `include/`: Header files mirroring `src/` structure
- `test/`: Manual executables and unit tests (currently disabled)

## WHERE TO LOOK
- **Optimization Logic**: `src/core/optimizer_manager.cpp`
- **CUDA Kernels**: `src/gpu/*.cu`
- **UI/Controller**: `src/gui/mainscreen.cpp`
- **Cost Metrics**: `src/cost_functions/`
- **Data Models**: `src/core/data_structures_6D.cpp`

## CODE MAP
- `MainScreen`: Central God Class (5.8k lines). Handles UI, VTK rendering, and orchestration.
- `OptimizerManager`: Manages optimization loops, GPU resource allocation, and thread sync.
- `CostFunctionManager`: Factory for cost metrics; manages active criteria and weights.
- `GPUModel` / `GPUFrame`: GPU-side representations of 3D models and 2D x-ray frames.
- `DirectDataStorage`: Storage for optimization hyperboxes and results.

## CONVENTIONS
- **Language**: C++20 (use `std::span`, `concepts`, `ranges` where appropriate).
- **CUDA**: Target CUDA 13.1. Use `cuda_deleters.cuh` for RAII.
- **Formatting**: `pixi run format` (clang-format).
- **Linting**: `pixi run tidy` (clang-tidy).
- **Memory**: Prefer `std::unique_ptr` and `std::shared_ptr`. Avoid raw `new`/`delete`.

## ANTI-PATTERNS (See .plan/audit.md)
- **Monolithic GUI**: Do not add logic to `MainScreen`. Extract to workers or helpers.
- **Raw CUDA**: Avoid raw `cudaMalloc`/`cudaFree`. Use `CudaFreeDeleter` with smart pointers.
- **Blocking UI**: Never run PyTorch inference or heavy GPU work on the Qt main thread.
- **Unchecked CUDA**: Wrap all CUDA calls in `CUDA_CHECK` macro.
- **Silent Truncation**: Ensure `Parameter<double>` doesn't store values as `int`.

## COMMANDS
- `pixi run build`: Compile project
- `pixi run configure`: Run CMake configuration
- `pixi run format`: Format code with clang-format
- `pixi run tidy`: Run static analysis with clang-tidy
- `pixi run run`: Execute the main application

## QUALITY GATES
- `pixi run build` must pass without errors.
- `pixi run tidy` must not introduce new warnings.
- All CUDA, Qt, and VTK targets must be verified before merge.

## MEMORY BANK
- Durable knowledge: `.memory-bank/`
- Operational logs: `.tasks/`
- Long-running plans: `.protocols/`
- Entry points: `/cold-start`, `/mb`, `/prd`, `/execute`, `/verify`
