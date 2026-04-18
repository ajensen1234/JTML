# Modernization Strategy — Joint Track Machine Learning (JTML)

## 1. Executive Summary
The JTML codebase requires a transition from a monolithic, legacy-style C++ application to a robust, high-performance medical imaging platform. The strategy avoids a "big-bang" rewrite in favor of a **stabilization-first strangler** approach. This method prioritizes fixing critical correctness bugs and memory leaks before restructuring the architecture. By incrementally extracting logic from the 5.8k-line `MainScreen` God Class into dedicated workers and controllers, we ensure the application remains functional and testable throughout the modernization process.

## 2. Priority Matrix
Modernization efforts follow a strict hierarchy of needs to ensure a stable foundation:

| Priority | Focus Area | Key Actions |
| :--- | :--- | :--- |
| **1. Correctness** | Math & Logic | Fix C1–C7 bugs; implement `CUDA_CHECK` everywhere; guard stubs. |
| **2. Memory** | Stability | Fix M1–M7 leaks; introduce RAII via `cuda_deleters.cuh` and smart pointers. |
| **3. UI/Threading** | Responsiveness | Move Torch inference and heavy GPU work to background workers. |
| **4. Architecture** | Maintainability | Decompose `MainScreen`; extract controllers for session and scene management. |
| **5. Tooling** | Quality Gates | Expand `clang-tidy` coverage; integrate `compute-sanitizer` into CI. |

## 3. Refactoring Patterns (MainScreen Decomposition)
To shrink `MainScreen.cpp`, logic is moved into specialized components using the following patterns:

### Worker Objects (Async Execution)
- **`SegmentationWorker`**: Handles TorchScript loading and inference. Emits signals for results and progress.
- **`ImplantEstimationWorker`**: Manages high-res segmentation, STL processing, and GPU model preparation.
- **`OptimizationWorker`**: Encapsulates the core optimization loop, separating it from UI event handling.

### Controllers (Orchestration)
- **`OptimizationSessionController`**: Manages the lifecycle of an optimization run, including resource allocation and cleanup.
- **`SceneSelectionController`**: Centralizes logic for model visibility, pose synchronization, and camera updates.
- **`SettingsPresenter`**: Decouples the complex settings UI from the underlying parameter models.

### Data Access
- **`CalibrationLoader` / `ImageLoader`**: Extract parsing and validation logic from UI slots into reusable service classes.

## 4. Tooling & CI Integration
Leverage the Pixi-based environment to enforce quality:
- **Static Analysis**: Enable `clang-tidy` checks for `bugprone-*`, `modernize-*`, and `performance-*`.
- **GPU Verification**: Add Pixi tasks for `compute-sanitizer --tool memcheck` to detect CUDA memory errors.
- **Testing**: Re-enable the `test/` directory; convert manual VTK tests into automated smoke tests.

## 5. Risk Mitigation
### Platform & Path Drift
- **Hardcoded Paths**: Replace Windows-style absolute paths (e.g., `C:/JTML/...`) with `std::filesystem` and CMake-injected fixture paths.
- **Toolchain Alignment**: Resolve the discrepancy between the target (CUDA 13.1) and the current Pixi environment (CUDA 12.4) before major architectural changes.
- **Regression Testing**: Implement "tiny" regression tests for every correctness fix to prevent backsliding.

## 6. Multi-Wave Roadmap

### Wave 0: Baseline (Days 1-3)
- Fix hardcoded paths in tests.
- Re-enable the test harness in CMake.
- Align toolchain versions in `pixi.toml`.

### Wave 1: Correctness (Week 1)
- Resolve critical bugs C1–C7.
- Implement mandatory `CUDA_CHECK` and error polling.
- Add regression cases for fixed bugs.

### Wave 2: Memory & RAII (Weeks 2-3)
- Surgical fixes for known leaks (M1–M7).
- Convert manual `cudaFree` calls to RAII deleters.
- Stabilize `OptimizerManager` resource lifecycle.

### Wave 3: Responsiveness (Weeks 4-5)
- Extract `SegmentationWorker` and `ImplantEstimationWorker`.
- Implement non-blocking progress bars and cancellation support.
- Ensure VTK rendering stays on the main thread while math runs in background.

### Wave 4: Architecture (Weeks 6-8)
- Peel `MainScreen` into Controllers.
- Refactor settings management to use a unified presenter pattern.
- Clean up global state and singleton usage.

### Wave 5: Modernization (Ongoing)
- Full `clang-tidy` enforcement.
- Migration to CUDA 13.1 features, deferred until Wave 5.
- Performance profiling and kernel optimization.
