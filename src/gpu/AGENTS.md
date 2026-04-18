# Agent Operating Guide — src/gpu

## OVERVIEW
High-performance CUDA kernels for image processing, 2D/3D registration metrics, and DRR (Digitally Reconstructed Radiograph) rendering. Handles GPU memory management, coordinate transformations, and parallelized cost function evaluations.

## STRUCTURE
- `gpu_image.cu`: Core GPU memory wrapper for image data.
- `gpu_frame.cu`: High-level frame abstraction (intensity, edge, dilated).
- `render_engine.cu`: Rasterization-based DRR engine using barycentric coordinates.
- `gpu_metrics.cu`: Registration cost functions (IOU, Mahfouz, Distance Map).
- `metric_toolbox.cu`: Shared CUDA utilities for metric calculations.

## WHERE TO LOOK
- **Kernels**: Look for `__global__` functions in `.cu` files.
- **Memory**: `InitializeCUDA` and `FreeCuda` methods in engine/frame classes.
- **Math**: `WorldToPixelKernel` in `render_engine.cu` for projection logic.
- **Headers**: `include/gpu/` contains class definitions and `launch_config.cuh`.

## CONVENTIONS (CUDA SPECIFIC)
- **Namespace**: All GPU logic lives in `gpu_cost_function`.
- **Error Handling**: Use `CUDA_CHECK(call)` for API calls and `CUDA_CHECK_KERNEL(launch)` for kernels.
- **Memory**: USE RAII. Prefer `unique_device_ptr<T>` and `unique_host_ptr<T>` from `cuda_deleters.cuh`.
- **Indexing**: Standard 1D/2D grid patterns. Use `threads_per_block` from `launch_config.cuh`.
- **Performance**: Prefer `atomicMin`/`atomicMax` for bounding box updates. Use CUB for prefix sums.
- **Cleanup**: Handled automatically by smart pointers.
