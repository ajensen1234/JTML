# TASK-MB-MAP-S-03: GPU/CUDA Layer Code Report

**Stage:** S-03  
**Scope:** GPU/CUDA rendering, metrics, and image processing  
**Paths:** `src/gpu/**`, `include/gpu/**`  
**Date:** 2026-04-18

---

## 1. C4 L2–L3: GPU Subsystem Structure

```
[GPU Subsystem: libjtml_gpu.so]
  namespace gpu_cost_function

  ├─ Data Layer
  │   ├─ GPUImage          (raw grayscale device buffer + pinned bounding box)
  │   ├─ GPUFrame          (owns GPUImage*; base for X-ray frames)
  │   │   ├─ GPUEdgeFrame  (adds Canny params)
  │   │   ├─ GPUDilatedFrame (adds dilation int)
  │   │   └─ GPUIntensityFrame (adds inverted image + dark_silhouette flag)
  │   └─ GPUHeatmap        (stacked keypoint heatmap array on device)
  │
  ├─ Rendering Layer
  │   ├─ RenderEngine      (CUDA rasteriser; owns all device geometry buffers)
  │   └─ GPUModel          (wraps 1 or 2 RenderEngines; mono/biplane)
  │
  ├─ Metrics Layer
  │   └─ GPUMetrics        (all metric kernels; owns pinned score accumulators)
  │       ├─ FastImplantDilationMetric
  │       ├─ ImplantMahfouzMetric
  │       ├─ DistanceMapMetric
  │       ├─ IOU
  │       ├─ L_1_1_MatrixDifferenceNorm
  │       └─ CurvatureHeatmapMetric  [STUB – returns 0]
  │
  ├─ Image Function Layer (free functions)
  │   └─ BlendGrayscaleImages, PasteNonBlackPixels, ScaleGrayscaleToRange,
  │      Convolve, AddUniformNoise, CompileGrid
  │
  └─ Pose/Calibration Layer (host-side helpers)
      ├─ Pose              (6-DOF: xyz location + ZXY Euler angles)
      ├─ RotationMatrix    (3×3 flat float)
      ├─ PoseMatrix        (frame-indexed pose store per model)
      └─ CameraCalibration (pinhole; supports "UF" and "Denver" types)
```

**Decommissioned (fully commented-out):**  
`RegistrationMetric`, `MetricToolbox` — all code is inside block comments in
`registration_metric.cu` and `metric_toolbox.cu`.

---

## 2. GPU Rendering Pipeline

### 2.1 Opaque Silhouette Render (`RenderEngine::Render`)

**Fact** — `src/gpu/render_engine.cu` lines 711–840.

Pipeline stages (all CUDA kernels):

| Step | Kernel | Purpose |
|------|--------|---------|
| 1 | `cudaMemset` | Clear output image to 0 |
| 2 | `ResetKernel` | Initialise bounding-box accumulator |
| 3 | `WorldToPixelKernel` | Rotate+translate vertices (ZXY Euler), perspective-project, snap to integer; compute back-face flag via dot-product |
| 4 | `BoundingBoxForTrianglesKernel` | Per-triangle screen bounding boxes (min/max of projected snapped vertices) |
| 5 | `BoundingBoxSizesKernel` | Count fragments per triangle; write global model bounding box (atomicMin/Max) |
| 6 | `cub::DeviceScan::ExclusiveSum` | Exclusive prefix sum of fragment counts |
| 7 | `PrepareLaunchPacketKernel` | Compute total fragment count |
| 8 | `cudaMemcpy` D→H | Retrieve bounding box and fragment count |
| 9 | Fragment overflow check | `fprintf(stderr,...)` if count > 10M × 255 |
| 10 | `StridePrefixKernel` | Build stride-level prefix table (binary search) |
| 11 | `FillTriangleKernel` | Barycentric point-in-triangle test; write 255 to output pixel |

**Fact** — Euler angle convention is ZXY: `R = Rz·Rx·Ry` (`render_engine.cu:364–374`).

**Fact** — Two camera calibration modes: `"UF"` negates focal length (`fx_ *= -1`); `"Denver"` negates only fy and adjusts cy (`render_engine.cu:93–109`).

**Fact** — Projection uses `sX = (tX/tZ)*fx + cx; sY = (tY/tZ)*fy + cy` (`render_engine.cu:468–469`). Old `dist_over_pix_pitch` path is preserved but no longer used.

**Fact** — Back-face culling: `dotProduct(transformed_normal, transformed_vertex) >= 0` → backface (`render_engine.cu:441–457`). Culled triangles get fragment size = 1 (not 0) to avoid CUB prefix-sum issues (`render_engine.cu:557`).

**Fact** — Maximum stride buffer: `10,000,000` ints (~40 MB device) (`cuda_launch_parameters.h:10`).

### 2.2 DRR Render (`RenderEngine::RenderDRR`)

**Fact** — `src/gpu/render_drr_engine.cu` lines 426–595.

Identical geometry pipeline to Render except:
- `dev_z_line_values_` (float[W×H]) is cleared and filled by `DRR_FillTriangleKernel` with barycentric-interpolated z depth values; front- and back-facing triangles contribute opposite signs.
- Final kernel `ZToLineIntegralToDRRConversionKernel` converts z-integral to a 3D line-integral length, then maps it linearly into [0,255] uchar using `lower_bound`/`upper_bound` parameters.
- Tangent triangles (`dotProduct == 0`) are skipped in depth accumulation.

---

## 3. Frame Types

### 3.1 GPUImage
**Fact** — `include/gpu/gpu_image.cuh`, `src/gpu/gpu_image.cu`.

- Owns `unsigned char* dev_image_` (device) and `int* bounding_box_` (pinned host, 4 ints).
- Three constructors: blank, host-upload, default.
- Destructor calls `cudaFree(dev_image_)` and `cudaFreeHost(bounding_box_)`.
- `GetDeviceImagePointer()` has a subtle bug: if `!image_on_gpu_`, it calls `cudaFree(bounding_box_)` (should be `cudaFreeHost`) before zeroing and returning null (`gpu_image.cu:332–336`).

### 3.2 GPUFrame
**Fact** — `include/gpu/gpu_frame.cuh`, `src/gpu/gpu_frame.cu`.

- Owns `GPUImage* gpu_image_` (raw pointer, deleted in destructor).
- Delegates device pointer access to `gpu_image_`.

### 3.3 GPUEdgeFrame : GPUFrame
**Fact** — Adds `int high_threshold_`, `low_threshold_`, `aperture_` (Canny parameters). No CUDA allocation of its own.

### 3.4 GPUDilatedFrame : GPUFrame
**Fact** — Adds `int dilation_`. No CUDA allocation of its own.

### 3.5 GPUIntensityFrame : GPUFrame
**Fact** — Adds `GPUImage* gpu_inverted_image_` (raw pointer, deleted in destructor) and `bool dark_silhouette_`. `GetWhiteSilhouetteDeviceImagePointer()` returns the inverted image if dark, otherwise the normal image (`gpu_intensity_frame.cu:39–44`).

### 3.6 GPUHeatmap
**Fact** — `include/gpu/gpu_heatmaps.cuh`, `src/gpu/gpu_heatmaps.cu`.

- Owns `unsigned char* dev_heatmap_` (device).
- Allocation is `width * height * num_keypoints * sizeof(uchar)` — a stacked 3D array of probability maps.
- Destructor: `cudaFree(dev_heatmap_)`.

---

## 4. GPU Model Representation

**Fact** — `include/gpu/gpu_model.cuh`, `src/gpu/gpu_model.cu`.

- `GPUModel` owns two raw `RenderEngine*` pointers: `primary_cam_render_engine_` (always used) and `secondary_cam_render_engine_` (biplane only, else nullptr).
- Geometry (triangles, normals) is passed as host `float*` at construction; `RenderEngine::InitializeCUDA` copies to device via `cudaMemcpy`.
- Triangle format: flat 9-float tuples (x1,y1,z1, x2,y2,z2, x3,y3,z3) × N; normals as 3-float tuples × N (`render_engine.cuh:168–177`).
- `biplane_mode_` is set only by the 13-argument constructor.
- `RenderPrimaryCamera_RotationMatrix` allows passing a pre-computed 3×3 rotation matrix directly, bypassing Euler-to-matrix conversion.

### 4.1 RenderEngine Device Buffers

**Fact** — All allocated via `cudaMalloc` in `InitializeCUDA` (`render_engine.cu:251–328`):

| Buffer | Size | Purpose |
|--------|------|---------|
| `dev_triangles_` | 9 × N × float | STL geometry |
| `dev_normals_` | 3 × N × float | STL normals |
| `dev_backface_` | N × bool | Backface flags |
| `dev_projected_triangles_` | 6 × N × float | Screen coords (float) |
| `dev_projected_triangles_snapped_` | 6 × N × int | Screen coords (pixel) |
| `dev_bounding_box_triangles_` | 4 × N × int | Per-triangle bounding boxes |
| `dev_bounding_box_triangles_sizes_` | N × int | Fragment counts |
| `dev_bounding_box_triangles_sizes_prefix_` | N × int | Exclusive prefix sums |
| `dev_bounding_box_` | 4 × int | Global model bounding box |
| `dev_fragment_fill_` | 1 × int | Total fragment count |
| `dev_stride_prefixes_` | 10M × int | Stride lookup table (~40 MB) |
| `dev_z_line_values_` | W × H × float | DRR depth accumulator |
| `dev_transf_vertex_zs_` | 3 × N × float | DRR transformed z per vertex |
| `dev_tangent_triangle_` | N × bool | DRR tangent flags |
| `dev_cub_storage_` | variable | CUB temporary scan storage |
| `renderer_output_` | GPUImage* | Output image (heap, not cudaMalloc) |

`fragment_fill_` (host): `cudaHostAlloc` (pinned).

---

## 5. Registration Metrics

### 5.1 FastImplantDilationMetric (DIRECT-JTA)

**Fact** — `src/gpu/fast_implant_dilation_metric.cu`, called via `GPUMetrics::FastImplantDilationMetric`.

Flow on rendered image (in-place):
1. Edge detect: 8-connected border of white pixels → mark `EDGE_PIXEL (100)`.
2. Dilate: each edge pixel fans out `dilation` in all 4 diagonal directions → mark `DILATED_PIXEL (99)`.
3. Difference: `DILATED_PIXEL || EDGE_PIXEL` pixels: +1 if comparison is black, -1 if comparison is white.
4. Returns `−score` (minimisation favours overlap).

Pixel value sentinel encoding: `WHITE=255, BLACK=0, EDGE=100, DILATED=99`.

### 5.2 ImplantMahfouzMetric

**Fact** — `src/gpu/implant_mahfouz_metric.cu`.

Composite of two sub-scores:
- **Intensity score**: sum of intensity comparison image under the rendered silhouette, normalised by silhouette pixel count; uses `GetWhiteSilhouetteDeviceImagePointer()` to auto-select inverted image.
- **Contour score**: edge-detect + inverse-distance-weighted dilation (dilation = 3 hardcoded), then edge overlap with dilated frame normalised by edge/dilated pixel sum.
- Final: `contour_score × (−2.67) + intensity_score × (−1)`.

**Fact** — `pixel_score_` null-check at line 324 is wrong: `if (pixel_score_ != 0)` checks the pointer, not its value. This means divide-by-zero can silently occur when the denominator is 0. (`implant_mahfouz_metric.cu:324`).

### 5.3 DistanceMapMetric

**Fact** — `src/gpu/distance_map_metric.cu`.

For each edge pixel in projected image: reads the distance-map value and accumulates. Returns `total_distance / (edge_pixel_count + 0.1)`. The `+0.1` avoids singularity when there are no edge pixels.

**Fact** — Bug: thread linearisation uses `(blockIdx.y + gridDim.x + blockIdx.x)` instead of `(blockIdx.y * gridDim.x + blockIdx.x)` (`distance_map_metric.cu:27`). This causes threads to access wrong pixels for non-trivial grid configurations.

### 5.4 IOU (Jaccard Index)

**Fact** — `src/gpu/iou.cu`. Standard |A∩B| / |A∪B| computed over bounding-box union; atomic add to shared device int, then single `cudaMemcpy`.

### 5.5 L_{1,1} Matrix Difference Norm

**Fact** — `src/gpu/l_1_1_matrix_diff_norm.cu`. Per-pixel absolute difference `|A[i] − B[i]|` accumulated atomically. `atomicSub` is used when diff < 0 (subtracts a negative, which adds — this correctly computes absolute value by always adding the magnitude).

### 5.6 CurvatureHeatmapMetric

**Fact** — `src/gpu/curvature_hausdorf_metric.cu`. Stub implementation — launches reset kernel and sets up grid dimensions but the main kernel `CurvatureHausdorfMetric_Kernel` has an incomplete body (computes `kp_loc` but does nothing with it) and the function returns `double test = 0` (`curvature_hausdorf_metric.cu:41–72`).

---

## 6. Pose Matrix Handling

**Fact** — `include/gpu/pose_matrix.h`, `src/gpu/pose_matrix.cpp`.

- `PoseMatrix` stores a `vector<vector<Pose>>` indexed by model index then frame index.
- `AddModel` appends; `GetModelPose(name, frame, ptr)` does linear name search.
- `UpdatePrincipalModelPose` uses `at()` (throws on out-of-bounds), not guarded.

**Fact** — `Pose` is a plain struct: `{x_location_, y_location_, z_location_, x_angle_, y_angle_, z_angle_}` in degrees, passed by value everywhere.

**Fact** — `RotationMatrix` default-constructs to identity (`render_engine.cu:64–74`).

---

## 7. Curvature / Heatmap Generation

**Fact** — `GPUHeatmap` uploads a flat `uchar` array of `width × height × num_keypoints` bytes as a stacked set of 2D probability maps (`gpu_heatmaps.cu:30–46`). Layout is assumed to be C-order (keypoint index is outermost dimension based on `kp_loc` index computation in the stub kernel).

**Fact** — `GPUMetrics::AllocateCurvatureHausdorfScore` must be called separately (after construction) with `num_keypoints` to allocate the per-keypoint score arrays. This is a two-phase initialisation pattern with no guard if skipped (`gpu_metrics.cu:97–111`).

*Inference* — The intended algorithm is a Hausdorff-style minimum-distance metric: for each keypoint's heatmap layer, find the minimum distance from projected edge pixels to the heatmap peak. The kernel is not yet implemented.

---

## 8. CUDA Memory Management Patterns

### 8.1 Pattern Summary

| Pattern | Used In | Notes |
|---------|---------|-------|
| `cudaMalloc` / `cudaFree` | All device buffers | Raw pointers; no smart pointers |
| `cudaHostAlloc` / `cudaFreeHost` | Score accumulators, bounding boxes, fragment_fill | Pinned memory for faster async DMA |
| `cudaMemset` | Image clear before render | Only on render path |
| `cudaMemcpy` D→H | All metric score readbacks | Synchronous |
| `cudaMemcpy` H→D | Image uploads, geometry uploads | Constructor-time |
| `malloc` / `free` | Temporary host image buffers for write-to-file | Not pinned |

### 8.2 Custom Deleters

**Fact** — `include/gpu/cuda_deleters.cuh` (untracked file, newly added):

```cpp
struct CudaFreeDeleter    { void operator()(T* p) { if(p) cudaFree(p); } };
struct CudaFreeHostDeleter{ void operator()(T* p) { if(p) cudaFreeHost(p); } };
```

**Fact** — These deleters are defined but **not yet used anywhere** in `src/gpu/` or `include/gpu/`. All existing code uses raw pointers with manual `cudaFree`/`cudaFreeHost` calls.

### 8.3 Current Smart Pointer State

**Fact** — There are **zero `std::unique_ptr` or `std::shared_ptr` usages** in the GPU source and header files. All CUDA memory (device and pinned host) is managed by raw pointers with manual lifecycle.

**Fact** — All C++ object ownership (GPUImage, RenderEngine) uses raw `new`/`delete` in constructors and destructors.

*Inference* — `cuda_deleters.cuh` was created as preparatory work for a planned smart-pointer conversion. The `smart_pointer_analysis/` and `smart_pointer_conversion.md` files in the repo root (untracked) support this inference.

---

## 9. Key Invariants

1. **Pixel value sentinels** (`pixel_grayscale_colors.h`): `WHITE=255, BLACK=0, EDGE=100, DILATED=99`. Mixing up these values in metric kernels would silently corrupt scores.

2. **Image dimensions must match** for all two-image metrics (IOU, L1_1, Mahfouz). Checked in `BlendGrayscaleImages` but **not enforced** in metric functions — only documented with a warning comment.

3. **Dilation ≥ 1** is required for `FastImplantDilationMetric` and `DilateEdgeDetectedImage`; enforced by clamping (`dilate_edge_detected_image.cu:83`).

4. **Fragment count must not exceed** `maximum_stride_size × (threads_per_block − 1) = 10M × 255 ≈ 2.55B`. Checked with `fprintf(stderr,...)` and returns `cudaErrorMemoryAllocation` but **does not set `initialized_correctly_ = false`**.

5. **`AllocateCurvatureHausdorfScore` must be called before `CurvatureHeatmapMetric`**. Skipping it leaves `dev_curvature_hausdorf_score_` / `curvature_hausdorf_score_` as uninitialized pointers; destructor will call `cudaFree(nullptr)` and `cudaFreeHost(nullptr)` (safe), but `Reset_CurvatureHausdorfScore_Kernel` launch would use a null pointer.

6. **`SetPose` computes rotation matrix** from Euler angles; `SetRotationMatrix` bypasses this. Callers using `RenderPrimaryCamera_RotationMatrix` must maintain their own Euler→matrix conversion.

7. **Thread layout**: kernels use `threads_per_block = 256` globally and square 2D grids (`ceil(sqrt(N/256)) × ceil(sqrt(N/256))`). The `LaunchConfigBuilder` in `launch_config.cuh` computes the same but is not used by any existing production kernel (only defined).

---

## 10. Risks

### High

| # | Risk | Evidence | Severity |
|---|------|----------|----------|
| R1 | **Device memory leak on partial init failure in GPUImage (blank constructor)** | `gpu_image.cu:57`: sets `initialized_correctly_ = true` unconditionally even if `cudaMalloc` failed; `image_on_gpu_` may be false while the object appears healthy | High |
| R2 | **`cudaFree(bounding_box_)` in `GetDeviceImagePointer()`** | Should be `cudaFreeHost` for pinned memory; calling `cudaFree` on pinned memory is undefined behaviour | High |
| R3 | **Distance map kernel indexing bug** | `(blockIdx.y + gridDim.x + blockIdx.x)` at `distance_map_metric.cu:27` — should be `*` not `+`; most threads address wrong pixels silently | High |
| R4 | **`CurvatureHeatmapMetric` is a stub returning 0** | `curvature_hausdorf_metric.cu:71` | High (if relied upon) |
| R5 | **Null pointer check on `pixel_score_` (pointer, not value)** | `implant_mahfouz_metric.cu:324,446`: `if (pixel_score_ != 0)` always true; denominator of 0 → silent integer divide by zero (UB for integer division) | High |

### Medium

| # | Risk | Evidence | Severity |
|---|------|----------|----------|
| R6 | **No cudaError_t propagation in metric functions** | All metric functions declared "THIS FUNCTION DOES NOT HAVE AN ERROR CHECK" in headers; kernel errors are silently ignored | Medium |
| R7 | **Fragment overflow does not mark engine invalid** | `render_engine.cu:809`: returns error but `initialized_correctly_` stays true; subsequent renders may produce garbage | Medium |
| R8 | **`curvature_hausdorf_score_` uninitialised before `AllocateCurvatureHausdorfScore`** | Destructor calls `cudaFreeHost` on uninitialised pointer if `AllocateCurvatureHausdorfScore` was never called | Medium |
| R9 | **`ScaleGrayscaleToRange` allocates `dev_max`/`dev_min` on every call** | `gpu_image_functions.cu:349–353`: temporary `cudaMalloc` on hot path with no error check before use | Medium |
| R10 | **Shared memory edge detection uses `extern __shared__`** | Multiple kernels use dynamic shared memory correctly; however, the shared array size is `blockDim.x * blockDim.y * sizeof(uchar)` which at 16×16=256 bytes is fine, but if `threads_per_block` were changed the kernels would need updating | Medium |

### Low

| # | Risk | Evidence | Severity |
|---|------|----------|----------|
| R11 | **`PoseMatrix::UpdatePrincipalModelPose` uses `at()` without index guard** | Could throw `std::out_of_range` | Low |
| R12 | **`WriteImage` uses raw `malloc`/`free` for host temp buffer** | No error check on `malloc`; if it returns null, `cudaMemcpy` will crash | Low |
| R13 | **`cuda_deleters.cuh` defined but unused** | Preparatory work not yet wired in | Low |
| R14 | **Camera calibration "Denver" mode has hardcoded `pixel_pitch = 0.375`** | `camera_calibration.h:53`: TODO comment present; may give wrong DRR results | Low |

---

## 11. Legacy / Decommissioned Code

**Fact** — Both `registration_metric.cu` and `metric_toolbox.cu` contain their entire implementations as block comments (`//` prefixed). The header files `registration_metric.cuh` and `metric_toolbox.cuh` are also fully commented out. The classes `RegistrationMetric` and `MetricToolbox` do not exist in the compiled binary.

*Inference* — The old architecture required each metric to hold its own device image pointers; the refactored architecture uses `GPUMetrics` as a stateless-ish executor that receives `GPUImage*` arguments, allowing the render engine to own the device images.

---

## 12. File-by-File Evidence Index

| File | Key Facts |
|------|-----------|
| `include/gpu/render_engine.cuh` | `Pose`, `RotationMatrix` structs; full `RenderEngine` private buffer list |
| `src/gpu/render_engine.cu` | Silhouette pipeline; ZXY rotation matrix; camera types |
| `src/gpu/render_drr_engine.cu` | DRR pipeline; z-line-integral accumulation; final conversion kernel |
| `include/gpu/gpu_image.cuh` | `GPUImage` interface |
| `src/gpu/gpu_image.cu` | `cudaFree(bounding_box_)` bug (line 332); blank-ctor init_correctly=true bug (line 57) |
| `include/gpu/gpu_frame.cuh` | Base frame; raw `GPUImage*` |
| `include/gpu/gpu_edge_frame.cuh` | Canny params |
| `include/gpu/gpu_dilated_frame.cuh` | Dilation param |
| `include/gpu/gpu_intensity_frame.cuh` | Inverted image + silhouette flag |
| `src/gpu/gpu_intensity_frame.cu` | `GetWhiteSilhouetteDeviceImagePointer` |
| `include/gpu/gpu_model.cuh` | Mono/biplane constructors; `RenderEngine*` ownership |
| `src/gpu/gpu_model.cu` | Render delegation; DRR delegation |
| `include/gpu/gpu_metrics.cuh` | `GPUMetrics` interface; pinned score buffers |
| `src/gpu/gpu_metrics.cu` | Constructor allocations; destructor frees; `AllocateCurvatureHausdorfScore` |
| `src/gpu/fast_implant_dilation_metric.cu` | DIRECT-JTA edge+dilate+diff pipeline |
| `src/gpu/implant_mahfouz_metric.cu` | Intensity + contour scores; null-ptr check bug (line 324) |
| `src/gpu/distance_map_metric.cu` | Distance map; grid index bug (line 27) |
| `src/gpu/iou.cu` | IOU Jaccard |
| `src/gpu/l_1_1_matrix_diff_norm.cu` | L1,1 norm |
| `src/gpu/curvature_hausdorf_metric.cu` | Stub — returns 0 |
| `src/gpu/edge_detect_rendered_implant_model.cu` | Edge detection (8-connected, shared memory tiling) |
| `src/gpu/dilate_edge_detected_image.cu` | Dilation (4-diagonal directions, 2 passes) |
| `src/gpu/gpu_heatmaps.cu` | Stacked heatmap upload |
| `src/gpu/gpu_image_functions.cu` | Blend, paste, scale, convolve, noise, grid |
| `include/gpu/cuda_deleters.cuh` | `CudaFreeDeleter` / `CudaFreeHostDeleter` — defined, not used |
| `include/gpu/cuda_launch_parameters.h` | `threads_per_block=256`, `maximum_stride_size=10M` |
| `include/gpu/pixel_grayscale_colors.h` | Sentinel values: WHITE=255, BLACK=0, EDGE=100, DILATED=99 |
| `include/gpu/launch_config.cuh` | `LaunchConfigBuilder` — defined, not used by production code |
| `include/gpu/pose_matrix.h` + `src/gpu/pose_matrix.cpp` | Frame-indexed pose store |
| `include/gpu/camera_calibration.h` | UF and Denver pinhole models |
| `src/gpu/metric_toolbox.cu` | Fully commented out (decommissioned) |
| `src/gpu/registration_metric.cu` | Fully commented out (decommissioned) |
