# JTML Codebase Audit
Generated: 2026-04-18

---

## 1. Correctness Bugs (wrong output, UB — fix first)

### C1. `this->za` never set in Point6D constructor
- **File:** `src/core/data_structures_6D.cpp:40`
- **Bug:** `this->xa = p.z_angle_` — `xa` already set on line 38, `za` never assigned
- **Fix:** `this->za = p.z_angle_;`

### C2. `Parameter<double>` stores value as `int` — silent truncation
- **File:** `include/cost_functions/Parameter.h:73`
- **Bug:** `int parameter_value_;` — all float cost params (PoleWeight, VVWeight, etc.) truncated
- **Fix:** Change field to `double parameter_value_;`

### C3. Wrong argument in `sym_trap_function::create_312_transform`
- **File:** `src/cost_functions/sym_trap_function.cpp:107`
- **Bug:** `p.z_location_` passed as x-translation; should be `p.x_location_`
- **Fix:** Replace `p.z_location_` with `p.x_location_` in that argument position

### C4. Uninitialized `min_dist` in `DD_NEW_POLE_CONSTRAINT`
- **File:** `src/cost_functions/DD_NEW_POLE_CONSTRAINT.cpp:134`
- **Bug:** `double min_dist;` then `min_dist +=` — undefined behavior when all axis flags false
- **Fix:** `double min_dist = 0.0;`

### C5. Stage validation condition is always true
- **File:** `src/cost_functions/CostFunctionManager.cpp:46`
- **Bug:** `if (stage_ != Trunk || stage_ != Branch || stage_ != Leaf)` — always true (De Morgan)
- **Fix:** Change `||` to `&&`

### C6. `cudaFree` called on pinned (host) memory
- **File:** `src/gpu/gpu_image.cu:332`
- **Bug:** `bounding_box_` allocated with `cudaHostAlloc` but freed with `cudaFree`
- **Fix:** `cudaFreeHost(bounding_box_)`

### C7. Grid-index arithmetic bug in distance map kernel
- **File:** `src/gpu/distance_map_metric.cu:27`
- **Bug:** `(blockIdx.y + gridDim.x + blockIdx.x) * blockDim.x` — should multiply, not add
- **Fix:** `(blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x`

---

## 2. Memory Leaks / Missing RAII

### M1. `OptimizerManager` destructor skips two resource vectors
- **File:** `src/core/optimizer_manager.cpp:1664–1722`
- **Bug:** Destructor frees most GPU ptrs but omits `gpu_heatmaps_` and `gpu_distance_maps_`
- **Fix:** Add delete loops for both vectors in destructor

### M2. `optimizer_manager` / `optimizer_thread` re-allocated without freeing
- **File:** `src/gui/mainscreen.cpp:4652–4653`
- **Bug:** Each `LaunchOptimizer()` call does `new OptimizerManager()` / `new QThread()` without deleting previous
- **Fix:**
  ```cpp
  if (optimizer_manager) {
      optimizer_thread->quit();
      optimizer_thread->wait();
      delete optimizer_manager;
      delete optimizer_thread;
  }
  optimizer_manager = new OptimizerManager();
  optimizer_thread = new QThread();
  ```

### M3. `DirectDataStorage` destructor commented out
- **File:** `src/core/direct_data_storage.cpp:45–48`
- **Bug:** Destructor body commented out — hyperboxes leak every optimization run
- **Fix:** Uncomment and call `DeleteAllStoredHyperboxes()`

### M4. `CostFunctionManager` destructor is empty
- **File:** `src/cost_functions/CostFunctionManager.cpp:92`
- **Bug:** `~CostFunctionManager() {};` — owns GPU resources, never cleans up
- **Fix:** Add cleanup or convert members to `unique_ptr`

### M5. `CostFunctionManager::getActiveCostFunctionClass()` leaks on not-found
- **File:** `src/cost_functions/CostFunctionManager.cpp:247, 261`
- **Bug:** `return new CostFunction()` on not-found path — caller has no ownership signal, likely leaked
- **Fix:** Return `unique_ptr<CostFunction>` or `nullptr` consistently

### M6. `malloc` without RAII in femoral estimation
- **File:** `src/gui/mainscreen.cpp:1890`
- **Bug:** `malloc(input_width * input_height * ...)` — early-return error paths (lines 1943–1955) don't free it
- **Fix:** `std::unique_ptr<unsigned char[]> host_image(new unsigned char[input_width * input_height]);`

### M7. `cuda_deleters.cuh` exists but is unused everywhere
- **File:** `include/gpu/cuda_deleters.cuh` (untracked)
- **Bug:** Custom RAII deleters were added for device/pinned memory but never wired in; all GPU code still uses raw `cudaMalloc`/`cudaFree`
- **Fix:** Adopt `std::unique_ptr<T, CudaFreeDeleter>` for device ptrs, `CudaFreeHostDeleter` for pinned

---

## 3. Threading / UI Freeze

### T1. PyTorch inference blocks Qt main thread
- **File:** `src/gui/mainscreen.cpp:1745–1835` (`segmentHelperFunction`)
- **Bug:** `torch::jit::load(...)` + CUDA inference loop runs on main thread; freezes UI for all frames
- **Fix:** Move to `QThread` worker, emit progress signals

### T2. CUDA GPU estimation blocks Qt main thread
- **File:** `src/gui/mainscreen.cpp:1847–2208` (`on_actionEstimate_Femoral_Implant_s_triggered`, ~361 lines)
- **Bug:** `torch::jit::load`, `cudaMemcpy`, `GPUModel` creation all on main thread
- **Fix:** Extract GPU work into worker thread; similar pattern needed for tibial equivalent

---

## 4. Unchecked CUDA Error Returns

All of these silently continue after a CUDA failure:

| File | Lines | Call |
|------|-------|------|
| `src/gpu/render_engine.cu` | 781–790 | `cudaMemcpy` (×2) |
| `src/gpu/render_engine.cu` | 852, 880 | `cudaMemcpy` (host_image leaks on error) |
| `src/gpu/render_engine.cu` | 716 | `cudaMemset` |
| `src/gpu/render_engine.cu` | 208–224 | 13× `cudaFree` |
| `src/gpu/gpu_image.cu` | 362–366 | `cudaMemcpy` |
| Kernel launches throughout | various | no `cudaGetLastError()` after launch |

**Recommended fix** — add this macro and wrap all CUDA calls:
```cpp
#define CUDA_CHECK(call) \
    do { cudaError_t _e = (call); \
         if (_e != cudaSuccess) \
             fprintf(stderr, "CUDA error %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(_e)); \
    } while(0)
```

---

## 5. Architecture: God Class

`src/gui/mainscreen.cpp` — **5806 lines**, monolithic view+controller.

Slot methods that need decomposition:

| Method | Approx. Lines | What to extract |
|--------|--------------|-----------------|
| `on_actionEstimate_Femoral_Implant_s_triggered` | ~361 | GPU estimation → worker + helper |
| `on_image_list_widget_itemSelectionChanged` | ~206 | pose loading, model visibility |
| `on_load_calibration_button_clicked` | ~204 | file parsing, camera setup |
| `on_camera_B_radio_button_clicked` | ~168 | pose conversion logic |
| `on_load_image_button_clicked` | ~172 | frame loading + validation |
| `on_model_list_widget_itemSelectionChanged` | ~142 | color/opacity updates |
| `on_load_model_button_clicked` | ~124 | VTK setup, dup-name handling |

`src/gui/settings_control.cpp:607–762` (155 lines) — repetitive trunk/branch/leaf param lookup; extract to factory/lookup.

---

## 6. Modernization / Code Quality

### Q1. `#define` constants → `constexpr`
- **File:** `include/gpu/pixel_grayscale_colors.h:9–12`
- `#define WHITE_PIXEL 255` etc. → `constexpr unsigned char WHITE_PIXEL = 255;`

### Q2. Pass-by-value `std::string` / `std::vector` → `const&`
- `include/cost_functions/CostFunction.h:50` — `setCostFunctionName(std::string)`
- `include/cost_functions/CostFunctionManager.h:53` — `setActiveCostFunction(std::string)`
- `include/gpu/gpu_model.cuh:113` — `SetModelName(std::string)`
- `include/gpu/gpu_frame.cuh:44` — `WriteGPUImage(std::string)`
- `src/core/direct_data_storage.cpp:134` — `DeleteHyperBoxes(std::vector<int>)`
- `include/gui/viewer.h:60,84` — `set_loaded_frames(vector<Frame>&)`, `set_actor_text(string)`

### Q3. Raw matrix arrays → `std::array` or Eigen
- `include/core/sym_trap_functions.h:15–28` — all `float arr[4][4]` args; Eigen is already a dep

### Q4. C-style casts → `static_cast`
- `src/Study2Grid/main.cpp:484–485` — `(int)floor(...)` → `static_cast<int>(floor(...))`

### Q5. `printf` / `fprintf` → logging
- `include/gpu/launch_config.cuh:51–99` — diagnostic prints
- `src/core/sym_trap_functions.cpp:275–282` — pose logging

### Q6. `std::endl` → `'\n'` in loops
- `src/Study2Grid/Study.cpp` (14+ occurrences), `ImageInfo.cpp` (10+)
- `src/core/optimizer_manager.cpp:1340, 1349, 1372`

### Q7. Delete dead commented-out code
- `src/gpu/metric_toolbox.cu:230–279` — 50+ lines of commented CUDA
- `src/gpu/registration_metric.cu:107–119, 160–188, 255–279`

### Q8. `= 0` on pointer members → `= nullptr`
- `src/core/optimizer_manager.cpp:63–64`

### Q9. `CurvatureHausdorffMetric` is a silent stub
- `src/gpu/curvature_hausdorf_metric.cu` — always returns 0; if selectable, will silently produce wrong results without error
- **Fix:** Either implement or throw/assert when selected

---

## Priority Order

1. **C1–C7** — correctness bugs (wrong math output)
2. **M1–M6** — leaks that compound across optimization runs
3. **T1–T2** — UI freeze on any NN/estimation action
4. **CUDA_CHECK macro** — one change catches all unchecked calls
5. **M7** — wire in `cuda_deleters.cuh` as smart pointers convert
6. **Architecture** — `MainScreen` decomposition (larger refactor)
7. **Q1–Q9** — modernization (low risk, do incrementally)
