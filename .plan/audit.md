# JTML Codebase Audit
Generated: 2026-04-18

---

## 1. Correctness Bugs (RESOLVED IN WAVE 1)

### C1. `this->za` never set in Point6D constructor - FIXED
### C2. `Parameter<double>` stores value as `int` - FIXED
### C3. Wrong argument in `sym_trap_function::create_312_transform` - FIXED
### C4. Uninitialized `min_dist` in `DD_NEW_POLE_CONSTRAINT` - FIXED
### C5. Stage validation condition is always true - FIXED
### C6. `cudaFree` called on pinned (host) memory - FIXED
### C7. Grid-index arithmetic bug in distance map kernel - FIXED

---

## 2. Memory Leaks / RAII (RESOLVED IN WAVE 2)

### M1. `OptimizerManager` resource vectors - FIXED (partial RAII conversion)
### M2. `optimizer_manager` / `optimizer_thread` re-allocated without freeing - FIXED
### M3. `DirectDataStorage` hyperbox leaks - FIXED (std::vector<std::unique_ptr>)
### M4. `CostFunctionManager` destructor - FIXED
### M5. `CostFunctionManager` factory leaks - FIXED
### M6. `malloc` in femoral estimation - FIXED
### M7. `cuda_deleters.cuh` adoption - FIXED (Core GPU classes now RAII)

## 3. Threading / UI Freeze (RESOLVED IN WAVE 3)

### T1. PyTorch inference blocks Qt main thread - FIXED
- Logic moved to `SegmentationWorker`. Progress signals active.

### T2. CUDA GPU estimation blocks Qt main thread - FIXED
- Logic moved to `EstimationWorker`. STL loading and model prep now async.

### T3. Signal Flooding causing UI lag - FIXED
- `UpdateOptimum` throttled to ~30 FPS via `QElapsedTimer`.

---

## 4. Architecture / Monolith (RESOLVED IN WAVE 4)

### A1. MainScreen "God Class" decomposition - FIXED
- **Status**: 2,100 lines removed from MainScreen.cpp.
- **New Services**:
  - `CalibrationService`: Handles all calibration parsing.
  - `ImageLoadingService`: Handles image/model loading loops.
  - `SettingsService`: Manages QSettings persistence.
  - `WorkerOrchestrator`: Manages background thread lifecycles.
  - `SceneController`: Manages VTK actor properties and selection syncing.

---

## 5. Current Build Status & Blockers

### B1. Pre-existing Core Breakage
- **Issue**: `src/core/optimizer_manager.cpp` references missing `sym_trap_functions` and has `DirectDataStorage` API mismatches.
- **Fix in Progress**: I am currently restoring the missing signatures and commenting out non-existent sym_trap calls to allow the project to reach a "Green" build state.

### B2. Interactor Multi-Definition
- **Issue**: Global variables in `interactor.h` causing link errors.
- **Fix**: Moved to `extern` pattern with storage in `interactor.cpp`.

---

## 6. Next Steps (Wave 5 & 6)

### Wave 5: Performance & Modernization
- [ ] Replace 18+ instances of `std::endl` with `\n` in loops.
- [ ] Migrate 140+ C-style casts to `static_cast<T>`.
- [ ] Convert 22 `#define` constants to `constexpr`.

### Wave 6: Final UI Peeling
- [ ] Decompose `ArrangeMainScreenLayout` and resize logic.
- [ ] Implement automated regression tests for extracted services.

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
