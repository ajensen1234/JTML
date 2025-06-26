# TASK-MB-MAP-S-01: Build & Tooling Layer Report

**Stage:** S-01 — Build system, tooling, CI, environment setup
**Scope:** CMakeLists.txt, pixi.toml, justfile, *.sh, Doxyfile
**Date:** 2026-04-18

---

## C4 Level 1: What Is This System?

**Joint Track Machine Learning (JTML)** is a GPU-accelerated desktop application for autonomously extracting 6-degree-of-freedom (6D) joint kinematics from fluoroscopic (X-ray video) images. It is developed at the Gary J. Miller Orthopaedic Biomechanics Lab (University of Florida). The system renders synthetic DRR (Digitally Reconstructed Radiograph) images of implant models, computes image registration metrics on the GPU, and uses PyTorch-based neural networks to optimize implant pose estimates. The result is a Qt5 GUI application backed by CUDA kernels, libtorch inference, VTK rendering, and OpenCV image processing.

---

## 1. Build System

### CMake

- **Minimum required version:** 3.26 — **Fact** (`CMakeLists.txt:30`)
- **Language standards:** C++20 and CUDA 20 — **Fact** (`CMakeLists.txt:35-38`)
- **Generator (preferred):** Ninja — **Fact** (`pixi.toml:59`, `vtk_installer.sh:81`)
- **Build type:** Release (configured via pixi task) — **Fact** (`pixi.toml:63`)
- **Build output directory:** `.build/` (configured in pixi task) — **Fact** (`pixi.toml:61`)
- **Binary output:** `.build/bin/` — **Fact** (`CMakeLists.txt:108`)
- **Library output:** `.build/lib/` — **Fact** (`CMakeLists.txt:109-110`)
- **Compile commands export:** `CMAKE_EXPORT_COMPILE_COMMANDS=ON` — **Fact** (`pixi.toml:75`)
- **Compile commands symlink:** `compile_commands.json -> ./.build/compile_commands.json` — **Fact** (repo root)
- **Packaging:** CPack configured for DEB (Linux) and NSIS (Windows) — **Fact** (`packaging/CMakeLists.txt`)
- **Testing:** Disabled — `enable_testing()` and `add_subdirectory(test)` are commented out — **Fact** (`CMakeLists.txt:128-130`)

### CUDA Architecture

- **Default architecture:** `native` (detects host GPU at build time) — **Fact** (`CMakeLists.txt:56`, `src/gui/CMakeLists.txt:42`)
- **Separable compilation:** Enabled for `jtml_gpu` — **Fact** (`src/gpu/CMakeLists.txt:36`)

### RPATH Strategy

- Build and install RPATHs set to include `$CONDA_PREFIX/lib` and VTK lib dir — **Fact** (`src/gui/CMakeLists.txt:45-48`, `src/gpu/CMakeLists.txt:38-39`)
- System CUDA paths (`/usr/local/cuda`, `/usr/lib64`) explicitly ignored to prevent conda/system conflicts — **Fact** (`CMakeLists.txt:13-19`)

### Qt Auto-tools

- `CMAKE_AUTOMOC`, `CMAKE_AUTORCC`, `CMAKE_AUTOUIC` all `ON` at root and in `core`, `gui`, `cost_functions` — **Fact** (multiple CMakeLists.txt)

---

## 2. Package Manager (pixi / conda)

- **Tool:** pixi (conda-based) — **Fact** (`pixi.toml:1`)
- **Channels:** `https://prefix.dev/conda-forge`, `https://prefix.dev/nvidia` — **Fact** (`pixi.toml:6`)
- **Platform lock:** `linux-64` only — **Fact** (`pixi.toml:7`)
- **Lock file version:** 6 — **Fact** (`pixi.lock:1`)

### Environments

| Environment | Features included |
|---|---|
| `build` | `[feature.build]` — cmake, ninja, gcc, cxx-compiler, pkg-config, cuda 12.4 |
| `dev` | `[feature.build]` + `[feature.dev]` (dev deps currently empty) |

- **Fact** (`pixi.toml:111-113`)

### Activation Environment Variables (set on `pixi shell` / task run)

| Variable | Value |
|---|---|
| `Qt5_DIR` | `$CONDA_PREFIX/lib/cmake/Qt5` |
| `VTK_DIR` | `$CONDA_PREFIX/lib/cmake/vtk-9.3` |
| `Torch_DIR` | `$CONDA_PREFIX/lib/cmake/Torch` |
| `OpenCV_DIR` | `$CONDA_PREFIX/lib/cmake/opencv4` |
| `CUDA_HOME` | `$CONDA_PREFIX` |
| `CUDA_TOOLKIT_ROOT_DIR` | `$CONDA_PREFIX` |
| `CUDACXX` | `$CONDA_PREFIX/bin/nvcc` |
| `QT_QPA_PLATFORM` | `xcb;wayland` |
| `GDK_BACKEND` | `x11;wayland` |

- **Fact** (`pixi.toml:115-124`)

---

## 3. External Library Dependencies (with versions)

All versions come from `pixi.lock` (locked state) unless noted.

| Library | Version (locked) | Source |
|---|---|---|
| CUDA toolkit | 12.4.0 | conda-forge (pixi.lock:29) |
| CUDA runtime (cudart) | 12.4.99 | conda-forge (pixi.lock:36) |
| cuDNN | **9.3.0.75** | conda-forge (pixi.lock:77) |
| NVTX (nvtx) | (latest from conda-forge) | pixi.toml:18 |
| PyTorch (pytorch-gpu) | **2.4.1** (cuda120 build) | conda-forge (pixi.lock) |
| libtorch | **2.4.1** (cuda120 build) | conda-forge (pixi.lock) |
| OpenCV | **4.10.0** (qt5 build, py312) | conda-forge (pixi.lock) |
| Qt | **5.15.8** | conda-forge (pixi.lock) |
| Qt-wayland | 5.15.8 | conda-forge (pixi.lock) |
| Eigen3 | **3.4.0** | conda-forge (pixi.lock) |
| VTK | **9.3.0** (built from source) | vtk_installer.sh:9 |
| CMake | **3.31.4** | conda-forge (pixi.lock:27) |
| Ninja | (latest from conda-forge) | pixi.toml:17 |
| GCC / cxx-compiler | (conda-forge latest) | pixi.toml:30,43-44 |
| clang-tools (format+tidy) | **19.1.2** | conda-forge (pixi.lock:26) |
| ccache | 4.10.1 | conda-forge (pixi.lock:23) |
| SDL2 | >=2.30.7,<3 | pixi.toml:31 |
| GLEW | >=2.1.0,<3 | pixi.toml:33 |
| libcurl | >=8.11.1,<9 | pixi.toml:25 |

### VTK Build Notes

VTK is **not** installed via conda; it is built from source by `vtk_installer.sh`:
- Clones `https://github.com/Kitware/VTK.git` and checks out tag `v9.3.0`
- Installs to `_deps/vtk/install/` (relative to repo root)
- Configured with Qt5 support, SDL2 windowing (no X11 or EGL), Ninja generator
- CMake detects lib path: checks `_deps/vtk/install/lib64` then falls back to `_deps/vtk/install/lib` — **Fact** (`CMakeLists.txt:83-91`)

---

## 4. CMake Build Targets

| Target | Type | Source directory |
|---|---|---|
| `joint-track-machine-learning` | Executable (Qt5 GUI) | `src/gui/` |
| `jtml_core` | Static library | `src/core/` |
| `jtml_gpu` | Shared library (CUDA) | `src/gpu/` |
| `JTA_Cost_Functions` | Shared library | `src/cost_functions/` |
| `Study2Grid-Cmake` | Executable | `src/Study2Grid/` |

- `src/shape_sensitivity` is commented out — *Inference*: not yet integrated or deprecated
- `test/` subdirectory is commented out — **Fact** (`CMakeLists.txt:128`)

### Target Dependency Graph

```
joint-track-machine-learning (exe)
  ├── jtml_core (static)
  │     ├── jtml_gpu (shared)
  │     ├── JTA_Cost_Functions (shared)
  │     ├── Qt5::Core/Gui/Widgets
  │     ├── TORCH_LIBRARIES
  │     ├── VTK_LIBRARIES
  │     └── OpenCV
  ├── jtml_gpu (shared)
  │     ├── CUDA::cublas
  │     ├── CUDA::cudart
  │     ├── CUDA::curand
  │     └── OpenCV
  └── JTA_Cost_Functions (shared)
        ├── jtml_gpu
        ├── TORCH_LIBRARIES
        ├── CUDA::cudart / cublas
        └── Qt5, OpenCV
```

---

## 5. Available Tasks / Commands

### pixi tasks

| Task | Command | Depends on | Description |
|---|---|---|---|
| `pixi run build-vtk` | `./vtk_installer.sh` | — | Clone and build VTK 9.3.0 from source into `_deps/vtk/` |
| `pixi run configure` | `cmake -GNinja -S. -B.build ...` | `build-vtk` | Configure CMake with Ninja, Release mode, all library paths |
| `pixi run build` | `cmake --build .build` | `configure`, `build-vtk` | Compile all targets |
| `pixi run run` | `.build/bin/joint-track-machine-learning` | `build` | Launch the GUI application |
| `pixi run format` | `bash format.sh` | `configure` | Run clang-format on all src/include files |
| `pixi run tidy` | `bash tidy.sh --path .build` | `configure` | Run clang-tidy static analysis |
| `pixi run check-display` | echo env vars | — | Diagnose display server environment |

- **Fact** (`pixi.toml:54-110`)

### justfile commands

| Command | Description |
|---|---|
| `just` | List all available commands |
| `just format [ARGS]` | Run `format.sh` (wraps clang-format) |
| `just tidy [ARGS]` | Run `tidy.sh` (wraps clang-tidy) |
| `just all` | format then tidy |
| `just format-check` | Dry-run format (show diff, no changes) |
| `just tidy-fix` | Run tidy with `--fix` flag |
| `just fix` | format + tidy-fix |

- **Fact** (`justfile:1-26`)

### format.sh

- Finds all `*.cpp`, `*.hpp`, `*.h`, `*.cu`, `*.cuh`, `*.cc`, `*.cxx` under `./src` and `./include`
- Excludes: `build/`, `cmake-build*/`, `_deps/`, `.pixi/`, `*_autogen/`
- Runs `clang-format -i` in-place
- Supports `--dry-run` / `-d` and `--verbose` / `-v` flags
- **Fact** (`format.sh:44-81`)

### tidy.sh

- Same file discovery as format.sh (excludes `.cu`/`.cuh` — **Fact** `tidy.sh:90-101`)
- Requires `compile_commands.json` in build path (default: `.build`)
- Runs `clang-tidy` in parallel with `xargs -P $(nproc)`
- Supports `--fix`, `--fix-errors`, `--explain-config`, `--dump-config`, `--verify-config`
- Exports fixes to `clang-tidy-fixes.yaml` when `--fix` is used
- **Fact** (`tidy.sh:1-134`)

---

## 6. CUDA Version and GPU Requirements

- **CUDA version:** 12.4 (pinned in pixi.toml and system-requirements) — **Fact** (`pixi.toml:13-14,52`)
- **CUDA locked build:** `cuda-12.4.0` from conda-forge — **Fact** (`pixi.lock:29`)
- **cuDNN:** 9.3.0.75 — **Fact** (`pixi.lock:77`)
- **CMake USE_CUDNN:** `set(USE_CUDNN 1)` in root CMakeLists.txt — **Fact** (`CMakeLists.txt:43`)
  - Note: This variable is set but no `find_package(CUDNN ...)` call is present — *Risk (see section 9)*
- **CUDA architectures:** `native` — builds for the GPU present on the compile machine only — **Fact** (`CMakeLists.txt:56`)
- **GPU requirement:** NVIDIA GPU required at both build and runtime (native arch detection, CUDA::cublas, curand, cuDNN inference)
- **NVCC path:** forced to `$CONDA_PREFIX/bin/nvcc` — **Fact** (`CMakeLists.txt:63`, `pixi.toml:67`)
- **System CUDA ignored:** `/usr/local/cuda` and `/usr/local/cuda-12.6/compat` are in `CMAKE_IGNORE_PATH` — **Fact** (`CMakeLists.txt:13-19`)

---

## 7. How to Build From Scratch (Step by Step)

Prerequisites: pixi installed (`curl -fsSL https://pixi.sh/install.sh | bash`), NVIDIA GPU with driver >= 520, Linux x86_64.

```bash
# 1. Clone the repository
git clone <repo-url> JTML
cd JTML

# 2. Install all conda/pixi dependencies (creates .pixi/ env)
pixi install

# 3. Build VTK 9.3.0 from source (clones from GitHub, ~15-30 min)
#    This is automatically triggered by pixi run configure, but can be run standalone:
pixi run build-vtk

# 4. Configure CMake (Ninja, Release, all library paths wired up)
pixi run configure

# 5. Build all targets
pixi run build

# 6. Run the application
pixi run run
```

Alternatively, steps 3-5 are all handled by a single `pixi run build` (which depends on configure and build-vtk).

For static analysis and formatting:
```bash
pixi run format       # auto-format code
pixi run tidy         # run clang-tidy
# or via just:
just fix              # format + tidy-fix
```

---

## 8. CI Configuration

- **File:** `.github/workflows/cmake.yml` — **Fact**
- **Trigger branches:** `actions-test` only (push and pull_request) — **Fact** (`cmake.yml:7-9`)
- **Runner:** `ubuntu-latest` — **Fact** (`cmake.yml:22`)
- **Build type:** Release — **Fact** (`cmake.yml:13`)
- **Steps:** checkout → cmake configure → cmake build → ctest
- **Critical issue:** The CI workflow does NOT use pixi, does not install CUDA, Qt5, VTK, libtorch, or OpenCV. It will fail immediately on any real build attempt. This workflow appears to be a template placeholder that was never adapted for this project's actual dependencies. — *Inference with strong evidence* (`cmake.yml:26-39`)
- **Active branch coverage:** CI only runs on `actions-test`, not `main` or `pixi-dev` — **Fact** (`cmake.yml:7-9`)

---

## 9. Risks and Unknowns

| # | Risk / Unknown | Severity | Evidence |
|---|---|---|---|
| R-01 | **CI is non-functional**: The GitHub Actions workflow does not install any project dependencies (CUDA, Qt, VTK, PyTorch) and targets only the `actions-test` branch. There is effectively no automated CI. | High | `.github/workflows/cmake.yml` |
| R-02 | **`USE_CUDNN` variable set but never consumed**: `set(USE_CUDNN 1)` exists in CMakeLists.txt but no `find_package(CUDNN ...)` call or conditional compile definitions reference it. cuDNN may be silently unused or assumed via libtorch. | Medium | `CMakeLists.txt:43` |
| R-03 | **VTK must be built from source**: No conda package for VTK with Qt5 support exists in the channels. The `vtk_installer.sh` clones from GitHub and builds — this step is slow, network-dependent, and a first-build blocker. | Medium | `vtk_installer.sh`, `pixi.toml:54-56` |
| R-04 | **`native` CUDA architecture**: The binary is not portable. A build on a machine with an RTX 4090 will not run on a machine with a different GPU microarchitecture. | Medium | `CMakeLists.txt:56`, `src/gui/CMakeLists.txt:42` |
| R-05 | **`shape_sensitivity` subdirectory commented out**: The `src/shape_sensitivity/` directory exists but is not built. Intent unclear — may be unfinished or temporarily disabled. | Low | `src/CMakeLists.txt:6` |
| R-06 | **Test suite disabled**: `enable_testing()` and `add_subdirectory(test)` are commented out. No automated tests are run. | Medium | `CMakeLists.txt:128-130` |
| R-07 | **`compile_instructions.md` references outdated versions**: The legacy doc mentions CUDA 11.6 / Qt 5.15.2 / VTK 7.1.1 / libtorch 1.12.1, which conflict with the current pixi.toml (CUDA 12.4, Qt 5.15.8, VTK 9.3.0, PyTorch 2.4.1). New contributors may follow stale instructions. | Low-Medium | `compile_instructions.md`, `LINUX_BUILD.md` |
| R-08 | **CMakeSettings.json targets Windows with VS2022**: This file configures Visual Studio generators and MSVC paths. The pixi.toml is Linux-only. Cross-platform support is partial/stale. | Low | `CMakeSettings.json` |
| R-09 | **`find_package(CUDA)` is commented out**: The project uses `CUDA::cudart` etc. via modern CMake CUDA language support, but the old `find_package(CUDA)` call being commented out could cause confusion if `CUDA_LIBRARIES` is still referenced in `src/core/CMakeLists.txt:33` and `src/Study2Grid/CMakeLists.txt:33`. | Medium | `CMakeLists.txt:73`, `src/core/CMakeLists.txt:33` |
| R-10 | **`feature.dev.dependencies` is empty**: The `dev` environment has no development-only packages defined. This is either intentional (everything is in base deps) or an oversight. | Low | `pixi.toml:49-50` |

---

## 10. Doxyfile (Documentation)

- Doxyfile present, version 1.9.5 — **Fact** (`Doxyfile:1`)
- Project: "Joint Track Machine Learning", brief: "Autonomously extracting 6D kinematics from fluoroscopic images." — **Fact** (`Doxyfile:45,57`)
- No `doxygen` task is defined in pixi.toml or justfile — *Inference*: docs generation is not integrated into the standard build flow

---

## Summary Table: Key Versions

| Component | Version |
|---|---|
| CMake | >= 3.26 (locked: 3.31.4) |
| C++ Standard | C++20 |
| CUDA | 12.4 |
| cuDNN | 9.3.0.75 |
| PyTorch / libtorch | 2.4.1 (cuda120 build) |
| Qt | 5.15.8 |
| VTK | 9.3.0 (source build) |
| OpenCV | 4.10.0 |
| Eigen3 | 3.4.0 |
| clang-tools (format/tidy) | 19.1.2 |
| Platform | linux-64 only |
