---
description: HOW to build, run, configure, and develop JTML. See architecture/overview.md for WHAT.
status: active
---
# Build Guide

See also: [architecture/overview.md](../architecture/overview.md) for system context.

## Prerequisites
- NVIDIA GPU with CUDA 12.4 support
- Linux (Fedora 43 tested)
- `pixi` installed ([prefix.dev](https://prefix.dev/docs/pixi/))

## First-time setup

```bash
# Install pixi environment (downloads all deps including CUDA/Qt/VTK from conda-forge)
pixi install

# Build VTK from source (SLOW — only needed once)
pixi run build-vtk

# Configure CMake
pixi run configure

# Build everything
pixi run build
```

VTK build is the longest step (~10–30 min). Output goes to `./_deps/vtk/build/`.

## Daily development

```bash
pixi run build       # configure → build (incremental)
pixi run run         # build → run the application
pixi run format      # clang-format all sources
pixi run tidy        # clang-tidy (pass --path .build)
```

## Key build outputs
- `.build/bin/joint-track-machine-learning` — main GUI binary
- `src/gpu/libjtml_gpu.so` — GPU library
- `src/core/libjtml_core.a` — core static library
- `src/cost_functions/libJTA_Cost_Functions.so` — cost functions

## Environment variables (set by pixi activation)
```
Qt5_DIR         = $CONDA_PREFIX/lib/cmake/Qt5
VTK_DIR         = $CONDA_PREFIX/lib/cmake/vtk-9.3
Torch_DIR       = $CONDA_PREFIX/lib/cmake/Torch
OpenCV_DIR      = $CONDA_PREFIX/lib/cmake/opencv4
CUDA_HOME       = $CONDA_PREFIX
CUDACXX         = $CONDA_PREFIX/bin/nvcc
```

## Quality gates
Before merging:
1. `pixi run build` — must compile with 0 errors
2. `pixi run tidy` — check for new clang-tidy warnings
3. Manual smoke test via `pixi run run`

No automated test suite exists. See [testing/index.md](../testing/index.md).

## CMake configuration notes
- Always use Release build (`-DCMAKE_BUILD_TYPE=Release`)
- CUDA arch is `native` — binary not portable across GPU generations
- System CUDA paths are explicitly ignored; conda env CUDA is used
- `USE_CUDNN=1` is set but not consumed by `find_package` (known gap)

## Troubleshooting
- If CMake can't find Qt5: check `Qt5_DIR` env var
- If CUDA not found: ensure `pixi shell` is active, check `$CONDA_PREFIX/bin/nvcc`
- If VTK not found: run `pixi run build-vtk` first, check `VTK_DIR`
- Display issues: check `QT_QPA_PLATFORM=xcb;wayland` and `GDK_BACKEND=x11;wayland`
