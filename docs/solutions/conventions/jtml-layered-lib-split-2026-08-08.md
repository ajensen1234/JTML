---
module: build-system
date: 2026-08-08
problem_type: convention
component: tooling
severity: medium
applies_when:
  - "restructuring a monolithic Qt/CUDA lib into architecture-aligned layers"
  - "adding a source file to a layered lib and wiring its CMake target"
  - "rewriting include paths across layers (prefix standardization)"
  - "splitting or merging CMake targets without breaking consumers"
  - "attesting zero behavior change for a pure relocation (R15)"
tags:
  - layered-libs
  - cmake
  - jtml
  - include-prefixes
  - grep-guards
  - hegel
related_components:
  - cmake
  - testing
---

# JTML Layered-Lib Split (003) — conventions that make it stick

## Context

Plan 003 turned one monolithic `jtml_core` STATIC lib (+ `jtml_gpu`/`JTA_Cost_Functions`
SHAREDs + a `src/gui` exe) into six architecture-aligned targets, with **zero runtime
behavior change** (R15). The structure now is:

| Target | Type | Contents |
|---|---|---|
| `jtml_domain` | STATIC | pure logic (9 modules); Qt/GPU-free — the future FFI surface |
| `jtml_services` | STATIC | model/stl/optimizer_settings/location_storage/calibration (Qt-linked, purity decouples deferred) |
| `jtml_coordinator` | STATIC | optimize_coordinator + optimizer_manager (QObject) |
| `jtml_view` | STATIC | the Qt Widgets classes (`.ui`/`.qrc`/`Resources` atomic under `include/view`) |
| `jtml_compute` | SHARED | the merged GPU + cost-functions surface (`src/compute`) |
| `app` (exe) | exe | thin composition root `main.cpp` + `Study2Grid` |

Dependency graph is **acyclic and downward-only**: `domain ← services ← coordinator ← view ← app`; `compute` is standalone; `app` links `view` + `coordinator` + `compute`.
`JTA_LIBS` is the single CMake handle to `jtml_compute` (defined exactly once, before `add_subdirectory(src)`).

## Guidance

**Per-layer CMakeLists (STATIC libs).** Every layer dir owns its `CMakeLists.txt`:
`file(GLOB HEADER_FILES CONFIGURE_DEPENDS <include/layer>/*.h[ + .cuh/.ui])` **plus an
explicit `.cpp`/`.cu` source list**. Never GLOB sources: a header globbed without its
impl in the explicit list causes an AUTOMOC undefined-symbol link error, and a new
`Q_OBJECT` header must be GLOBbed into its owning target so AUTOMOC mocs it.
`CONFIGURE_DEPENDS` is mandatory — without it a new `.h`/`.cuh`/`.ui` is silently
excluded until a manual reconfigure.

**Include prefixes are standardized + grep-gated.** Every header is included by its
layer prefix (`domain/x.h`, `compute/gpu_frame.cuh`, `view/mainscreen.h`). Zero bare
or cross-layer prefixes are allowed. The pre-003 `gpu_*.cuh` family is now
`compute/gpu_*.cuh`. Grep guards (zero `#include "core/`, `"gui/`, bare `gpu_`)
are the regression gate for the whole tree.

**The one SHARED node owns CUDA.** `CUDA_SEPARABLE_COMPILATION ON` + `CUDA_ARCHITECTURES`
live only on `jtml_compute`. STATIC libs that merely reference compute symbols just
link `${JTA_LIBS}`.

**Composition root is thin.** `main.cpp` only sets the QVTK default surface format,
constructs the view, and runs the loop; exe links are `PRIVATE` with
`CUDA_ARCHITECTURES native` + explicit RPATHs (the conda toolchain overrides
`BUILD_RPATH`, so append `-Wl,-rpath` via `target_link_options`).

**`jtml_view` AUTOUIC autogen is PUBLIC.** Because `mainscreen.h` includes
`ui_mainscreen.h`, consumers (the app exe) inherit the view's autogen include dir:
`target_include_directories(jtml_view PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/jtml_view_autogen/include)`.
This bakes in the target name — renaming `jtml_view` breaks the app compile, so keep
the in-file note.

## Why This Matters

A directory that expresses the architecture is grep-able and navigable: you open the
layer you are touching, the pure FFI surface is provably free of
`QObject|vtk|cuda|torch|opencv`, and the headless/PBT suite compiles layer sources
directly. A header change silently losing the build (missing `CONFIGURE_DEPENDS`),
a bare include rebinding to the wrong layer, or a one-sided `JTA_LIBS` edit breaking
every consumer are the exact failure classes the split was meant to eliminate.

## When to Apply

- Adding a source: put it in the layer that owns its concept, add to the explicit
  list (not just the GLOB), prefix its includes.
- Touching the link graph: keep `JTA_LIBS` single-set, keep the graph downward-only
  (`services` must not depend on `view`, `domain` includes nothing above it).
- Relocating files: the canonical tool of record is a **verified exact-string
  scripted rewrite** for include prefixes (ast-grep's tree-sitter cannot target
  C++ `#include` preprocessor directives) — then `jj diff` every changed line and
  run the zero-prefix greps.

## Examples

```cmake
# src/compute/CMakeLists.txt (the merged SHARED lib)
set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/compute)
file(GLOB HEADER_FILES CONFIGURE_DEPENDS ${HEADER_DIR}/*.h ${HEADER_DIR}/*.cuh)
add_library(jtml_compute SHARED <explicit .cu/.cpp list> ${HEADER_FILES})
set_target_properties(jtml_compute PROPERTIES CUDA_SEPARABLE_COMPILATION ON ...)
target_link_libraries(jtml_compute PUBLIC ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES}
                      CUDA::cublas CUDA::cudart CUDA::curand)
```

Include rewrite (149 sites in U4): `#include "gpu/X"` → `#include "compute/X"`,
bare `gpu_Y.cuh` → `compute/gpu_Y.cuh` — then
`grep -rIn '#include "\(core/\|gui/\|gpu/\|gpu_\|cost_functions/\)' src include test` must return zero.