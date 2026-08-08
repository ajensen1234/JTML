---
title: "JTML Qt5->Qt6 migration, hegel PBT, and oracle tooling recipes"
date: 2026-08-08
category: tooling-decisions
module: JTML
problem_type: tooling
component: build_test_tooling
severity: medium
related_components:
  - build
  - testing_framework
applies_when:
  - Migrating a Qt/CMake/CUDA desktop app from Qt5 to Qt6
  - Adding a property-based-testing (PBT) library via CMake FetchContent on a conda-forge toolchain
  - Building a GPU-labeled appearance-oracle test whose labels live in image files
  - Working around conda-forge GCC overriding CMake's BUILD_RPATH
---

# JTML Qt5->Qt6 migration, hegel PBT, and oracle tooling recipes

Captured after migrating JTML Qt5 -> Qt6, adding a hegel property-based test
layer, and building the Tier-2 GPU appearance oracle (U6-U8). These are the
concrete, non-obvious steps that cost real time the first time.

## Qt5 -> Qt6 migration recipe

- **The conda-forge Qt6 metapackage is `qt6`, but pin `qt6-main` + `qt6-wayland`
  (`6.7.2`) in pixi**, NOT `qt = "6.*"` (the `qt` package only ships Qt5).
  `pixi add "qt6-main=6.*" "qt6-wayland=6.*"`.
- **VTK rebuild against Qt6** (`vtk_installer.sh`): switch `-DVTK_QT_VERSION=6`,
  `-DVTK_USE_QT6=ON`, `-DVTK_GROUP_ENABLE_Qt6=YES`, `-DQt6_DIR=$CONDA_PREFIX/lib/cmake/Qt6`,
  and DELETE the stale Qt5 VTK `build`/`install` dirs before reconfigure.
- **CMake**: `find_package(Qt6)` + `Qt6::` (Core/Gui/Widgets, and `Qt6::Test` for
  QtTest seams); pixi configure/activation use `-DQt6_DIR` / `Qt6_DIR`.
- **API deltas that break the build**: `QRegExp` is gone from QtCore ->
  `QRegularExpression` (escaped `[\\r\\n]`); `QDesktopWidget`/`qdesktopwidget.h`
  are removed (drop the include, use `QGuiApplication::primaryScreen()` if needed);
  and **`QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat())`
  MUST run BEFORE `QApplication`** in main() (latent bug, louder on Qt6).
- **OpenCV stays on its qt5-linked conda build** while Qt6 is present: benign, the
  app links Qt6::Widgets and uses no OpenCV Gui. A qt6-linkage OpenCV would bump
  OpenCV to 5.0.0 (major API change), so deliberately avoided.
- **Gate the migration with the pre-migration oracle + headless suite**, then a
  bounded manual GUI smoke (event-loop-ran-without-crash on a real display).

## hegel property-based testing via CMake FetchContent (conda-forge GCC)

- hegel-cpp is a static lib pulled via `FetchContent` (`GIT_TAG v0.11.1`); it also
  downloads a prebuilt `libhegel_c.so` at configure time and needs a `uv`-launched
  Python `hegel-core` server at runtime (cached under `.hegel/`, auto-gitignored).
- **Two link-time fixes on the conda-forge GCC toolchain**: (a) link `dl`
  (`dladdr`/`dlsym`), and (b) **append** `-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel`
  because the conda toolchain does NOT honour CMake's `BUILD_RPATH` and the lib is
  emitted there at configure time. GNU ld accumulates `-rpath`, so the pixi lib
  path stays.
- hegel is runner-agnostic: call `hegel::test([](hegel::TestCase& tc){ ... })`
  INSIDE a Catch2 `TEST_CASE`. This is property-based testing (auto-generates +
  shrinks) — complementary to, not a replacement for, the deterministic Tier-1
  golden.
- **Deliberate deviation**: hegel is introduced via FetchContent, not pixi.toml
  (AGENTS.md convention). Documented as a swappable layer; gate it behind an
  option if offline configure must not break.

## Tier-2 GPU appearance oracle + label-origin convention

- The load-bearing gate is a **SILHOUETTE/IOU comparison**, not raw pose (DIRECT
  numeric convergence is noisy). Render the implant at the recovered pose and IoU
  vs a known-good label image.
- **Binary label TIFFs use a bottom-left y-origin** while the GPU renderer outputs
  top-left: `cv::flip(label, 0)`. Pin the frame<->label correspondence EMPIRICALLY
  (start-pose render IoU == 1.0), not by filename (names are often unaligned).
- Feed the oracle the **processed Frame outputs** (edge/dilation/intensity/
  distance-map), not the raw x-ray, or the search is misled.
- Oracle reads fixture-relative paths, so run it from the repo root: set
  `WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}` or `TEST_WORKING_DIRECTORY` on the
  ctest entry (ctest defaults CWD to `.build`). Label `oracle` only, never headless.

## Conda-forge GCC overrides CMake BUILD_RPATH

A reusable cross-target gotcha: the conda GCC toolchain forces its own rpath as an
RPATH and ignores `set_target_properties(... BUILD_RPATH ...)`. Fix: append
`-Wl,-rpath,<dir>` via `target_link_options` on the specific target.
