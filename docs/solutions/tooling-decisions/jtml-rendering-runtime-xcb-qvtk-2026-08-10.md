---
module: render
date: 2026-08-10
problem_type: tooling_decision
component: tooling
severity: high
applies_when:
  - "the app's central VTK view renders blank after a driver/Qt/platform change"
  - "writing an offscreen VTK render test that mirrors the app's real path"
  - "debugging VTK factory registration or GL/GLX/EGL library resolution"
  - "adding or reworking pixi tasks that build long external dependencies"
  - "suspecting the refactor broke rendering (verify with a baseline diff first)"
tags:
  - vtk
  - qvtk
  - xcb
  - wayland
  - opengl
  - pixi
  - rendering
related_components:
  - qt6
  - view
  - pixi
---

# JTML rendering runtime: xcb default, QVTK smoke, VTK standalone-window limitation

## Context

The app's central interaction window went blank (images/models loaded into the
lists, but nothing drew). Debugging traced it through several misleading layers;
the resolution produced durable rules for running and verifying rendering in
this project.

## Guidance

**Run the app under `xcb`, not the Wayland default.** `QVTKOpenGLNativeWidget`
renders via Qt's GL context; on machines where the Wayland/EGL stack is
incomplete (e.g. missing `/usr/share/glvnd/glx_vendor.d/`, EGL context creation
fails), the widget gets no GL context and the VTK view draws nothing. The app's
historical working platform is xcb. Default it in the pixi task:

```toml
[tasks.run]
cmd = ".build/bin/joint-track-machine-learning"
depends-on = ["build"]
env = { "QT_QPA_PLATFORM" = "xcb" }
```

**Verify rendering with the QVTK-mirror smoke, not a standalone render window.**
`test/oracle/render_smoke.cpp` drives the app's REAL mechanism —
`QVTKOpenGLNativeWidget` + `vtkGenericOpenGLRenderWindow` + the `Viewer` wiring
(`load_renderers_into_render_window` → `setup_camera_calibration` →
`set_loaded_frames` → `update_display_background_to_*` →
`place_image_actors_according_to_calibration` → `rw->Render()`), loads the
Kneel_1 calibration + top-level tifs + fem/tib STLs, and captures via
`widget->grab()` → PNGs in `render-smoke-output/`. Run with
`ctest --test-dir .build -L render` (test env forces `QT_QPA_PLATFORM=xcb`;
DISPLAY is inherited, never baked into CMake). A rendered PNG is non-blank:
check `cv2` stats (original mean ~124 / std ~80; a blank image is uniform).

**Don't use a standalone `vtkRenderWindow` for this project.** This VTK build is
configured with `VTK_USE_X=OFF`, `VTK_OPENGL_HAS_EGL=OFF`,
`VTK_OPENGL_HAS_OSMESA=OFF`, `VTK_USE_SDL2=ON`, so VTK's own window backend has
no display path: `vtkRenderWindow::New()` returns the BASE class (object factory
never produces `vtkOpenGLRenderWindow`), `vtkTextRenderer::GetInstance()` is null,
and `vtkWindowToImageFilter` segfaults reading the framebuffer. Even linking
`vtk_module_autoinit(TARGETS ... MODULES ${VTK_LIBRARIES})` (the VTK-9 factory
registration) does not rescue standalone windows here. The QVTK path (Qt's GL
context) is unaffected — always test through it.

**pixi task caching for long external builds.** `[tasks.build-vtk]` re-ran the
full VTK build on every `pixi run` because its outputs glob didn't match. Fix:
declare `inputs` and correct `outputs` (no leading `./`):

```toml
[tasks.build-vtk]
cmd = ["./vtk_installer.sh"]
inputs = ["vtk_installer.sh"]
outputs = ["_deps/vtk/build/CMakeCache.txt"]
```

pixi then reports `Task 'build-vtk' can be skipped (cache hit)`.

## Why This Matters

The blank-window symptom had a long, misleading causal chain (half-updated NVIDIA
driver breaking `nvidia-smi`/EGL; Wayland default vs xcb; VTK standalone backend
compiled out; factory autoinit). Following these rules turns "why is the view
blank" into a 30-second check: run under xcb; if still blank, run the render
smoke; the smoke's PNGs distinguish app-wiring problems from GL/environment
problems.

## When to Apply

- Any report of a blank render window: first check the QPA platform (xcb), then
  the driver (nvidia-smi must work), then the render smoke.
- Adding render tests: mirror the QVTK path and capture with `widget->grab()`.
- Changing `vtk_installer.sh` options: only needed if standalone VTK windows are
  ever required (e.g. the oracle); the app's QVTK path does not need a rebuild.
- Re-suspecting the MVVM refactor for rendering: diff the display path against
  the pre-refactor baseline first (it is behavior-identical; `curr_frame()`
  routes through `SyncSessionState()` and works).
