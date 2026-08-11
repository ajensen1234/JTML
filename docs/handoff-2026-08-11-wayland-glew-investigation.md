# Handoff — investigate: Wayland + GLEW init failure ("GLEW could not be initialized: Unknown error")

**Read first:** `docs/solutions/tooling-decisions/jtml-rendering-runtime-xcb-qvtk-2026-08-10.md`
(the documented xcb-only constraint — this error is the concrete mechanism
behind the "Wayland/EGL blank render" that forced `QT_QPA_PLATFORM=xcb`),
`AGENTS.md` (build/test conventions), and this repo's layered-lib layout.

## The symptom

Running either app under Wayland (`QT_QPA_PLATFORM=wayland`) on this box
produces a repeating error from the VTK render window:

```
Window (0x...): GLEW could not be initialized: Unknown error
vtkGenericOpenGLRenderWindow (0x...): GLEW could not be initialized: Unknown error
```

repeating every ~7ms (the render loop retrying). The failing code is
`vtkOpenGLRenderWindow::OpenGLInitContext` (`_deps/vtk/source/Rendering/OpenGL2/vtkOpenGLRenderWindow.cxx`
~line 704): `glewInit()` returns a non-`GLEW_OK` result, and
`glewGetErrorString(result)` returns the literal string **"Unknown error"**.

Under `QT_QPA_PLATFORM=xcb` the same code works (the app's current working
configuration).

## Environment facts (verified 2026-08-11)

- **Session:** niri (Wayland compositor), `XDG_SESSION_TYPE=wayland`,
  Fedora 44 KDE Plasma Edition (system). The app is normally forced to xcb
  for rendering; the user also noticed "wayland doesn't work because glew
  isn't initialized" — this error.
- **VTK 9.3 build** (via `vtk_installer.sh`, in `_deps/vtk/install`):
  `VTK_USE_X=OFF`, `VTK_OPENGL_HAS_EGL=OFF`, `VTK_OPENGL_HAS_OSMESA=OFF`,
  `VTK_USE_SDL2=ON`. VTK has **no native GL backend of its own** — rendering
  goes through the Qt-provided context (`vtkGenericOpenGLRenderWindow` inside
  `QVTKOpenGLNativeWidget` / `QQuickVTKItem`).
- **pixi env GL stack** (`.pixi/envs/default/lib`): conda-packaged
  `libEGL.so.1`, `libGLESv1_CM.so`, `libGLdispatch.so.0` (glvnd pieces),
  plus the rest of the Mesa/glvnd set. The conda GL libs shadow the system's
  inside the pixi environment.
- **Both apps exhibit it** (the widgets app `joint-track-machine-learning`
  and the QML app `jtml_experimental`) — this is a shared render-runtime
  issue, not QML-specific (verify by running the widgets app under wayland).
- The error is thrown from the render-thread/context-init path; the render
  loop keeps retrying, so the log floods.

## What "Unknown error" tells us

`glewGetErrorString` only returns "Unknown error" for error codes outside its
known enum (`GLEW_ERROR_NO_GL_VERSION`, `GLEW_ERROR_GL_VERSION_10_ONLY`,
`GLEW_ERROR_GLX_VERSION_11_ONLY`, `GLEW_ERROR_NO_GLX_DISPLAY`). Two readings:

1. **No current context at `glewInit()` time** — the Qt EGL context failed to
   create or wasn't current when VTK initialized (most likely: context
   creation itself failed under Wayland, so `glewInit` cannot query a GL
   version and the error path is odd/uninitialized).
2. **GLEW symbol/runtime mismatch** — the `glew` used to compile
   (`vtkglew`-style vendored or conda glew) differs from what's resolved at
   runtime through the conda `libGLdispatch`, producing an out-of-range error
   code. (Check which glew the VTK build actually links and what
   `LD_DEBUG=libs` shows at runtime.)

## Hypotheses (ranked)

- **H1 — Qt context creation fails under Wayland on this box.** niri is a
  minimal compositor; the conda `libEGL` may not find a usable EGL
  implementation/ICD, or the requested `QSurfaceFormat` (VTK's
  `QVTKOpenGLNativeWidget::defaultFormat()` — 4.5 core, etc.) cannot be
  satisfied by the EGL stack, so `QOpenGLContext::create()` fails and the
  context VTK calls `makeCurrent()` on is invalid → `glewInit` fails.
- **H2 — conda GL stack vs system GL conflict.** The pixi env's
  `libGLdispatch`/`libEGL` shadow the system's glvnd; the conda EGL may not
  load the system GPU driver's ICD (or loads a mismatched one), breaking
  context creation under Wayland while X11 (GLX via XWayland) still works
  through a different path.
- **H3 — GLEW platform expectations.** VTK's vendored glew may be built with
  GLX assumptions that don't apply to a pure EGL context (even though
  `VTK_USE_X=OFF`); the GLX-specific error enum would then be misleading.
- **H4 — App-surface-format interaction.** The QML app additionally calls
  `QQuickVTKItem::setGraphicsApi()`; the widgets app sets
  `QSurfaceFormat::setDefaultFormat(...)` before `QApplication`. If the
  surface format isn't applied identically under the Wayland platform plugin,
  context creation differs.

## Investigation steps (fresh session)

1. **Isolate Qt GL under Wayland on this box:** a minimal probe —
   `QOpenGLWindow`/`QOpenGLWidget` with a clear color under
   `QT_QPA_PLATFORM=wayland` (plain Qt, no VTK). If that fails, the problem
   is Qt+EGL+niri (H1/H2), not VTK. (Note: other apps on this session render
   GL fine — Okular uses the gtk3 theme with in-process GL — so the system
   GL stack works; the conda-env GL stack is the prime suspect.)
2. **Check context creation directly:** in the app, before VTK initializes,
   create a `QOpenGLContext` with the app's surface format, call `create()`
   and `makeCurrent()`, and print the result + `QOpenGLContext::errorString()`
   under both platforms. This pinpoints whether the context itself fails.
3. **Print the raw `glewInit` result:** a one-off patch (or a probe that
   replicates the init sequence) printing the `GLenum` value — confirm
   whether it's a known GLEW error or out-of-range (H3).
4. **Library resolution:** `LD_DEBUG=libs` (or `ldd` on the binaries) under
   both platforms — which `libEGL`/`libGLX`/`libGLdispatch` resolve, and
   whether the conda copies shadow the system's. Try running with the conda
   GL libs excluded (e.g., unset/trim `LD_LIBRARY_PATH` for the GL libs) to
   test H2 — **careful**: do this as a pure experiment, do not change the
   pixi env or system config.
5. **Compare the two apps** under wayland — if the widgets app shows the
   same flood, the QML-specific `setGraphicsApi` is exonerated.
6. **Decide:** the app works under xcb (the documented constraint). The
   goal of this investigation is understanding + optionally enabling
   Wayland; it is NOT a blocker for any current work. If H2 pans out, the
   fix might be a pixi-env tweak (e.g., a GL-lib exclusion) rather than code.

## Decision boundary

- xcb remains the shipping configuration; this is investigation only.
- Do not change system portal/GL configuration (the user's system is
  off-limits after the portal incident — repo and pixi-env experiments only,
  and even pixi-env changes need the user's OK).
- If the investigation stalls, document findings in
  `docs/solutions/tooling-decisions/` and leave xcb as the working path.

## Key pointers

- `docs/solutions/tooling-decisions/jtml-rendering-runtime-xcb-qvtk-2026-08-10.md`
- `_deps/vtk/source/Rendering/OpenGL2/vtkOpenGLRenderWindow.cxx` (~line 704,
  the `glewInit` failure + the 3.2-version check below it)
- `_deps/vtk/source/GUISupport/Qt/QVTKOpenGLNativeWidget.h` (the Qt-GL
  surface path the widgets app uses)
- `_deps/vtk/install/include/vtk-9.3/QQuickVTKItem.h` + `setGraphicsApi()`
  (the QML path)
- pixi env GL libs: `.pixi/envs/default/lib/{libEGL,libGLESv1_CM,libGLdispatch}*`
- The app entry points: `src/app/main.cpp` (widgets) and
  `src/app/experimental/main.cpp` (QML) — both set the default surface
  format before the app object.
