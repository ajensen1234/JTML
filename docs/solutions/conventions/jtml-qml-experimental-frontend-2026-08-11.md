---
date: 2026-08-11
last_updated: 2026-08-11
module: jtml_view
tags: [qml, qtquick, qquickvtkitem, architecture, view-model]
problem_type: convention
severity: medium
---

# QML experimental front-end: what landed, what it proved, what it costs (plan 005)

Plan 005 built a **parallel QML experimental app** (`jtml_experimental`,
`src/app/experimental/`) — the sanctioned vehicle from panoptes angle 04. It
links the widget-free backend (domain/services/coordinator/compute) + Qt Quick
+ VTK's in-tree `GUISupportQtQuick`, drives the REAL `OptimizerManager`, and
coexists with the widgets app byte-identical (R2 — the phase was additive).

## What the spike proved (U1 go/no-go → GO)

- `QQuickVTKItem` renders + interacts + captures on this box's
  xcb + Qt 6.7.2 + VTK 9.3. The in-tree `libvtkGUISupportQtQuick` needs zero
  VTK rebuild; root `find_package(Qt6 COMPONENTS ... OpenGL Quick Qml)` was
  the only root-CMake edit (additive, behavior-neutral).
- **DPI bug tail is real on this box (DPR≈2):** the pinned 9.3
  `QQuickVTKItem` never calls `QVTKInteractorAdapter::SetDevicePixelRatio` —
  drag/pick sensitivity is halved; interaction still correct. Our
  model-centric style deliberately avoids hardware picking because of it.

## Render-thread contract (the core discipline)

`QQuickVTKItem` creates the `QVTKInteractor` + a default trackball-camera style
in its own `initializeVTK` wrapper BEFORE the subclass override runs —
`renderWindow->GetInteractor()->SetInteractorStyle(...)` works from there and
from `dispatch_async` bodies. `dispatch_async` lambdas run on the **Qt Quick
render thread** (queue drained in `updatePaintNode`), NOT the GUI thread:
copy scene state on the GUI thread, capture by value, never read app-owned
mutable state inside lambdas. Scene-graph node recreation (window teardown /
item removal) re-runs `initializeVTK` — rebuild the whole pipeline from an
app-thread mirror.

## Interaction modes (feedback round)

- Camera mode: trackball camera with the scene camera focal point pinned at
  the PRIMARY model — rotation pivots at the model (the widgets app's
  near-origin focal visibly orbits a far point at pose-z distances).
- Model mode: a `vtkInteractorStyleTrackballActor` subclass whose
  `OnLeftButtonDown` sets `InteractionProp` = primary model (no picking;
  `FindPickedActor` is non-virtual in 9.3 — override `OnLeftButtonDown`
  mimicking the base's successful-pick path). Note: `vtkCallbackCommand.h`
  must be included for the complete type (`GrabFocus` conversion).
- **Default is Model mode** (the owner's workflow: line up the model, let the
  optimizer refine). Camera mode is the secondary view-orbit toggle.
- Rotation application verified against pinned VTK 9.3 source: with no user
  matrix, `Prop3DTransform` falls back to `SetPosition`/`SetOrientation` —
  `GetPosition`/`GetOrientation` DO reflect trackball-actor interaction (the
  `SetUserMatrix` path only applies when a user matrix is set).

## Model pose sync-back (the interaction → storage round trip)

The pattern that makes "drag the model, the optimizer starts from your
arrangement" work:

1. **Render-thread observer** (vtkCallbackCommand on the model style's
   `EndInteractionEvent`) reads the primary actor's `GetPosition`/`GetOrientation`.
2. **Plain thread-safe reporter method** on the renderer calls
   `emit modelPoseAdjusted(sceneIndex, x, y, z, xa, ya, za)` with by-value
   data — `emit` is thread-safe; **AutoConnection queues delivery** to
   GUI-thread receivers (Direct same-thread, Queued cross-thread).
3. **QML glue** (a ~4-line `Connections` hop): `viewport.onModelPoseAdjusted`
   → `studyBridge.applyViewerPose`.
4. **The bridge writes** `LocationStorage::SavePose` (the optimizer's starting
   point — `OptimizerBridge` passes the storage by value into `Initialize`) +
   the scene pose, then `viewport.updatePose` refreshes the readout
   (idempotent re-apply).

**The delivery mechanism matters:** `QMetaObject::invokeMethod(this, functor,
Qt::QueuedConnection)` **silently failed** in this context — posted (returned
true) but the functor never executed; the direct by-value signal emit works.
See `docs/solutions/ui-bugs/jtml-qml-model-pose-sync-queued-functor-never-delivered-2026-08-11.md`
for the full debug trail (including the disproved user-matrix hypothesis and
the stale-binary trap).

## The view-model layer (the architecture lesson)

The bridges (`StudyBridge`/`SettingsBridge`/`OptimizerBridge`/`MlBridge`/
`PoseBridge` behind an `AppBridge` hub) are the **VM layer QWidgets never
needed**: QML speaks QObject properties/signals, the seams are plain C++
(domain deliberately Qt-free). They are thin pass-throughs by rule — all
behavior lives in the seams. The genuine duplication residue (R13-forced):
the `LaunchOptimizer` drive sequence mirrored in `OptimizerBridge`, the
`BuildCostFunctionRegistryEntries` mapping replicated + golden-fixture pinned,
`matToVTK` copied. **Extraction follow-ups:** shared controllers
(`OptimizerRunController` etc.) in coordinator/services that BOTH apps call —
the same seam as the multi-stage oracle (panoptes synthesis item 3).

## QML render smoke recipe (R15)

`QQuickWindow::grabWindow` is the only capture path — `vtkWindowToImageFilter`
is a documented segfault on this build (VTK_USE_X=OFF). Gate the first grab on
a rendered frame (expose/afterRendering). Load the scene through
`QQmlApplicationEngine` so the smoke exercises type registration + import
resolution. Destroy-mid-update: QML-created items are `JavaScriptOwnership` —
`deleteLater` never completes without a top-level loop; teardown via engine
destruction. Register `LABELS "oracle;render"`, xcb env, repo-root cwd.

## Native file dialogs (xdg portal)

`libqxdgdesktopportal.so` ships in the pixi env; under xcb Qt defaults to the
built-in dialog. Force the portal theme:
`QT_QPA_PLATFORM=xcb QT_QPA_PLATFORMTHEME=xdgdesktopportal` — routes
`FileDialog` through `org.freedesktop.portal.FileChooser`. **Verified in the
Qt 6.7 sources** (the xdgdesktopportal plugin + QML `FileDialog`):
`FileDialog.OpenFiles → QFileDialogOptions::ExistingFiles → multiple:true` is
passed to the portal end-to-end. If multi-select still fails in the native
dialog, the remaining variable is the **portal backend implementation** (try
ctrl+click in GNOME's chooser; check which portal you run). Fallbacks:
`QFileDialog::getOpenFileNames` (same theme, different code path) or
`FileDialog.DontUseNativeDialog` (Qt's own dialog, reliable multi-select).

## Other conventions

- **Versionless QML imports** (`import QtQuick`) — the Qt 6 form; `2.15` is
  the legacy versioned import.
- Theme singleton: `pragma Singleton` + a `qmldir` entry (`singleton Theme
  1.0 Theme.qml`) — property access via a type name requires the singleton;
  a plain same-directory file gives `[undefined]` colors.
- Material Dark for Controls fixes default-light-on-dark clashes.
- Qt Quick `Dialog`/`Popup` for secondary surfaces (settings, poses) — the
  main screen stays lists + viewport + progress.

## Related

- `docs/solutions/tooling-decisions/jtml-rendering-runtime-xcb-qvtk-2026-08-10.md`
- `.panoptes/jtml-research-horizons/angles/04-qml-vs-widgets.org` (the evidence base)
- Plan: `docs/plans/2026-08-11-005-feat-qml-experimental-frontend-plan.md`
- `docs/solutions/ui-bugs/jtml-qml-model-pose-sync-queued-functor-never-delivered-2026-08-11.md`
- `docs/solutions/build-errors/jtml-moc-signals-section-placement-duplicate-definition-2026-08-11.md`
