---
title: "QML model-pose sync: queued invokeMethod functor silently never delivered"
date: 2026-08-11
category: ui-bugs
module: jtml_view
problem_type: ui_bug
component: frontend_stimulus
symptoms:
  - "dragging the model in Model interaction mode never updated the pose readout; the stored pose kept the pre-drag value"
  - "an optimize run would start from the stale stored pose, not the visually arranged one — silent loss of the user's arrangement"
  - "instrumented smoke showed syncCount=0 on the receiver while the EndInteraction observer DID read the rotated actor pose (59.56, -0.28, -502.16 vs drag baseline)"
  - "QMetaObject::invokeMethod returned true (event posted), so nothing reported an error"
root_cause: async_timing
resolution_type: code_fix
severity: high
tags: [qml, qtquick, qquickvtkitem, vtk, signals, queued-connection, render-thread]
---

# QML model-pose sync: queued invokeMethod functor silently never delivered

## Problem

In the experimental QML front end (plan 005, feedback #2), dragging the primary
model in Model interaction mode was supposed to sync the rotated actor pose back
into `LocationStorage` so the optimizer starts from the visually arranged pose —
but the pose never arrived: the readout never updated and the stored pose stayed
stale, with no error anywhere in the chain.

## Symptoms

- Pose readout in the QML UI never changed after a model drag.
- `LocationStorage` kept the pre-drag pose — a subsequent optimize run started
  from the stale pose (silent user-workflow breakage).
- Smoke instrumentation showed `syncCount=0` on the receiver side while the
  render-thread observer *did* read the rotated transform
  (`59.56, -0.28, -502.16` vs baseline) — the break was downstream of the
  observer, not in the interaction.
- The queued `invokeMethod` returned `true`, so nothing looked broken.

## What Didn't Work

- **(a) User-matrix hypothesis (dead end):** suspected
  `vtkInteractorStyleTrackballActor::Prop3DTransform` applies the rotation via
  `SetUserMatrix`, so `GetPosition`/`GetOrientation` would never see it.
  Verified against the pinned VTK 9.3 source: with **no** user matrix the
  else-branch calls `SetPosition`/`SetOrientation` — the transform *is*
  visible. The observer readouts confirmed the hypothesis was wrong.
- **(b) Stale-build trap (dead end):** a missing `<QThread>` include broke the
  build, but the smoke kept running a **stale binary** hiding the new
  instrumentation — every "no change" observation during the debug was against
  an old executable.
- **(c) moc signals-placement trap (red herring here):** the duplicate-definition
  link errors from the moc trap looked like the pose-sync culprit; it was a
  distinct build-only bug.

## Solution

Replace the queued `invokeMethod` hop with a **direct signal emit carrying
by-value data**.

```cpp
// BEFORE (broken — silent delivery failure)
// render-thread observer (vtkCallbackCommand on EndInteractionEvent):
renderer->queueModelPoseSync(0, pos[0], pos[1], pos[2], orient[0], orient[1], orient[2]);

// GUI-thread renderer object, invoked FROM the render thread:
void QmlVtkRenderer::queueModelPoseSync(/* ... */) {
    QMetaObject::invokeMethod(this, [this, ...] {
        emit modelPoseAdjusted(...);
    }, Qt::QueuedConnection);   // returns true ... functor never executes
}

// AFTER (working)
// plain public method, safe to call from any thread; emit is thread-safe
void QmlVtkRenderer::reportModelPoseAdjusted(int sceneModelIndex, double x,
                                             double y, double z,
                                             double xa, double ya, double za) {
    emit modelPoseAdjusted(sceneModelIndex, x, y, z, xa, ya, za);
}
```

The observer calls `renderer->reportModelPoseAdjusted(0, pos[0], ..., orient[2])`
directly. Receivers (`Connections` in `main.qml` → `StudyBridge::applyViewerPose`
→ `LocationStorage::SavePose` + scene + readout) have GUI-thread affinity, so
**AutoConnection queues delivery** to them — exactly the thread-hop the
invokeMethod was supposed to provide. Also flipped the default interaction mode
to Model mode (owner's workflow: line up the model, let the optimizer refine).

## Why This Works

Root cause: `QMetaObject::invokeMethod` with a functor + `Qt::QueuedConnection`
posted the event (returned `true`) but the functor **never executed** in this
harness/app context — a silent functor-queue delivery failure (no error, no
assertion, no signal). A direct `emit` has no intermediate hop: the signal
machinery resolves the connection type per receiver (Direct for same-thread,
Queued for cross-thread under AutoConnection), carries plain by-value
parameters, and has no lambda capture or functor-ownership question.

## Prevention

- **Smoke leg 3.5 pattern** (`test/oracle/qml_render_smoke.cpp`): QTest
  press/move/move/move/release drag through the real Qt Quick pipeline; connect
  to the actual signal with a capture counter; assert `syncCount > 0` **and**
  `poseDelta >= 0.01` (pose actually rotated). Converts the silent failure into
  a loud, render-labeled test.
- When a queued delivery is the suspect, instrument the **receiver** side
  (counter), not just the sender.
- Rebuild-and-re-run discipline after any compile fix (stale-binary trap);
  treat "build failed" as "the previous binary is now suspect".
- Keep `signals:` sections last in Q_OBJECT classes (the moc trap) and include
  what you use (`<QThread>` etc. — the transitive-include era is over).

## Related Issues

- `docs/solutions/build-errors/jtml-moc-signals-section-placement-duplicate-definition-2026-08-11.md`
- `docs/solutions/conventions/jtml-qml-experimental-frontend-2026-08-11.md`
- `docs/solutions/conventions/jtml-mainscreen-decomposition-patterns-2026-08-10.md`
  (the same silent-wiring-death failure class as the QListView auto-connect swap)
