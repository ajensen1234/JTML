---
title: "Guarding an allocation path: null-init the freed members and validate the guard's precondition (GPUHeatmap 0-keypoint)"
date: 2026-08-12
category: logic-errors
module: jtml_compute
problem_type: logic_error
component: tooling
symptoms:
  - "optimizer run aborted with 'Error uploading heatmap to GPU!' on studies loaded without ML segmentation"
  - "after the guard: non-deterministic behavior at teardown (GPUMetrics destructor cudaFree'd uninitialized pointers)"
  - "the guard sometimes silently re-entered the pre-fix abort (keypoint count read as garbage)"
root_cause: missing_validation
resolution_type: code_fix
severity: high
tags: [cuda, heatmap, gpu, guard, uninitialized, destructor]
related_components:
  - frontend_stimulus
---

# Guarding an allocation path: null-init the freed members and validate the guard's precondition

## Problem

The optimizer aborted initialization with `Error uploading heatmap to GPU!`
whenever a study was loaded without ML segmentation — curvature heatmaps
only exist after a segmentation, so the GPU upload got a null source buffer.
The fix (a 0-keypoint no-upload guard) introduced two follow-on defects that
a code review caught: the guarded members were freed uninitialized at
teardown, and the guard's precondition was itself uninitialized.

## Symptoms

- `Error uploading heatmap to GPU!` — `OptimizerManager::Initialize` failed
  on `GPUHeatmap` upload for a no-segmentation study (0 keypoints, null
  host buffer; a 0-size `cudaMemcpy` from `nullptr` fails on this driver).
- After the guard landed: `GPUMetrics::~GPUMetrics` ran
  `cudaFree(dev_curvature_hausdorf_score_)` / `cudaFreeHost(...)` on
  **indeterminate** pointers whenever the 0-keypoint path skipped
  `AllocateCurvatureHausdorfScore` — UB at teardown on every heatmap-less
  run.
- The guard keyed on `Frame::GetNumCurvatureKeypoints()`, a member that was
  never initialized in the `Frame` constructor (only `setCurvatureHeatmaps`
  wrote it) — the no-segmentation path read garbage, so the guard could
  silently re-enter the pre-fix abort (the owner's first fixed-binary run
  worked by luck).

## What Didn't Work

- Passing `nullptr` through to `cudaMemcpy` and relying on the driver to
  tolerate a 0-size copy — the driver rejects the null source on this box.
- Guarding at the call site on the keypoint count alone, without verifying
  the count was initialized — garbage read as "> 0" bypassed the guard.
- Guarding the allocation without null-initializing the members the
  destructor frees — the skip path made the dtor's `cudaFree` indeterminate.

## Solution

Three coordinated changes (the "guard triad"):

1. **Constructor-level no-upload state** (`src/compute/gpu_heatmaps.cu`):
   `num_keypoints <= 0` → `initialized_correctly_ = true;
   heatmap_on_gpu_ = false; dev_heatmap_ = 0; return;` — no `cudaMalloc`,
   no `cudaMemcpy`, and the destructor's `cudaFree(0)` is a legal no-op.
2. **Null-init the guarded members** (`include/compute/gpu_metrics.cuh`):
   `int* curvature_hausdorf_score_ = nullptr; int*
   dev_curvature_hausdorf_score_ = nullptr;` — the dtor's `cudaFree`/
   `cudaFreeHost` on null are no-ops. Also guard
   `AllocateCurvatureHausdorfScore(0)` with an early return (0-size
   `cudaHostAlloc` is not guaranteed on every driver).
3. **Validate the guard's precondition** (`include/compute/frame.h`):
   `int num_curvature_keypoints_ = 0;` — the member is now initialized at
   construction; the caller's skip-if-empty check and the ctor's count
   check can no longer disagree.

## Why This Works

The root failure class is "guarding a resource path without hardening the
resource's other lifecycle points": every allocation guard must (a) leave
the object in a state the destructor can safely run, and (b) key on a value
that is initialized on every path. The 0-keypoint contract is now explicit
end-to-end: `Frame` reports 0, the upload loop skips, the heatmap object
initializes with 0 keypoints (vector stays frame-aligned so cost functions'
`at(i)` access remains valid), the score allocation no-ops, and teardown
frees only null pointers.

## Prevention

- When adding an early-return guard to any allocation path: grep every
  member the destructor frees and null-initialize them; verify the guard's
  condition variable is initialized on ALL construction paths.
- Keep the frame-aligned vector contract: a skipped upload must still push
  a valid (0-keypoint) object so index-based consumers stay aligned.
- The 0-keypoint GPU path is oracle/manual-visual territory (no headless
  GPU) — the pins are the constructor/guard invariants plus the owner's
  heatmap-less run; run compute-sanitizer on teardown after such a run.

## Related Issues

- `docs/plans/2026-08-12-007-refactor-qml-experimental-improvement-pass-plan.md`
  (owner-feedback round 1, item 3; the review round's two P1s)
- `docs/handoff-2026-08-12-qml-experimental-improvement.md`
