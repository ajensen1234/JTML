# Handoff — QML experimental improvement pass (plan 007) complete

> **PHASE COMPLETE (2026-08-12).** Executed by
> `docs/plans/2026-08-12-007-refactor-qml-experimental-improvement-pass-plan.md`
> (U1–U7 landed + two owner-feedback fix rounds + a ce-code-review fix
> round): structured review → theme tokens + component extraction
> (main.qml 1020 → ~517 lines post-U4, further shrunk by the review
> round's toolbar extraction — the plan's ~300-line target was superseded
> by the composition pass, see the review round in the plan) →
> view-state gap fixes (pose-cell commit contract, table refresh,
> ML-seed invalidation, run-lock matrix, keyboard wiring) → UX audit +
> visual composition pass → pose-table virtualization (ListView reuse,
> commit-on-pool) → Qt Quick Test harness (38 test functions, headless) →
> profiling baseline. Gates: 48/48 headless (incl. `jtml.qml_lint`),
> `jtml.qml_parity_check` green (IoU band intact), build green. The
> harness caught 3 real product bugs during U5/U6 (required-property
> FINAL collision, broken dirty-close guard, commit-on-pool focus
> assumption). Conventions: `docs/solutions/conventions/jtml-qml-view-testing-2026-08-12.md`.

**Read first:** `AGENTS.md`, the plan (U-IDs below), and the conventions
entry above. Prior context: `docs/handoff-2026-08-11-vm-layer-extraction.md`.

## Owner-feedback rounds (all fixed + committed + gated)

Round 1 (2026-08-12 run):
1. Toolbar pills overlapping the viewport — fixed (toolbar sizes from
   content + clip).
2. Dragging the femur moved the tibia on release — fixed: the model-style
   EndInteraction observer reported a hardcoded `sceneModelIndex = 0`;
   it now carries the moved actor's index (`QmlVtkRenderer.cpp`).
3. Optimizer aborted with "Error uploading heatmap to GPU!" on a study
   without ML segmentation — fixed: zero-keypoint heatmaps are a
   legitimate no-upload state (`GPUHeatmap` ctor guard,
   `AllocateCurvatureHausdorfScore` guard, `OptimizerManager::Initialize`
   skip-if-empty). **Note: this touched `src/compute/` +
   `src/coordinator/` at the owner's explicit request.**
4. "Black sil." checkbox black text on the dark panel — fixed via a
   `contentItem` text override + a WCAG-contrast pin in the harness.

Round 2 (frame picker): 13 clicks never moved `currentIndex` —
`ListView.view` is null in nested delegate items (attached `view` reaches
the delegate root only). Fixed with the list id; pinned with
`test_frameClickMovesIndex` + `test_modelClickTogglesRow`. The old pin
had skipped the mouse path as "ListView-standard" — the exact gap.

## Where things stand

- `jtml_experimental` (src/app/experimental/): bootstrap main.qml +
  StudyPanel/MlStrip/RunBar/ViewportPanel/PosesDialog/PosesTable/PoseCell/
  SettingsPanel/Theme; bridges stay thin relays (D1 injection);
  virtualized pose table with the corrected commit-on-pool; run-lock
  matrix incl. Camera/Model toggles; keyboard + focus + a11y; contrast-
  bumped tokens; dirty-close guards.
- Test inventory: 48 headless ctest entries incl. `jtml.qml_lint`
  (qmllint gate) and `jtml.qml_view` (38 QML test functions over real
  component sources with fake bridges).
- Profiling: `.build-prof` via the `configure-profiling`/`build-profiling`
  pixi tasks; first report shows zero jank and a 3.2 s file-dialog open
  cost (the queued picker's evidence).

## Remaining (owner queued)

- **Re-verify on the next app run:** the 4 round-1 fixes + the frame
  picker (all in the current build).
- **Re-profile** now that the frame picker works — the first trace's
  session was truncated by the regression; the Poses table scroll/edit
  paths are still unmeasured.
- **QML path-bar picker** (plan 007 deferred item B): replace the
  image/model QFileDialogs with a `PathPickerDialog.qml` (copyable
  Location field, paste-to-jump, checkboxes, remembered dirs). The
  minimal interim fix landed: per-purpose remembered dir + sidebar MRU
  bookmarks in `FileDialogBridge` (QSettings org `JointTrackAutoGPU` /
  app `jtml_experimental` — outside the oracle-pinned registry scope).
- **Environment caveat:** the in-flight pixi.lock change moved CUDA
  headers to `$CONDA_PREFIX/targets/x86_64-linux/include`; a FRESH cmake
  configure now fails at Torch's `cuda.h` probe (the warm cache hides it
  in `.build`). Workaround for fresh dirs: symlink the headers into
  `$CONDA_PREFIX/include` during configure, then remove. Worth fixing in
  the pixi recipe or the cuda package selection.

## Next phase

The recorded algorithm roadmap the owner deferred on 2026-08-12 (plan 007
Problem Frame): synthesis items 1 → 3 → 2 → 5 → 8 (cost-path bug fixes,
metric-ablation harness, multi-stage oracle, DIRECT variants, compute
perf). The QML front-end is in a state where it can be the experiment
vehicle for that work.
