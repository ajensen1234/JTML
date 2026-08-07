---
date: 2026-08-07
topic: testability-mvvm-refactor
---

# JTML Testability + MVVM Refactor (Fearless Editing)

## Problem Frame

JTML is a working, validated 2D-3D knee-implant registration tool (Qt + VTK + CUDA + OpenCV). It currently cannot be edited fearlessly: `src/gui/mainscreen.cpp` is a 5,806-line god object that mixes UI wiring, app state, compute, I/O, and optimization orchestration, and `src/core/optimizer_manager.cpp` (1,722 lines) couples the DIRECT optimization algorithm directly to CUDA. Testing is effectively disabled in the build (`#add_subdirectory(test)`, `#enable_testing()` commented out), and there is no working CI.

Because the thesis is defended, this is now a maintainability and future-research ground. The goal is to be able to change an algorithm or a button path and get a fast, headless pass/fail — never having to launch the GUI, click buttons, and discover a hang or regression after the fact.

A prior agent attempt was abandoned. Its lesson drives this plan: it produced passing tests that (a) gave **false confidence** by covering pure math and a GPU-bound "run the real MainScreen" test while never touching the thread/orchestration seams where the regressions actually lived, and (b) were **circular** — several test helpers re-derived the code-under-test's own math (e.g., pose-file and denormalization tests reimplemented production logic). Both failure modes must be structurally prevented.

The oracle is a **behavior-preservation (golden-master) gate, not an independent correctness check.** Because the baseline is captured from the current app itself, a green oracle proves refactors preserved current behavior — it does *not* certify the current behavior is correct. Genuine correctness checks come from independent sources: analytic cost functions and the known-good Labels projections (B).

---

## Actors

- A1. **Developer/researcher (user):** owns the app, writes the golden-oracle spec, runs the existing GPU app for the one-time baseline.
- A2. **Coding agent (pi):** performs the extraction and tests under the user's direction.

---

## Key Flows

- F1. **Optimize lifecycle (the seam where hangs live)** — Trigger: user clicks an optimize action in the GUI. Actors: A1 (via thin view), coordinator (headless), worker + cost function. Steps: (1) view forwards an "optimize" intent to a headless coordinator; (2) coordinator moves idle → running, manages the worker thread; (3) worker runs DIRECT over an injected cost function; (4) coordinator emits completion, moves running → finished → idle (or a defined error/cancelled state on failure/stop); (5) view is safe to re-launch. Outcome: the coordinator returns to a safe idle state and accepts a second launch; failure/stop paths return to a defined non-hung state. **Covered by:** R4, R5, R6, R7.

- F2. **Golden-oracle regression gate** — Trigger: any refactor or the Qt6 migration touches behavior. Actors: A2 (running tests), A1 (one-time baseline). Steps: (1) use the oracle spec in `golden_oracle.org` (Kneel_1: silhouettes from (A), femoral STL (D), the documented optimizer settings) to establish expected poses ≈ `fem.jts` (C) and known-good projections (B); (2) a regression test reproduces this pipeline against the captured/derived baseline; (3) refactored code must satisfy it within the recorded tolerance. Outcome: a refactor that changes the lifecycle fails the headless suite immediately; numeric drift fails the explicitly-triggered oracle gate — neither requires button-clicking. **Covered by:** R1, R2, R3, R15, R16.

- F3. **Extraction-with-gate** — Trigger: each seam cut. Actors: A2. Steps: (1) extract a boundary into a dependency-light unit; (2) land its headless test green before the next cut (presentation-only cuts get compile + scheduled manual-visual check); (3) shrink MainScreen. Outcome: every extraction is individually shippable and gated; no logic/service/coordinator change lands untested. **Covered by:** R8, R9, R10.

---

## Requirements

**[Golden oracle / anti-circularity]**
- R1. Establish a behavior-preservation baseline from the current, working Qt5 app (per `golden_oracle.org` — Kneel_1 test case) before any behavioral/structural change or the Qt6 migration. The baseline must come from validated behavior, not from re-implementing code.
- R2. No test may assert an expected value derived by re-deriving the code-under-test's own math. Extracted logic must be exercised against the golden oracle *or another independent ground truth* (analytic cost functions, real fixture round-trips, known-good projections). The oracle guards drift; the independent sources guard correctness.
- R3. Compute/cost regression comparisons must be deterministic and use documented tolerances. The DRR/cost comparison reference (CPU vs GPU) and the numeric tolerance values are a pre-condition of the oracle phase, resolved before planning (see Outstanding Questions) and recorded in `golden_oracle.org`.

**[Optimize-lifecycle seam (hang-prone) — testable headless]**
- R4. Make the optimize lifecycle testable headlessly: a coordinator/orchestration object with no widgets owns the state machine (idle → running → finished → idle, plus a defined error/cancelled state) and the worker thread, and exposes signals assertable with QSignalSpy under a plain `QCoreApplication`.
- R5. A lifecycle test drives launch → running → finished → ready-to-re-launch using an injected stub cost function (no GPU) and asserts the coordinator returns to a safe idle state and accepts a second launch — catching the "hangs after optimizer finishes" class.
- R6. The GUI binds to the coordinator as a thin caller, so the thread/completion logic lives in headless-testable code rather than inside a widget slot.
- R7. Cover init-failure and stop paths, split by what the headless default run can cover: (a) cost-init-failure and stop are covered headlessly with a stub cost (no GPU) — an error/cancel is surfaced, the coordinator returns to a defined state and re-launches cleanly, with a bounded `promptly` bound (e.g., a stub cycle must not block the calling thread and tests time out); (b) the real GPU-init-failure path lives in the explicitly-flagged GPU target only, not the default headless run.

**[MVVM decomposition]** *(long-term architectural goal, sequenced after the oracle + lifecycle + DIRECT/cost seams are green)*
- R8. Decompose `MainScreen` so presentation (widgets, slots, VTK render binding) is separated from app-state/command orchestration (frames, models, selection, save/load, optimize intent) and from services (optimization, persistence, cost, compute). This is not a prerequisite of the fast headless pass/fail; it is committed to only after phases 1–3 land and are re-validated against the human outcome.
- R9. Each extraction lands with its test gate already green before the next cut; no extraction ships untested. Gate by layer: logic/service/coordinator extractions get headless unit gates; presentation-only cuts (widgets, slots, render binding) are gated by a successful compile plus an explicitly-scheduled integration/manual-visual check — they cannot be headless-tested and god-object characterization is prohibited (R12).
- R10. MainScreen's remaining code shrinks measurably across the effort (track line count and `ui.`-reference count as a coarse coupling signal, not as a hard gate).

**[Test harness / infrastructure]**
- R11. Enable testing in CMake; the default test run is headless (no interactive QApplication/GUI, no VTK render window, no GPU). GPU/real-VTK cases compile into a separate target that runs only when explicitly requested (ctest label / flag) on a GPU machine.
- R12. No Qt-mocking infrastructure (no wrapping `QFileDialog`/`QMessageBox` behind function pointers) and no "instantiate the real `MainScreen` god object" characterization tests.
- R13. Provide a working CI on the pixi build that configures, builds, and runs the headless test suite so every change is gated by the headless seams. Bind every headless lifecycle test (and the default ctest run) with a timeout so a hang becomes a failing test rather than a hung CI job. Numeric fidelity to the oracle is verified on a separate, explicitly-triggered GPU gate (not in default CI).
- R14. Add the chosen test framework as an explicit project dependency (declared in pixi) and wire it into the build; tests must not rely on undeclared includes.

**[Preserve validated numerics]**
- R15. Preserve the validated DIRECT implementation and cost functions; refactors around them must pass the oracle/convergence tests rather than silently re-deriving behavior.
- R16. Migrate Qt5 → Qt6 as part of this effort: update the pixi `qt` pin (`5.*` → `6.*`), rebuild VTK against Qt6 (`vtk_installer.sh` currently forces `VTK_QT_VERSION=5` / `VTK_USE_QT6=OFF`), switch to `find_package(Qt6)`, and revalidate Qt5-API calls across `MainScreen`. Gated by the golden oracle + headless suite (both captured pre-migration). Because a Qt6 port most likely breaks GUI/render/UI-binding behavior the headless suite cannot see, a bounded manual GUI smoke step accompanies the migration; migration regressions are not fully subsumed by the headless suite.

---

## Acceptance Examples

- AE1. **Covers R4, R5, R7.** Given a headless `QCoreApplication` and a stub cost function, when the coordinator runs an optimize cycle to completion (and again after a stop), it returns to the idle state and accepts another launch; every lifecycle test is bounded by a timeout so a hang fails rather than hangs.
- AE2. **Covers R1, R2, R16.** Given the `golden_oracle.org` Kneel_1 case, when the code is refactored or migrated to Qt6, a regression test reproducing silhouettes (A) + femoral STL (D) with the documented settings yields a pose within the recorded tolerance of `fem.jts` (C) and projections matching the known-good Labels (B), on the explicitly-triggered oracle gate.
- AE3. **Covers R11, R12.** The default test target runs and passes with no GPU and no display; real-VTK/GPU cases require the explicit flag/target.
- AE4. **Covers R8, R9, R10.** After the decomposition, the coordinator and services no longer live in `MainScreen`; logic/service/coordinator seams are covered by headless unit gates, and `MainScreen`'s line/`ui.`-reference count has moved in the tracked direction.
- AE5. **Covers R13.** The CI job configures, builds, and runs the headless suite and fails on a regression; a deliberately stuck lifecycle test fails via timeout instead of hanging the job.

---

## Success Criteria

- **Human outcome:** A change to an algorithm or a button path is verifiable by a fast, headless test — never requiring launching the GUI and clicking to find a hang or regression.
- **Handoff quality:** This doc plus `golden_oracle.org` gives a planner the decided seams, constraints, and pre-conditions; the ordered seam/test-gate sequence and per-seam gating is produced by `ce-plan`, not authored here.

---

## Scope Boundaries

- Not a from-scratch rewrite; the validated numerical core (DIRECT, cost functions, CUDA) is preserved and gated, not silently re-derived.
- No Rust port in this effort. The optimizer seam is designed to map onto a future Rust crate (pure core + injected cost), but no Rust is written now.
- No Qt-mocking wrappers and no "run the real `MainScreen`" characterization tests.
- Full MVVM/UI overhaul of every dialog is out of scope; the focus is the `MainScreen` god object, the optimize lifecycle, persistence, and the cost/compute seams.
- Full MVVM decomposition (R8–R10) is sequenced after phases 1–3 are green and re-validated against the human outcome; it is not a prerequisite of the fast headless pass/fail.
- GUI and GPU/real-VTK tests are not part of the default headless run; numeric oracle fidelity is a separate, explicitly-triggered GPU gate.
- Regression safety currently spans a single oracle case (Kneel_1). Extending the oracle to additional fixtures is a follow-on — triggered when a refactor touches changed-case behavior or before claiming fearlessness across multiple cases.
- The ML-integration pathway (running a trained model over the original image) is deferred; the binary-silhouette path is the primary oracle.

---

## Key Decisions

- **Strangler extraction, not rewrite:** the app works and is validated; incremental, test-anchored extraction preserves that while ending at MVVM.
- **Golden-oracle-first (golden-master):** capture/define the baseline from the known-good Qt5 app before any migration or refactor. The oracle is a behavior-preservation gate (catches drift/regression, not existing bugs); anti-false-confidence and anti-circularity additionally rely on independent correctness sources (analytic cost functions, known-good projections).
- **Lifecycle-seam-first:** seal the optimize-lifecycle (headless) before the pure DIRECT math, on the working hypothesis that the hangs live in orchestration/threading; confirm the hang reproduces through the coordinator seam before committing the priority.
- **No Qt-mocking, no characterize-on-god-object:** tests target services, coordinator, and compute seams — not widgets.
- **Migrate to Qt6, gated by the oracle:** the Qt6 migration is real scope here (pixi pin + VTK rebuild + `MainScreen` API revalidation), but behavior is protected by the pre-migration baseline + headless suite, with a bounded manual GUI smoke step.
- **Optimizer takes an abstract eval:** keep validated DIRECT/cost logic; the seam is Rust-friendly without writing Rust.

---

## Dependencies / Assumptions

- **D1.** `golden_oracle.org` documents the oracle (Kneel_1: silhouettes (A), Labels (B), `fem.jts` (C), femoral STL (D)) and the optimizer settings; it is the authoritative reference for R1–R3.
- **D2.** `example_studies/Kneel_1/` contains the fixtures for the right-knee Kneel_1 oracle. `test/vtk/test_case/` is a separate left-knee case (`KR_left_8_*`) and is not usable to reproduce the Kneel_1 oracle.
- **D3 (pre-flight gate).** The known-good Qt5 baseline is captured by the user (per D1) on a GPU-capable environment before any migration/refactor. Phase 1 (oracle foundation) is not considered startable until this exists; if it cannot be produced, R1/R16 need a fallback (e.g., defer the Qt6 migration).
- **D4.** The pixi environment can host a headless Qt test runner (off-screen platform).
- **ASSUMPTION (verified):** no effective CI exists — `.github/workflows/cmake.yml` is inert boilerplate (triggers only on a branch literally named `actions-test`, runs plain `cmake -B build` without pixi/CUDA paths).
- **ASSUMPTION (verified):** no test framework is currently declared in pixi (e.g., Catch2 is not present), so R14 must add one.

---

## Outstanding Questions

### Resolve Before Planning

- [Affects R3, AE2][User decision] Confirm whether the DRR/cost oracle comparison uses a CPU render reference or the GPU path, and agree the numeric tolerance values; record both in `golden_oracle.org`. This is the load-bearing precondition of the golden gate.
- [Affects R1, R16, D3][User decision] Confirm the Qt5 GPU baseline capture is feasible before Phase 1 (pre-flight gate), and define the fallback if it is not obtainable.
- [Affects D1][User decision] Confirm the optimizer-settings block in `golden_oracle.org` (currently inconsistent — dilation lists "branch 6px" twice and omits trunk) so the settings used by R3/AE2 are correct.

### Deferred to Planning

- [Affects R14][Planner] Test framework selection (Catch2 vs QtTest) and adding it to pixi.
- [Affects R4, R5][Technical] Exact coordinator design — where the state machine lives and how the headless `QCoreApplication` + `QSignalSpy` lifecycle test is wired.
- [Affects R11][Technical] Headless Qt platform choice (offscreen) for the default test target.
- [Affects R16][Technical] Qt6 migration mechanics beyond the pixi-pin/VTK changes (autogen/UI build approach).

---

## Next Steps

- Resolve the three "Resolve Before Planning" items (golden-oracle tolerance/reference, Qt5 baseline pre-flight, `golden_oracle.org` settings correction) before planning the oracle phase.
- -> /ce-plan for structured implementation planning, phased as: (1) oracle + test-runner + CI foundation; (2) lifecycle seam; (3) DIRECT/cost seams; (4) MVVM decomposition (after re-validation); (5) Qt6 migration gated by the oracle.
