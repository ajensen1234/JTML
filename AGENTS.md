# JTML — Agent Working Guide

JTML is a Qt5 + VTK 9.3 + CUDA 12.4 + OpenCV C++20 desktop app for 2D-3D knee-implant
registration (D.R.E.C.T. global optimizer over a GPU cost function). This file captures the
conventions a coding agent needs to work here without re-deriving them.

## Build & environment (pixi)

Everything build-related is handled by pixi tasks — never invoke cmake/make/nvcc directly.

- `pixi run configure` — cmake configure (pulls deps; builds VTK once via `vtk_installer.sh`)
- `pixi run build` — build all targets (incl. the GUI + tests)
- `pixi run test` — run the **headless** test suite via `ctest -L headless --timeout 600`
- `pixi run run` — launch the GUI app
- `pixi add <pkg>` / `pixi run tidy` / `pixi run format` — deps / clang-tidy / clang-format

Add any new C++ test framework/tool to `pixi.toml` (the lockfile is `pixi.lock`).

## Version control is `jj` (Jujutsu), NOT git

- Always use `jj` commands. Never raw git.
- The repo owner's workflow is **`jj describe -m "<scope>: <msg>"` then `jj new`** for each
  logical change. `jj new` starts the next change after `describe`.
- `jj st` to see the working-copy change; `jj log --no-graph` to read history.

## Test suite

Layout under `test/`, all registered in `test/CMakeLists.txt`:
- `test/unit/` — **Catch2** pure-logic tests (no Qt event loop, no GPU): data structures,
  the pure `DirectOptimizer` (Tier-1 analytic golden), future persistence.
- `test/lifecycle/` — **QtTest** for QObject/QThread/QSignalSpy seams (the headless
  `OptimizeCoordinator`), run under `QCoreApplication` with zero GPU/display.
- `test/golden/` — golden-oracle baseline (`baseline.json`, `fem_golden.jts`,
  `fem_oracle_captured.jtak`, `calibration.txt`).
- `test/oracle/` — (planned, U6) the GPU-labeled Tier-2 oracle.

Conventions:
- **QtTest for Qt/threading seams; Catch2 for pure math.** Both register via CTest.
- **Default `headless` label must never touch GPU, VTK render, or a widget.** GPU/real-VTK
  cases go under a separate `oracle`/`gpu` label, run explicitly on a GPU machine.
- **New Qt test target gotcha:** CMake AUTOMOC does not auto-moc an included shared header,
  so add the Q_OBJECT header to the `add_executable(...)` source list (see
  `jtml_test_coordinator` in `test/CMakeLists.txt`).
- `QSignalSpy` must observe a signal on the **main/test thread**, never the worker thread
  (QTBUG-2842) — the coordinator re-emits on its own thread.

## The in-flight refactor (testability + MVVM)

- Plan: `docs/plans/2026-08-07-001-refactor-testability-mvvm-plan.md` (Units U1..U8, stable
  U-IDs; checkboxes track progress).
- Requirements: `docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md`
  (R1..R16, AE1..AE5 — normative contract).
- Oracle spec: `golden_oracle.org` (two-tier; **Tier-2 is appearance-based**: render the
  implant at the optimized pose and compare the silhouette to `Labels/`, NOT the raw pose,
  because DIRECT convergence is noisy). Baselines in `test/golden/baseline.json`.
- Current status / what's next: see `docs/handoff-2026-08-07-testability-mvvm.md`.
- **Documented solutions:** `docs/solutions/` — resolved problems and conventions (bugs,
  best practices, workflow patterns), organized by category with YAML frontmatter
  (`module`, `tags`, `problem_type`). Search it when implementing or debugging in an area
  that already has a documented learning (e.g. headless-testing / CMake-AUTOMOC
  conventions).

Architecture seams introduced so far:
- `include/core/direct_optimizer.h` / `src/core/direct_optimizer.cpp` — pure DIRECT with an
  injected `std::function<double(const Point6D&)>` cost. **Preserves the cumulative budget**
  (effective 10k/20k/30k across trunk/branch/leaf). Has call-offset + iteration/improvement
  callbacks for the production caller.
- `include/core/optimize_coordinator.h` / `src/core/optimize_coordinator.cpp` — headless
  state machine (Idle→Running→Idle) + persistent worker thread, for the GUI to bind to.

## Repo gotchas

- `src/core/CMakeLists.txt` uses `file(GLOB ...)` for headers **and** an explicit source
  list. New core `.cpp` files must be added to that explicit list (GLOB only catches headers;
  a header globbed without its impl in the target causes an AUTOMOC undefined-symbol link
  error). See `direct_optimizer.cpp`/`optimize_coordinator.cpp` entries there.
- Several files historically relied on **transitive standard includes** that used to arrive
  via `gpu/render_engine.cuh`. The data-structures decoupling (U3) removed that — always
  include what you use (`<cmath>`, `<iostream>`, `<climits>`, ...).
- Don't reify Qt-mocking wrappers (function-pointer `QFileDialog`/`QMessageBox` shims) or
  "instantiate the real `MainScreen`" characterization tests — both were rejected as low-value.
