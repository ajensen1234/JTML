# user tools
 │ C++ / CUDA: readseek mis-detects .h as C and .cu/.cuh as unknown. For .h, .cu, .cuh always pass
 │ language: "cpp" on digest/edit/grep/search/refs/def calls (.cpp/.hpp/.hh/.hxx are already correct).

# JTML — Agent Working Guide

JTML is a Qt6 (qt6-main/wayland 6.7.2) + VTK 9.3 built against Qt6 + CUDA 12.4 + OpenCV C++20 desktop app for 2D-3D knee-implant
registration (DIRECT global optimizer over a GPU cost function). This file captures the
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
- `test/unit/` — **Catch2** logic and fast compute-lifecycle tests (no GUI/display or VTK render):
  data structures, the pure `DirectOptimizer` (Tier-1 analytic golden), CUDA ownership seams, future persistence.
- `test/lifecycle/` — **QtTest** for QObject/QThread/QSignalSpy seams (the headless
  `OptimizeCoordinator`), run under `QCoreApplication` with zero GPU/display.
- `test/golden/` — golden-oracle baseline (`baseline.json`, `fem_golden.jts`,
  `fem_oracle_captured.jtak`, `calibration.txt`).
- `test/oracle/` — GPU-labeled oracles: Tier-2 appearance, bit-identity, layered-correctness,
  evaluation-executor, and the U8 throughput harness (plan 011). Never in the headless default.
- `test/qml/` — **Qt Quick Test** for the view layer (plan 007 U6):
  `quick_test_main` harness over the REAL `src/app/experimental/*.qml`
  sources (qrc-aliased, no drift) with injected fake bridges; headless
  (offscreen + `QT_QUICK_CONTROLS_STYLE=Material`, no VTK). The
  `jtml.qml_lint` qmllint gate (Qt 6.7.2 binary) runs from `test/`
  (`test/qml_lint.cmake`).

Conventions:
- **QtTest for Qt/threading seams; Catch2 for pure math.** Both register via CTest.
- **Prefer hegel property-based tests for extracted pure logic.** When a piece of pure,
  CUDA/Qt-free logic has invariants worth locking down (length preservation,
  collision-freedom, monotonicity, determinism), add a hegel PBT test alongside its
  deterministic Catch2 unit test — PBT complements, never replaces, the deterministic cases.
  See `test/HEGEL-PBT-GUIDE.md` for the authoring patterns, built-in generator survey, and
  how to discover the hegel API.
- **Default `headless` means no GUI/display dependency, VTK render window, or widget.** Fast compute-only CUDA tests are allowed and may initialize a GPU. Expensive fixture/render/performance gates stay under separate `oracle`/`gpu` labels.
- **New Qt test target gotcha:** CMake AUTOMOC does not auto-moc an included shared header,
  so add the Q_OBJECT header to the `add_executable(...)` source list (see
  `jtml_test_coordinator` in `test/CMakeLists.txt`).
- `QSignalSpy` must observe a signal on the **main/test thread**, never the worker thread
  (QTBUG-2842) — the coordinator re-emits on its own thread.

## Where to find information

`docs/` is the knowledge store. Fastest way in: run `ctx_index` over `docs/` once per
session, then `ctx_search` for focused snippets. It holds plans (`docs/plans/`), handoffs
(`docs/handoff-*.md`), brainstorms/requirements (`docs/brainstorms/`), and documented
solutions (`docs/solutions/` — bugs, best practices, and workflow patterns organized by
category with YAML frontmatter `module`/`tags`/`problem_type`). Search it before
implementing or debugging in a documented area.

## Current work (plan 011 — CUDA-Graph greedy evaluation executor)

- **Active plan:** `docs/plans/2026-08-19-011-feat-cuda-graph-greedy-evaluation-executor-plan.md`
  (U1..U8, real CUDA-Graph greedy evaluation executor). U1–U5 are landed and checkpointed;
  U6–U8 are deepened designs pending implementation.
- **Status / what's next:** `docs/handoff-2026-08-19-cuda-graph-greedy-evaluation-executor.md`.
  Treat the plan checkboxes + handoff as the current source of truth.
- Requirements: `docs/brainstorms/2026-08-19-cuda-graph-greedy-evaluation-executor-requirements.org`

## Prior refactor (testability + MVVM) — landed / historical

- Plan: `docs/plans/2026-08-07-001-refactor-testability-mvvm-plan.md` (U1..U8, done).
- Requirements: `docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md`
  (R1..R16, AE1..AE5 — normative contract for that refactor).
- Oracle spec: `golden_oracle.org` (two-tier; **Tier-2 is appearance-based**: render the
  implant at the optimized pose and compare the silhouette to `Labels/`, NOT the raw pose,
  because DIRECT convergence is noisy). Baselines in `test/golden/baseline.json`.

Architecture seams introduced so far:
- `include/domain/direct_optimizer.h` / `src/domain/direct_optimizer.cpp` — pure DIRECT with an
  injected `std::function<double(const Point6D&)>` cost. **Preserves the cumulative budget**
  (effective 20k/25k/30k across trunk/branch/leaf). Has call-offset + iteration/improvement
  callbacks for the production caller.
- `include/coordinator/optimize_coordinator.h` / `src/coordinator/optimize_coordinator.cpp` — headless
  state machine (Idle→Running→Idle) + persistent worker thread, for the GUI to bind to.
- Plan-011 compute layer (CUDA-graph executor): `include/compute/evaluation_context.h`
  (context + pool), `include/compute/evaluation_executor.h` (greedy `RunBatch`),
  `include/compute/graph_recipe.h` + `src/compute/graph_recipe_direct_dilation.cu`
  (per-context graph capture of the production render+metric chain) — see the plan-011
  handoff for details.

> **003 layered layout:** `src/core`+`include/core` was split into `domain/` (pure
> logic), `services/` (non-pure headless services), `coordinator/` (QObject
> orchestration), `compute/` (GPU/CUDA). U3 split the single `jtml_core` lib into
> `jtml_domain` / `jtml_services` / `jtml_coordinator` STATIC libs; U4 merged the GPU +
> cost-functions libs into the single SHARED `jtml_compute`; U5 added the STATIC
> `jtml_view` (QWidgets, QML-swappable) + moved the thin GUI composition root and
> `Study2Grid` to `src/app`. `jtml_domain` is
> the Qt/GPU-free Rust-interop surface; `jtml_services`/`jtml_coordinator` are Qt-linked
> until the deferred purity decouples. The old `src/core/` dir is gone.

## Repo gotchas

- Each layered lib (`src/{domain,services,coordinator}/CMakeLists.txt`) uses
  `file(GLOB ...)` for headers **and** an explicit `.cpp`/`.cu` source list. New `.cpp`
  files must be added to the explicit list (GLOB only catches headers; a header globbed
  without its impl in the target causes an AUTOMOC undefined-symbol link error). See
  `direct_optimizer.cpp`/`optimize_coordinator.cpp` entries there.
- Several files historically relied on **transitive standard includes** that used to arrive
  via `gpu/render_engine.cuh`. The data-structures decoupling (U3) removed that — always
  include what you use (`<cmath>`, `<iostream>`, `<climits>`, ...).
- Don't reify Qt-mocking wrappers (function-pointer `QFileDialog`/`QMessageBox` shims) or
  "instantiate the real `MainScreen`" characterization tests — both were rejected as low-value.
