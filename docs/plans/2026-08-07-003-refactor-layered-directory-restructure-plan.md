---
title: refactor: Restructure JTML src/ + include/ into architecture-aligned layers
type: refactor
status: active
date: 2026-08-07
origin: docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md
extends_plan: docs/plans/2026-08-07-002-refactor-mvvm-controller-oracle-expansion-plan.md
---

# Restructure JTML src/ + include/ into architecture-aligned layers

## Overview

The MVVM + compute decomposition has extracted the seams (state, services,
coordinator, pure optimizer), but `src/core/` is still a **grab-bag** — 19 mix
pure domain, headless services, a QObject coordinator, GPU-bound orchestration,
and CUDA/ML code all in one `jtml_core` static lib. `src/cost_functions/` is
really part of the compute layer (it calls `gpu/*`), and `src/gui/` holds a
QWidgets view plus the composition root.

This plan restructures the on-disk layout and CMake targets so the directory
**expresses** the layered architecture and gives a pure, dependency-light
`domain/` surface that is the natural future Rust/FFI boundary, while `view/`
stays QML-swappable later. It is **pure reorganization + path re-pointing** with
**zero runtime behavior change** — the regressions are caught by the existing
headless suite, the GPU oracle, and a manual GUI smoke (R15 / AE4 discipline).

The restructuring is done incrementally in four independently-gated cuts (U2–U5) plus two cleanup units (U1, U6), so each cut stays green and identifies the bad one.

---

## Problem Frame

`include/core` + `src/core` group unrelated layers into one target, so:
- `jtml_core` drags Qt, Torch, VTK, OpenCV, and CUDA links for files that are
  conceptually pure (making a future Rust interop surface impossible to isolate).
- A developer can't tell from the directory where a file belongs architecturally;
  `drr_interactor.h` (GUI) even sits inside `include/core`.
- `include/gpu` is the *only* reason bare `gpu_*.cuh` includes resolve
  (propagated transitively via `jtml_gpu`), which is fragile and silent.
- 75 committed in-source build artifacts (`*.a`, `*.so`, the GUI binary,
  `*_autogen/`, `Makefile`, `cmake_install.cmake`) sit under `src/` and would
  shadow any `jj file mv`.

The goal is a structure that makes "easy to work on" real: you open the dir
matching the layer you're touching, pure code is visibly separable, and the
build/test gate stays green through every cut.

---

## Requirements Trace

- R8. Decompose MainScreen so presentation vs state/command vs services are
  separated (structure should express it).
- R9. Each extraction lands with its gate green before the next cut.
- R10. MainScreen coupling trends down (reorg adds no new `ui.` refs).
- R12. No god-object characterization / no premature abstraction (a directory
  restructure must not invent new abstractions, only relocate + re-point).
- R15. Preserve behavior — no runtime change, no algorithm re-derivation.

**Origin acceptance examples:** AE4 (covers R8/R9/R10 — layer decomposition with
per-cut gates; coupling metrics still tracked). No new product behavior.

---

## Scope Boundaries

- **Pure relocation + path re-pointing.** No behavior change, no re-implementation,
  no new abstractions. The optimizer/budget/render logic is byte-identical.
- The directory **names** are the deliverable; target **names** may change only
  insofar as a layered lib split requires (new `jtml_domain`/`jtml_services`/
  `jtml_coordinator`; merge to `jtml_compute`), with the existing `JTA_LIBS`
  variable re-pointed.
- The oracle, fixtures, `WORKING_DIRECTORY` (repo root), and `.build`/pixi task
  contract are unchanged.

### Deferred to Follow-Up Work

- **Actually stripping VTK/Qt from `model` etc.** — the reorg *relocates* files
  to their target layer and re-points includes; it does **not** rewrite `model` to
  drop VTK or `stl_reader` to drop Qt. Those purity *refactors* are separate
  follow-up work (see U3 note) and are out of scope to keep this a pure reorg.
- **Re-enabling `shape_sensitivity`** — currently unbuilt; its includes rot under
  renames. Left disabled; noted as drift to fix if/when re-enabled.
- **Consolidating `stl_reader` vs `STLReader` duplicate** — flagged, not merged here.

---

## Context & Research

### Relevant Code and Patterns

- `include/{core,gui,gpu,cost_functions}/`; `src/{core,gui,gpu,cost_functions}/`
  plus `src/Study2Grid`, `src/shape_sensitivity`.
- Targets: `jtml_core` (STATIC), `jtml_gpu` (SHARED, CUDA separable), 
  `JTA_Cost_Functions` (SHARED), GUI exe `joint-track-machine-learning`,
  `Study2Grid-Cmake`, `shape-sensitivity` (unbuilt). `JTA_LIBS` set in root
  `CMakeLists.txt`.
- Every target adds `${PROJECT_SOURCE_DIR}/include` as an include root, so
  prefixed `core/x.h`, `gui/x.h`, `gpu/x.cuh`, `cost_functions/x.h` resolve.
  `jtml_gpu` additionally PUBLIC-exports `include/gpu`, making bare
  `gpu_*.cuh`/`camera_calibration.h`/`pose_matrix.h` resolve transitively.
- The `src/core/CMakeLists.txt` pattern: `file(GLOB HEADER_FILES ${HEADER_DIR}/*.h)`
  + an **explicit** `.cpp`/`.cu` list (AUTOMOC undefined-symbol trap if GLOB
  misses an impl).

### Institutional Learnings

- `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md`
  — the explicit-impl + header-GLOB discipline; Q_OBJECT headers must be listed
  target sources; per-layer gate.
- `docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md`
  — oracle runs from repo root (`WORKING_DIRECTORY`); rpath via
  `target_link_options` not `BUILD_RPATH`; keep `.build` + pixi tasks + fixtures
  untouched.

### External References

- None required — strong local patterns; this is a mechanical reorg.

---

## Key Technical Decisions

- **Split `core` first, merge `compute` second, isolate `view` last.** Ordering
  isolates the highest-risk path-prefix refactor (cut 1) before any lib/target
  restructuring, so a failure is attributable to exactly one change.
- **Phase the prefix mechanism separately from the lib split.** Cut 1 renames
  `core`→{domain,services,coordinator,compute} paths + rewrites includes while
  *keeping a single* `jtml_core` target; cut 2 breaks it into real layered libs.
  This keeps each change verifiable (path typo vs split-logic error).
- **Standardize on prefixed includes.** Convert the fragile bare-transitive
  `gpu_*.cuh`/`camera_calibration.h` reads to the new prefix (e.g.
  `compute/...`), because prefixed includes are grep-verifiable.
- **Do NOT decouple purity files during the reorg.** `model`/`stl_reader`/
  `optimizer_settings`/`calibration` are relocated to their target layer **as-is**
  (the reorg preserves behavior, R15). The actual VTK/Qt/QMetaType stripping is
  deferred (U3 note) so this stays a pure relocate+re-point.
- **Prune committed build artifacts BEFORE moving** (cleanup cut 0), so `jj file
  mv` doesn't drag stale `*.a`/`*.so`/binaries/`_autogen` dirs, and add a
  `.gitignore` guard.
- **Merging `cost_functions`+`gpu`→`compute` is acyclic** (one-way dep
  `cost→gpu`), so `JTA_LIBS` can become a single `jtml_compute` with no link-order
  break, keeping `CUDA_SEPARABLE_COMPILATION ON` and SHARED.

---

## Open Questions

### Resolved During Planning

- Where does `optimizer_manager` (Qt QObject + GPU) live? **`coordinator`** —
  it is orchestration; it stays a Qt target. `compute` holds the pure GPU primitives
  + `frame.cu`/`machine_learning_tools`/`curvature_utilities`/`cost_functions`.
- Is `domain` truly Rust-clean after the reorg? **No — not yet.** Only the 9
  truly-pure files (`data_structures_6D`, `direct_data_storage`, `direct_optimizer`,
  `sym_trap_functions`, `ambiguous_pose_processing`, `model_list_builder`,
  `optimize_intent_controller`, `pose_file_io`, `session_state`) are pure today.
  `calibration`/`location_storage` become pure only after `camera_calibration.h`
  moves out of `gpu/` (deferred decouple). The reorg's `domain/` gate is "these 9
  have zero `QObject|vtk|cuda|torch`" (achievable now); the wider pure surface is
  a follow-up.

### Deferred to Implementation

- Exact per-file mapping confirmed against the working tree at cut time (the
  research briefs give a table; re-verify each `#include`).
- Whether `shape_sensitivity` is re-enabled (currently not built).
- Exact consolidation of `stl_reader`/`STLReader` (deferred).

---

## Implementation Units

- [x] U1. **Cleanup: prune committed in-source build artifacts + gitignore** — done: pruned 1016 tracked artifacts (`src/**` Makefiles, `cmake_install.cmake`, `*_autogen/`, `*.a`/`*.so`, GUI binary) + generated `docs/html`+`docs/latex` (976); added `.gitignore` guards (`.a`, `.so`, `*_autogen/`, binary, `docs/html`, `docs/latex`); source untouched; build + headless 11/11 green

**Goal:** Delete the 75 committed non-source build artifacts under `src/` and add
a `.gitignore` guard before any move, so renames don't drag stale binaries.

**Requirements:** R9, R15 (clean base).

**Dependencies:** None.

**Files:**
- Delete: committed `src/**/*.a`, `src/**/*.so`, `src/gui/<binary>`,
  `src/**/*_autogen/`, `src/**/Makefile`, `src/**/cmake_install.cmake`
- Modify: `.gitignore` (add `*.a`, `*.so`, `*_autogen*`, build binaries,
  `src/**/Makefile`, `cmake_install.cmake`)
- Modify (record only, no code): `jj st` expected to show only this.

**Approach:**
- Inventory tracked artifacts (`jj file list` → the 75), delete the stale ones,
  add guards. Do NOT touch source.
- Regenerate/ignore `docs/html`+`docs/latex` (committed Doxygen referencing old
  paths) — decide to delete-and-ignore or leave; note not to carry into reorg.

**Patterns to follow:** repo `.gitignore` existing entries for `/build*`.

**Test expectation:** none — no behavioral change; verify `pixi run build` +
`pixi run test` (headless) still green from a clean tree.

**Verification:** `jj st` shows source files untouched; `pixi run configure &&
build && test` green; `jj file list` no longer lists `.a`/`.so`/binary/`_autogen`.

---

- [x] U2. **Cut 1 — rename `core` to layered dirs + re-point includes (single lib
  intact)**

**Goal:** Relocate `src/core`/`include/core` files into `domain/`, `services/`,
`coordinator/`, `compute/` dirs and rewrite every `core/`-prefixed include, while
**keeping a single** `jtml_core` target so the change is path-only.

**Requirements:** R8, R9, R15.

**Dependencies:** U1.

**Files:**
- Move per mapping table (research briefs): `domain/` = 9 pure files + header-only
  `metric_enum.h`, `preprocessor-defs.h`, `settings_constants.h`,
  `mainscreen_size_constants.h`, `settings_window_size_constants.h`;
  `services/` = `model`, `stl_reader`, `STLReader`, `optimizer_settings`,
  `location_storage`, `calibration.h`; `coordinator/` = `optimize_coordinator`,
  `optimizer_manager`; `compute/` = `frame.cu`, `curvature_utilities`,
  `machine_learning_tools`.
- **`drr_interactor.h` stays in core/domain paths through U2–U4 and is moved to
  `view/` ONLY in U5 (with the rest of the gui layer).** It is GUI-code (Qt + VTK
  + `gui/drr_tool.h`) and should not be relocated before the view layer exists;
  U5 owns the single move.
- Rewrite all `core/`-prefixed includes. This is ~104 sites (not ~53): the
  grep-derived count includes `mainscreen`, `test/unit/*`, `test/lifecycle/*`,
  `test/oracle/*`, `include/cost_functions/*.h` (8), **and 12 `include/gpu/*`
  headers** (`render_engine.cuh`, `gpu_model.cuh`, `camera_calibration.h`, etc.
  each `#include "core/preprocessor-defs.h"`), plus ~33 internal self-references
  inside src/core+include/core. `preprocessor-defs.h` → `domain/` is the common
  target. Do NOT touch bare-transitive `gpu_*.cuh`/`camera_calibration.h` reads
  in U2 (gpu has not merged to compute/ yet; that conversion belongs to U4).
- Modify: `src/core/CMakeLists.txt` (header GLOB dirs now span multiple new
  dirs + explicit list), remaining single `jtml_core` target; `test/CMakeLists.txt`
  (~20 re-pointed source paths).
- Docs: update `AGENTS.md`, `docs/plans/*`, `docs/solutions/*` path refs.

**Approach:**
- Do the directory rename + include rewrite + test-path re-point as **one** change;
  keep the `jtml_core` target (paths renamed, structure intact). The single
  `jtml_core` header GLOB (`include/core/*.h`) must be replaced by multiple header
  GLOBs spanning the new dirs, or by listing headers explicitly.
- **Do NOT rewrite bare `gpu_*.cuh`/`camera_calibration.h` reads to `compute/` in
  this cut** — `include/compute` does not exist until U4.
- Keep `.cu` in the (single) core target's explicit list.

**Patterns to follow:** existing `src/core/CMakeLists.txt` explicit-impl +
header-GLOB; `test/CMakeLists.txt` CUDA-free-from-source targets.

**Test scenarios:**
- Happy path: after the move, `pixi run configure` (re-runs GLOB) + `pixi run
  build` + `pixi run test` headless all green.
- Edge (regression guard): the full headless suite (domain/services/coordinator
  sources compiled directly in test targets) passes → proves every re-pointed path
  and every rewritten include is correct.
- Grep guard: assert **zero remaining `#include "core/` sites** after U2 (a
  complete rewrite, not ~53), and run the same grep across `src/shape_sensitivity`
  even though it is unbuilt. `domain/` contains no `QObject|vtk|cuda|torch` — but
  `services/` legitimately retains VTK/Qt until the deferred decouples, so the
  purity grep is scoped to `domain/` only.

**Verification:** `pixi run build` green; `pixi run test` headless green; `ctest
-L oracle` green; grep guard passes; `jj st` shows only intended moves.

---

- [x] U3. **Cut 2 — break `core` into real layered libs (`domain`/`services`/
  `coordinator`)**

**Goal:** Turn the single `jtml_core` into `jtml_domain` (STATIC, Qt/GPU-free),
`jtml_services` (STATIC, **Qt-linked** — purity decouples deferred) and
`jtml_coordinator` (STATIC, Qt). This is what makes a pure FFI surface real in
`domain/`; `services` is a dependency-direction boundary that becomes pure after
the deferred decouples, not a Qt-free lib on this cut.

**Requirements:** R8, R9.

**Dependencies:** U2.

**Files:**
- Create: `src/domain/CMakeLists.txt`, `src/services/CMakeLists.txt`,
  `src/coordinator/CMakeLists.txt` (each explicit-impl + header-GLOB; the
  coordinator lists its `Q_OBJECT` headers explicitly + links Qt).
- Modify: root `CMakeLists.txt` (+`add_subdirectory` for the new dirs), `JTA_LIBS`
  composition, `src/gui/CMakeLists.txt` + `test/CMakeLists.txt` link lines to the
  new lib names.
- **Modify `packaging/CMakeLists.txt` IN THIS CUT**: its `install(TARGETS
  jtml_gpu jtml_core JTA_Cost_Functions ...)` (line 42) references `jtml_core`,
  which U3 deletes; `install(TARGETS <name>)` of a deleted target is a
  configure-time error, so packaging must be re-pointed to the new layered lib
  names in the same cut or `pixi run configure` fails. Extend the U3 gate with an
  install/packaging step so the break is caught here.
- **State where the three compute-mapped sources compile during U3**: `frame.cu`,
  `curvature_utilities.cpp`, `machine_learning_tools.cpp` were moved to
  `src/compute/` in U2, but `src/compute/CMakeLists.txt` is not created until U4.
  The implementer must temporarily house them in the coordinator target (via
  `../compute/*.cpp`/`*.cu` paths) for U3's build to stay green; U4 moves them
  into `jtml_compute`.
- Decouple of `camera_calibration.h` out of `gpu/` is **deferred** (see follow-up);
  U3 does NOT perform it.

**Approach:**
- Three new STATIC libs. `domain` links nothing heavy (or just what its 9 files
  need). `services` links domain AND Qt (its members — `model`, `stl_reader`,
  `STLReader`, `optimizer_settings`, `location_storage`, `calibration` — retain
  VTK/Qt/QMetaType until the deferred decouples; it is NOT Qt-free on this cut).
  `coordinator` links services+domain+compute.
- Keep the `Q_OBJECT` headers (`optimize_coordinator.h`) listed as coordinator
  sources; `optimizer_manager.h` likewise.
- Grep-guard `domain` for `QObject|vtk|cuda|torch|opencv` = 0 (the FFI surface).
  Do NOT grep-guard `services` for purity — it is not pure yet.

**Patterns to follow:** `src/core/CMakeLists.txt`; `jtml_test_coordinator` Q_OBJECT
-precedent.

**Test scenarios:**
- Happy path: headless suite green against the new libs (test targets link
  `services`/`coordinator` or compile sources directly).
- Edge: `domain` has no Qt/GPU/torch/cuda/opencv (grep = 0) — the FFI surface.
  (`services` is NOT guarded for purity on this cut; it ships Qt-linked.)
- Integration: coordinator links services+domain+compute (incl. the 3
  compute-temporarily-housed sources) and compiles.
- Error: the install/packaging target does not reference the deleted `jtml_core`.

**Verification:** `pixi run build` + `pixi run test` green; grep purity gate;
`ctest -L oracle` green; `pixi run build` + an install/packaging step green.

---

- [ ] U4. **Cut 3 — merge `gpu` + `cost_functions` → `compute`(single lib)**

**Goal:** Merge `jtml_gpu` + `JTA_Cost_Functions` into one `jtml_compute` SHARED
lib (acyclic one-way dep), relocate `src/gpu`/`src/cost_functions` → `src/compute`,
and re-point `JTA_LIBS`/bare includes.

**Requirements:** R8, R9, R15 (no link-order break).

**Dependencies:** U3.

**Files:**
- Move: `src/gpu/*` + `src/cost_functions/*` → `src/compute/`; `include/gpu` +
  `include/cost_functions` → `include/compute/`.
- Modify: new `src/compute/CMakeLists.txt` (explicit `.cu`/`.cpp` + header GLOB,
  Qt+Torch+OpenCV+CUDA, `CUDA_SEPARABLE_COMPILATION ON`, SHARED, PUBLIC compute
  header dir); root `JTA_LIBS` → `jtml_compute`; `packaging/CMakeLists.txt`
  `install(TARGETS ...)`; `cost_functions/*.h` bare gpu includes → `compute/...`.
- Re-point `Study2Grid` unprefixed `gpu_*.cuh`/`render_engine.cuh`.
- **`JTA_LIBS` is set TWICE in root `CMakeLists.txt` (line 78 and line 119, the
  latter immediately before `add_subdirectory(src)`). Re-point or delete BOTH, or
  assign once before all subdirectories; list every consumer (jtml_core, gui,
  Study2Grid, shape_sensitivity, test line 208) as check-offs.**

**Approach:**
- One SHARED `jtml_compute`; update `JTA_LIBS` everywhere it's used.
- Standardize unprefixed `gpu_*.cuh` → `compute/...` (H-14) during the merge.
- Keep `.cu` nvcc-compiled in the merged target's explicit list.

**Patterns to follow:** existing `src/gpu/CMakeLists.txt` (SHARED + separable +
rpath); `src/cost_functions/CMakeLists.txt`.

**Test scenarios:**
- Integration: on a GPU machine, run `ctest -L oracle` + a GUI launch and record
  the pass (an enforced artifact, since CI runs headless only) to attest the
  merged compute lib links + renders.
- Edge: `Study2Grid` builds (its unprefixed gpu includes re-pointed).
- Error guard: no bare `gpu_*.cuh` left unresolved (grep for `include "gpu_`);
  both `set(JTA_LIBS ...)` occurrences consistent.

**Verification:** GUI `pixi run run` launches + `ctest -L oracle` green on GPU
(recorded, since CI is headless-only — this is the highest-risk cut and R15 rests
on the recorded GPU+GUI pass), `Study2Grid` builds, headless `pixi run test` green.

---

- [ ] U5. **Cut 4 — view + app isolation**

**Goal:** Move `src/gui` → `src/view` (QWidgets, QML-swappable later), move the
composition root `main.cpp` + `Study2Grid` into `src/app`, and keep `.ui`/`.qrc`/
`Resources` co-located under view.

**Requirements:** R8, R9, R15.

**Dependencies:** U4.

**Files:**
- Move: `src/gui/{about,controls,drr_tool,interactor,mainscreen,settings_control,
  viewer}.cpp` + `include/gui/*` (incl. `mainscreen.qrc`, `*.ui`,
  `include/gui/Resources/` atomically) → `src/view`/`include/view`.
- **`drr_interactor.h` moves to `view/` here** (the single owning cut — it stayed
  in core/domain paths through U2–U4).
- Move: `src/gui/main.cpp` and `src/Study2Grid` → `src/app`. `shape_sensitivity`
  stays out of scope (left disabled).
- Modify: `src/view/CMakeLists.txt` + `src/app/CMakeLists.txt` (GUI exe links
  coordinator+services+domain+compute+VTK+Torch+OpenCV+Eigen).
- Modify: root/subdir `CMakeLists.txt`, `test/CMakeLists.txt` (oracle includes stay
  repo-root); sweep `packaging/CMakeLists.txt` for stale `src/gui` path refs.

**Approach:**
- Keep `.ui`/`.qrc`/`Resources` subtree atomic (qrc `Resources/` refs are
  file-relative).
- `view` links upstream libs; `app` is the thin composition root.
- Do NOT normalize the `OpenCV_LIBS`→`OpenCV_LIBRARIES` variable in this cut — it
  is a working, non-empty OpenCV 4 variable and the change is not required for the
  gui→view move (and would add a build-config failure mode to a path-only cut).
  If consistency is later wanted, do it in a separate dedicated commit.

**Patterns to follow:** existing `src/gui/CMakeLists.txt` AUTOUIC/AUTORCC setup.

**Test scenarios:**
- Integration: headless suite green; `ctest -L oracle` green.
- Manual visual: GUI launches and loads Kneel_1 (per R16 convention).
- Edge: `.ui`/`.qrc`/`Resources` resolve from the new view dir (no missing icons).

**Verification:** GUI builds + runs (real display smoke); headless + oracle green;
`pixi run test` green.

---

- [ ] U6. **Cleanup + docs finalization**

**Goal:** Update all documentation/AGENTS references to the new paths, delete any
stale artifacts surfaced by the moves (incl. the dead `include/core/
machine_learning_tools.cpp` duplicate), and confirm the tree is clean.

**Requirements:** R9, R15.

**Dependencies:** U5.

**Files:**
- Delete: dead `include/core/machine_learning_tools.cpp` duplicate (verify no
  active `#include` references the dead basename before deleting); any stale
  `docs/html`/`docs/latex` (or add to gitignore); leftover `src/core` remnants.
- Modify: `AGENTS.md` (path conventions now `domain/services/coordinator/compute/
  view/app`), `docs/plans/...`/`docs/solutions/...` path refs, and the
  **`golden_oracle.org` embedded path refs** (`src/core/optimizer_settings.cpp`,
  `src/core/settings_constants.h` moved to `services/`) — its `WORKING_DIRECTORY`
  stays repo-root, but its authoritative-path notes text changes.

**Approach:**
- Sweep docs for `src/core`/`include/core`/`src/gui`/`src/gpu` references; update.
- Confirm `.gitignore` catches any re-emerging artifact.

**Test expectation:** none — docs/cleanup only; verify the full suite + oracle +
GUI smoke green once more.

**Verification:** no stale path refs in AGENTS/docs (grep); `pixi run build`
+`test` + `ctest -L oracle` green; `jj st` clean.

---

## System-Wide Impact

- **Interaction graph:** every `#include "core/..."` site (~104, incl. internal
  self-refs + 12 `include/gpu` headers) and every `src/core`/`include/core`
  test-target source path (~20) — all re-pointed by U2.
- **Error propagation:** none (no behavior change); a missed include is a compile
  error caught by the per-cut build gate.
- **State lifecycle risks:** none (no runtime state). Prune artifacts first (U1)
  so moves don't inherit stale binaries.
- **API surface parity:** `JTA_LIBS` variable retained; optimizer/budget/render
  logic and pose output byte-identical (R15).
- **Integration coverage:** headless unit suite (dominant), GPU oracle, and manual
  GUI smoke together cover the risk that the reorg actually changed behavior.
- **Unchanged invariants:** cumulative budget, DIRECT loop, pose semantics, oracle
  `WORKING_DIRECTORY`, `.build`/pixi tasks, `.ui`/`.qrc`/Resources co-location.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| AUTOMOC undefined-symbol from split `Q_OBJECT` headers | List `Q_OBJECT` headers explicitly in their owning target's sources (precedent `jtml_test_coordinator`); keep header-GLOB + explicit-impl discipline per dir. |
| `file(GLOB)` missing new impl in split dirs | Re-run configure after each CMakeLists change; keep explicit-impl lists. |
| Bare `gpu_*.cuh`/`camera_calibration.h` includes silently breaking after `gpu`→`compute` | Standardize to prefixed `compute/...` during U4; grep-verify no bare `gpu_` includes remain. |
| 75 committed artifacts shadowing `jj file mv` | U1 prunes + gitignores first. |
| Reorg "looks done" but behavior changed | Per-cut gate (build + headless + oracle + GUI smoke); no runtime change by construction (R15). |
| `domain` purity claim oversold | Honest gate: only the 9 truly-pure files are Rust-ready today; `calibration`/`location_storage` need the deferred `camera_calibration.h` decouple. Flagged, not hidden. |

---

## Documentation / Operational Notes

- Update `AGENTS.md` build/test/layout conventions to the new layered dirs.
- Keep the `domain/ = pure Rust-interop surface` boundary documented; note the
  deferred decouples before it grows.
- No user-visible change; medical registration core untouched.

---

## Sources & References

- **Origin document:** [`docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md`](docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md) (R8/R9/R12/R15, AE4)
- **Parent plans:** [`2026-08-07-001`](docs/plans/2026-08-07-001-refactor-testability-mvvm-plan.md), [`2026-08-07-002`](docs/plans/2026-08-07-002-refactor-mvvm-controller-oracle-expansion-plan.md)
- Related code: `src/{core,gui,gpu,cost_functions}/CMakeLists.txt`, root `CMakeLists.txt`, `test/CMakeLists.txt`
- **Compounded learnings:** `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md`, `docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md`
