# Next Agent Prompt — JTML (resume here)

**Repo:** `JTML` — Qt6 (qt6-main/wayland 6.7.2) + VTK 9.3 (built against Qt6) + CUDA 12.4 + OpenCV C++20 desktop app for 2D-3D knee-implant registration (DIRECT global optimizer over a GPU cost). **VCS is `jj` (NOT git)** — owner's workflow: `jj describe -m "<scope>: <msg>"` then `jj new` per logical change. **Build via pixi only**: `pixi run configure`, `pixi run build`, `pixi run test` (headless), `ctest --test-dir .build -L oracle` (GPU). Never run bare cmake/nvcc.

## Read these first (durable record)
- `AGENTS.md` — conventions (jj, pixi, test layout, AUTOMOC/file(GLOB) gotcha, R12: no god-object characterization, no Qt-mocking).
- `docs/plans/2026-08-07-001-refactor-testability-mvvm-plan.md` — the plan (U1–U8; most done, U7 partially).
- `docs/handoff-2026-08-07-testability-mvvm.md` — status, marked COMPLETE (U1–U8) with re-scoped follow-ons.
- `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md` + `docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md` — compounded learnings (Qt6 recipe, hegel CMake `dl`+rpath, oracle vertical-flip label, conda `BUILD_RPATH` override, cumulative-budget 20k/25k/30k).
- `golden_oracle.org` + `test/golden/baseline.json` — two-tier oracle spec (Tier-1 CPU analytic in CI; Tier-2 appearance gate IoU > 0.85, measured 0.9936).

## CURRENT STATE (verified green under Qt6)
- Full build green; `pixi run test` → **8/8 headless**; `ctest -L oracle` → passes on the RTX 3090.
- Committed (recent chain, oldest→newest): ... (U6 step1 DirectOptimizer+hegel), (U6 step2 RunDirectStage rewire), (U6 Tier-2 GPU oracle), (U6 docs), (U7 pose_file_io), (U7 MainScreen strangle), (U7 session_state extraction), (**fix** review F1 stop + F2 kinematics alignment), (**refactor** remove dead DIRECT internals from OptimizerManager M-1), (docs compound tools+conventions+AGENTS).
- `SessionState` (`include/core/session_state.h`, `src/core/session_state.cpp`, `test/unit/test_session_state.cpp`) is EXTRACTED, tested (headless 8/8), and wired into `jtml_core`, but **NOT yet consumed by MainScreen** — it is a tested service with zero production consumers.

The two-part Tier-2 oracle runs a single stage / budget 3000 / frame 0 only; it does NOT exercise `RunDirectStage`'s cumulative multi-stage + biplane path end-to-end (documented gap).

## NEXT TASK — U7 MVVM: wire `SessionState` into `MainScreen`
This is the strangle continuation and resolves the code-review flag M-2 (SessionState shipped-tested-but-unused). Per the plan U7: extract app-state/command orchestration (model-list, selection, primary model, current frame, optimize intent) OUT of the 5,620-line `MainScreen` god object into headless-tested services/controllers; keep render binding in `Viewer`; shrink MainScreen (R10: line count + `ui.`-reference count trend DOWN, currently ~5620 lines / ~830 `ui.`).

Concrete first slice (behavior-preserving):
1. Add `jta::SessionState session_state_;` member to `MainScreen` (`include/gui/mainscreen.h`).
2. Sync it from widget state: model count (`loaded_models.size()`), frame count (`loaded_frames`/image-list), selection (`ui.model_list_widget->selectionModel()->selectedRows()` → rows), current frame (`ui.image_list_widget->currentRow()`) — called at the two selection-changed slots (`on_model_list_widget_itemSelectionChanged`, `on_image_list_widget_itemSelectionChanged`) and where models/frames load.
3. Replace app-state reads with the service: primary model = `session_state_.GetPrimaryModelIndex()` (was `selected[0].row()` — SAME rule: first selected row), `GetSelectedModels()`, `GetCurrentFrame()` — especially in `LaunchOptimizer` (primary model) and the pose/kinematics commands.
Keep the VIEW (colors/opacity/VTK render updates) in the slots; only the STATE moves.

**Per-layer gate (R9):** logic/service/coordinator extraction → headless unit gate; presentation-only cuts → compile + scheduled manual-visual check (no `MainScreen` characterization — R12). Verify: `pixi run build` green, `pixi run test` 8/8, `ctest -L oracle` green; do a bounded manual GUI smoke (offscreen + real display) after wiring.

## Key gotchas / decisions to preserve
- **jj only**: `jj describe` → `jj new` per change; never git.
- **Cumulative budget 20k/25k/30k** (`settings_constants`: trunk 20000, branch 5000, leaf 5000) is load-bearing — don't "fix" to per-stage in the optimizer.
- New core `.cpp` must be added to the **explicit** `src/core/CMakeLists.txt` source list (file(GLOB) only catches headers → AUTOMOC link breakage otherwise).
- Headless tests must not require a GUI/display, VTK render window, or widget. Fast compute-only CUDA is allowed; expensive fixture/render/performance gates use `oracle`/`gpu` labels.
- `SessionState` is a plain state holder, NOT an observable ViewModel — Qt Widgets has no binding (plan R12 anti-over-engineering); signal/notify lives in the QObject coordinator (`OptimizeCoordinator`).
- hegel PBT is a swappable FetchContent layer (network at configure; `dl` + rpath needed).
- Oracle label TIFFs are bottom-left y-origin → the oracle vertically flips them; run oracle from the repo root (`WORKING_DIRECTORY` set).
