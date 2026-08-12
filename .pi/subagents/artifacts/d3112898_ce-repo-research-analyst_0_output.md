All verification complete. I have full coverage of the 8 areas. Here is the research report.

---

# Repository Research Summary — JTML Graph Container Arc (Feasibility PoC Grounding)

**Scope:** full research run (technology, architecture, patterns, conventions) focused on the 8 named areas. No files modified.

## Technology & Infrastructure

- **Stack:** C++20 desktop app, Qt6 6.7.2 (`qt6-main`/`qt6-wayland`, pixi.toml), VTK 9.3 built against Qt6 via `vtk_installer.sh` (`_deps/vtk/`), CUDA 12.4 (`platforms = [{platform = "linux-64", cuda = "12.4"}]`), OpenCV, Eigen, torch (linked PUBLIC into `jtml_compute` but banned in the cost path — see synthesis Contradiction 5), hegel v0.11.1 (FetchContent, PBT), Catch2 3.x, QtTest.
- **Build:** pixi tasks only (`pixi run configure/build/test`; `ctest -L headless`; never bare cmake/nvcc per AGENTS.md). CMake, Ninja, `.build/` + `.build-prof/`.
- **Deployment model:** single desktop app; layered static/shared libs: `jtml_domain`, `jtml_services`, `jtml_coordinator` (STATIC), `jtml_compute` (SHARED, the only CUDA/torch lib), `jtml_view`, app. `JTA_LIBS` = `jtml_compute` single handle (root CMakeLists.txt:120).
- **VCS:** jj (Jujutsu), NOT git. AGENTS.md workflow: `jj describe -m "<scope>: <msg>"` then `jj new` per logical change.
- **API surface:** Qt signal/slot only (no REST/gRPC/GraphQL). GPU kernels in `src/compute/*.cu`. Module organization: `include/<layer>/` mirrors `src/<layer>/` (domain, services, compute, coordinator, view, app); `src/app/experimental/` holds the QML bridge layer.
- **Test layout:** `test/unit/` (Catch2 + hegel PBT twins), `test/lifecycle/` (QtTest, QObject/QThread seams), `test/golden/` (`baseline.json`, `fem_golden.jts`, `fem_oracle_captured.jtak`, `calibration.txt`), `test/oracle/` (GPU-only, label `oracle`), `test/qml/` (Qt Quick Test + qmllint gate), plus `test/HEGEL-PBT-GUIDE.md`, `test/qml_lint.cmake`.

## Architecture & Structure

### 1. `src/coordinator/optimizer_manager.cpp` — the Optimize() stage loop (current shape)

`OptimizerManager::Optimize()` (thread slot, connected to `QThread::started` in `Initialize`) runs, per frame in `img_indices_`:

| Landmark | Line (current) |
|---|---|
| Per-frame budget/call reset: `budget_ = trunk_budget; cost_function_calls_ = 0;` | 995–996 |
| `search_stage_flag_ = Trunk;` + `start_clock_`/`update_screen_clock_` | 999, 1002–1003 |
| Trunk: `trunk_manager_.InitializeActiveCostFunction` → dilate(trunk val, both cams) → `emit UpdateDilationBackground()` → `RunDirectStage(trunk_range, trunk_manager_)` → **unconditional** `DestructActiveCostFunction` | 1010–1046 |
| Branch group init (gated `enable_branch_ && number_branches > 0 && !error_occurrred_`): init → dilate(branch val) → emit — **once, outside the repeat loop** | 1052–1076 |
| Branch repeat loop: `for (branch_index < enable_branch_ * number_branches)` with `if (error_occurrred_) break;`; per repeat: `search_stage_flag_ = Branch`; `SetStartingPoint(current_optimum_location_)`; `SetSearchRange(branch_range)`; `budget_ += branch_budget`; `RunDirectStage(branch_range, branch_manager_)` | 1080–1102 |
| Leaf group init (gated `enable_leaf_ && !error_occurrred_`): init → dilate(leaf val) → emit | 1111–1132 |
| `if (sym_trap_call) CalculateSymTrap();` — between leaf emit and search | 1135 |
| Leaf search (gated `enable_leaf_ && !error_occurrred_ && !sym_trap_call`): `search_stage_flag_ = Leaf`; re-seed; `budget_ += leaf_budget`; `RunDirectStage(leaf_range, leaf_manager_)` | 1139–1156 |
| Leaf destruct — **error-gated** `if (enable_leaf_ && !error_occurrred_)` (asymmetry vs trunk's unconditional destruct) | 1158–1163 |
| Epilogue: dilate(trunk val) + emit + `emit OptimizedFrame(...)` | 1170–1189 |
| Meter site #1: `emit UpdateDisplay((clock()-start_clock_)/cost_function_calls_, ...)` end-of-frame | 1206–1211 |

**Key facts for the plan:**
- The stage *shape* is hard-coded (trunk → branch×N → leaf), but budgets/ranges come from `OptimizerSettings`; defaults in `include/domain/settings_constants.h:25,30,31,36`: `TRUNK_BUDGET=20000`, `BRANCH_BUDGET=5000`, `NUMBER_BRANCHES=2`, `Z_SEARCH_BUDGET=5000`. **The "trunk 20k → 2×branch 5k → leaf 5k" claim is the *defaults* (cumulative 20/25/30/35k), confirmed by `test/golden/baseline.json` (`budget_cumulative_effective: [20000, 25000, 30000, 35000]`).** Budget is cumulative by design: trunk resets, branch/leaf accumulate via `budget_ +=`; `RunDirectStage` reads `budget_` + `SetCallOffset(cost_function_calls_)`.
- **The three CFM members:** `trunk_manager_`, `branch_manager_`, `leaf_manager_` (header ~lines 100–102), passed **by value** into `Initialize(...)` (manager works on copies), `UploadData` per stage with per-stage GPU frame vectors (`gpu_dilated_frames_{trunk,branch,leaf}_{A,B}_`, `gpu_intensity_frames_*`, shared `gpu_edge_frames_*`, `gpu_distance_maps_`, `gpu_heatmaps_`). Dilation/dark-silhouette values derived by a parameter-name scan (optimizer_manager.cpp:237–345) — the exact block `DeriveStageCostParams` must relocate verbatim (name variants "Dilation"/"DILATION"/"dilation", ≤0→0 clamp, `DIRECT_MAHFOUZ → 3` special case, six dark-silhouette names).
- **Sym-trap path:** `opt_directive == "Sym_Trap"` → `sym_trap_call = true` → `CalculateSymTrap()` (line 1303): hard-coded `iter_val = 60` (comment says `iter_count * 3`), `create_vector_of_poses(..., 20)`, `EvaluateCostFunctionAtPoint(pose_list.at(i), 2)` (stage 2 = **leaf** manager, line ~1338–1361), 5s total sleep, writes `Results.csv`/`Results.xyz`/`Results2D.xy`, emits `onUpdateOrientationSymTrap`/`onProgressBarUpdate`. Confirms angle 04's `repeat=0` StageSpec semantics: leaf init+dilate+emit still runs, only the search is suppressed.
- **Two ms/call meter sites:** (a) 1206–1211 (end-of-frame), (b) 1282–1287 (inside `RunDirectStage`'s `SetIterationCallback`, ~30fps throttle `(clock() - update_screen_clock_) > 33`). Both emit `(clock() - start_clock_) / calls` — **CPU clock ticks per call, not ms** (Linux `CLOCKS_PER_SEC=1e6` → µs/call), while `mainscreen.cpp:4582 onUpdateDisplay` interprets it as ms/call via `div(remaining_calls / (1000.0 / iteration_speed), 60)`. This is the synthesis's Finding 11 (1000× unit error + CPU-time blind spot); mechanism verified.
- **Terminology nuance:** "costCalls/stageText" bookkeeping is **not** in the manager — the manager tracks `budget_` / `cost_function_calls_` / `search_stage_flag_` (`SearchStageFlag {Trunk=0, Branch=1, Leaf=2}`, `include/domain/metric_enum.h:9`). The `cost_calls_`/`stage_text_` names live in `OptimizerRunControllerCore` (`refreshProgress`, `StageLabel`, optimizer_run_controller_core.cpp:43–73). The synthesis's stage-bookkeeping gate (costCalls landing on 20k/25k/30k/35k) is a function of the *core's* progress mapping, which is counter-derived from the cumulative call count — the oracle seam (M12).

### 2. The driver seam — `include/coordinator/optimizer_run_driver.h` + `src/coordinator/optimizer_run_driver.cpp`

- `jta::OptimizerRunLaunch` — the **by-value** payload struct (Calibration, frames, models, `QModelIndexList`, LocationStorage, OptimizerSettings, **three by-value CostFunctionManagers**, directive, iter_count) — exactly the `Initialize` surface, packaged by the view.
- `jta::OptimizerRunDriver` interface: `Manager()`, `ThreadActive()`, `Initialize(launch, error_message)`, `Start()`, `Stop()`, `Wait()`. Fresh instance per run; controller binds **8 connects** (verified: optimizer_run_controller.cpp:357, 363, 368, 373, 380, 405, 411, 416 — finished FIRST per M6, then the 7).
- Production adapter `OptimizerManagerRunDriver`: new manager + QThread, `moveToThread`, H3 destructor contract (cooperative stop → quit + 5s bounded wait → warn + keep waiting; never delete a running thread). Initialize forwarding verbatim at lines 58–78 (synthesis cited 60–81 — trivial drift). **Verified: only ONE `new OptimizerManager` site in the repo — optimizer_run_driver.cpp:31** (synthesis's flagged verification resolves: no legacy construction paths).
- This seam is the reason the plan's Cut B stays additive: no signature changes; `OptimizerRunLaunch` gains additive fields only.

### 3. `include/domain/direct_optimizer.h` + `src/domain/direct_optimizer.cpp` — the extracted DIRECT

- Pure, Qt/VTK/CUDA-free; `std::function<double(const Point6D&)>` injected cost, invoked with the **denormalized physical** point (`DenormalizeFromCenter`). Loop: seed eval at unit center → `while ((cost_function_calls_ + call_offset_) < budget_ && !stop_requested_) { ConvexHull(); TrisectPotentiallyOptimal(); ... }` — the **cumulative** guard (offset 0 ⇒ identical to original `calls < budget`).
- `SetCallOffset(offset)`; `GetCostFunctionCalls() = cost_function_calls_ + call_offset_` (so the manager's write-back `cost_function_calls_ = opt.GetCostFunctionCalls()` keeps the running total). Improvement callback fires on new best (mirrors old UpdateOptimum); iteration callback is post-iteration (the cooperative break boundary).
- Split rule: `TrisectPotentiallyOptimal` — largest **denormalized** side (`DenormalizeRange(GetSides()).GetLargestDirection()`), one-side trisection (keep original center + two shifted), epsilon-free Jarvis gift-wrap with `slope >= highest_slope` — matches synthesis's Options-defaults mapping line-by-line (selection=Original, ε=0.0, split_rule=OneSide, L2 size, cumulative-budget guard).
- **Stale-comment flag:** direct_optimizer.h:19–20 says "trunk 10k → branch 20k → leaf 30k" — an illustrative example that does **not** match production defaults (20k/5k×2/5k → cumulative 20/25/30/35k). Minor doc inconsistency; the code itself is settings-driven.
- Ctor call sites verified 4-arg today: optimizer_manager.cpp:1243 (function def at 1234 — synthesis cited 1234 for the ctor, slight drift), oracle_test.cpp:292, optimize_coordinator.cpp:36, plus 12 in the two direct-optimizer test files.

### 4. `include/coordinator/optimizer_run_controller_core.h` — typed Directive + observation channel

- `Directive` enum: `{Single=0, All=1, Each=2, From=3, Backward=4, SymTrap=5}` — maps 1:1 to the manager's directive strings ("Single"/"All"/"Each"/"From"/"Sym_Trap"/"Backward"); shell maps strings↔enum; QML v1 always `Single` (OptimizerBridge.cpp:150, "the directive is always 'Single'", SymTrap relayed anyway at :348–351).
- Run-state machine (Idle/Running/Stopping/Completed/Error) + epoch counter (H1 stale-relay drops) + `GateInput`/`EvaluateGate` (H2 previous==current) + `ProgressBudgets`/`StageLabel`/`refreshProgress` (the counter-derived observation channel: `stageText()`, `costCalls()`, `currentMinimum()`, `progress()`) + seed lifecycle (M10a one-shot, stale guards).
- The relay observation channel is split: the **core** holds state/progress/seed logic (Qt-free); the **shell** (`optimizer_run_controller.h:150–199`) owns the by-value relay signals (`runStateChanged`, `progressChanged`, `messageRequested`, `updateDisplayRelayed`, `poseUpdated`, `optimizedFrameRelayed`, `dilationBackgroundRequested`, `orientationSymTrapUpdated`, `seedApplied`, `seedRestored`) + epoch/sender guard + thread lifecycle.

### 5. The named-registry pattern to mirror — `jta_cost_function::CostFunctionManager`

- **Registration:** `listCostFunctions()` (CostFunctionManager.cpp:343+) builds `CostFunction` instances by name with typed defaults (`instance.addParameter(Parameter<int>("Dilation", 3))`, `Parameter<double>("PoleWeight", 75)`, `Parameter<bool>(...)`) pushed into `available_cost_functions_`; ctor defaults active to `"DIRECT_DILATION"`.
- **Dispatch:** name-string if/else chains in `callActiveCostFunction` / `InitializeActiveCostFunction` / `DestructActiveCostFunction` (e.g. `active_cost_function_ == "DIRECT_DILATION"` → `costFunctionDIRECT_DILATION()`).
- **Convention:** `CostFunctionManager.cpp` carries an explicit banner (lines 8–12): "DO NOT EDIT ANYTING IN THIS FILE" (sic) — wizard-generated section; `CostFunctionManager.h` likewise marks "DO NOT EDIT" regions around the per-variant cost/init/destruct declarations and per-variant custom-variable headers. **Planning constraint:** the graph registry should *mirror* this pattern (C++-typed named data, R2) but must not require edits inside the wizard-owned regions; the stage-guard bug below sits inside the do-not-edit file.
- R2's "named registry" also has a services-layer precedent: `jta::BuildCostFunctionRegistryEntries` (`src/services/cost_function_registry.cpp`) — the shared mapping both front-ends call, pinned by `jtml_test_cost_function_registry` (golden 51-entry table).

### 6. Test layout & the planned-but-missing targets

- `test/unit/` — Catch2 + hegel PBT: `test_direct_optimizer.cpp` (+ `_properties.cpp`), `optimizer_run_controller_core_test.cpp`, `cost_function_registry_test.cpp`, etc. House rule (AGENTS.md + CMake comments): **PBT complements, never replaces** deterministic twins.
- `test/lifecycle/` — QtTest: `coordinator_test.cpp`, `optimizer_run_controller_test.cpp` (FAKE driver implementing the seam), `session_state_controller_test.cpp`.
- `test/oracle/` — GPU: `oracle_test.cpp` (**the hand-rolled cost twin at lines 286–291**: `auto cost = [&p](const Point6D& physical) -> double { p.model->SetCurrentPrimaryCameraPose(ToPose(physical)); return p.trunk->callActiveCostFunction(); }; DirectOptimizer opt(cost, SearchRange(), start, kBudget);` — exactly the RunDirectStage lambda body, monoplane); `qml_parity_check.cpp`, `segmentation_oracle_test.cpp`, render smokes. All `oracle` label, repo-root WORKING_DIRECTORY, xcb env.
- **`jtml_test_metric_semantics` + `_props` do NOT exist** — verified by grep of test/CMakeLists.txt (no `metric_semantics` anywhere). New targets follow the direct-compile pattern: `add_executable(jtml_test_X unit/foo.cpp src/.../foo.cpp)` + `target_include_directories(... ${PROJECT_SOURCE_DIR}/include)` + Catch2 (and hegel + `-Wl,-rpath,.../libhegel` + `dl` for props) + `add_test(NAME jtml.X ...)` with `LABELS "headless"` + TIMEOUT.
- `test/golden/baseline.json` confirms the canonical shape: `"stages": "trunk / 2-branch / 1-leaf"`, `budget_per_stage {trunk 20000, branch 5000, leaf 5000, number_branches 2}`, `budget_cumulative_effective [20000, 25000, 30000, 35000]`, spec_source `golden_oracle.org`.

### 7. CMake conventions

- `src/coordinator/CMakeLists.txt` — `jtml_coordinator` STATIC with an **explicit .cpp source list** (optimize_coordinator, optimizer_manager, optimizer_run_controller_core, optimizer_run_driver, optimizer_run_controller, session_state_controller) + `file(GLOB HEADER_FILES CONFIGURE_DEPENDS include/coordinator/*.h)`. **New .cpp files must be added manually (one line)** — this is exactly where `optimizer_stage_script.cpp` lands per angle 04 R2-2 (headers auto-globbed). `CMAKE_AUTOMOC ON`; comment documents the gotcha: Q_OBJECT headers must be in the source list or AUTOMOC never mocs them → undefined-vtable link error.
- test/CMakeLists.txt — same AUTOMOC gotcha for Q_OBJECT headers in test targets (e.g., `jtml_test_session_state_controller` explicitly lists `session_state_controller.h`); direct-compile pattern keeps pure targets Qt/CUDA-free; heavy targets (controller tests) need the full include set (torch/cuda/opencv/vtk) + rpath trio (`$ORIGIN/../lib`, `$CONDA_PREFIX/lib`, `${VTK_LIB_DIR}`); `vtk_module_autoinit` for VTK-rendering targets; CUDA targets need `CUDA_ARCHITECTURES native`.
- Root: `find_package(Torch REQUIRED)` at CMakeLists.txt:77; `TORCH_LIBRARIES` PUBLIC on `jtml_compute` at src/compute/CMakeLists.txt:67 (Cut F's situs, verified); unused `<ATen/ops/div_native.h>` at CostFunctionManager.h:28 (grep-verified unused).

### 8. The 7 known cost-path bugs — verified file:line homes

| # | Bug | Home (current) | Severity |
|---|---|---|---|
| 1 | **Distance-map kernel index formula:** `int i = (blockIdx.y + gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;` — `blockIdx.y` is *added* to `gridDim.x` instead of being the row multiplier (`blockIdx.y * gridDim.x`); wrong global-thread index for any 2-D launch | `src/compute/distance_map_metric.cu:27` | **high** — the "live kernel bug" (synthesis Finding 1: the critical-path re-baseline item; note the fallback `if (orig_loc < width*height)` bounds some but not all of the corruption) |
| 2 | **Stage guard always-true:** `if (stage_ != Stage::Trunk || stage_ != Stage::Branch || stage_ != Stage::Leaf) stage_ = Stage::Trunk;` — a value can't be unequal to all three, so `stage_` is always reset to Trunk | `src/compute/CostFunctionManager.cpp:46–48` | **medium** — inside the wizard "DO NOT EDIT" region; blast radius limited because downstream code largely ignores `stage_` |
| 3 | **Uninitialized `min_dist`:** `double min_dist;` then `min_dist += X_dist * pole_weight;` (UB read; also returns garbage when all three bool flags are false) | `src/compute/DD_NEW_POLE_CONSTRAINT.cpp:134` (use at 137/140/143, return at 159) | **high** |
| 4 | **Y_dist duplicates X_dist:** `Y_dist` computed from `s_x/s_y/s_z` (same vector as X_dist) instead of a perpendicular axis | `src/compute/DD_NEW_POLE_CONSTRAINT.cpp:117–120` | **medium** |
| 5 | **Sym-trap tibia x→z substitution:** `create_312_transform(x2tib, p.z_location_, p.y_location_, p.z_location_, ...)` — x translation fed `p.z_location_` | `src/compute/sym_trap_function.cpp:104–110` (arg at :106) | **medium** |
| 6 | **Mahfouz pointer guard:** `if (pixel_score_ != 0)` checks the *device pointer* (never null after allocation) instead of the denominator value `pixel_score_[0]`; the else branch is dead, so divide-by-zero is unguarded | `src/compute/implant_mahfouz_metric.cu:324` and `:446` (intensity + contour) | **medium** |
| 7 | **Curvature stub:** init calls `gpu_metrics_->AllocateCurvatureHausdorfScore(gpu_heatmaps_->at(current_frame_index_)->GetNumKeypoints());` but `costFunctionDIRECT_DILATION()` never consumes a curvature term (heatmaps uploaded, buffer allocated, score never computed); also a commented-out duplicate DistanceMapMetric at :101 | `src/compute/DIRECT_DILATION.cpp:42–43` (cost body 64–110) | **low/medium** |

## Implementation Patterns

- **Named registry + name-string dispatch** (CostFunctionManager `listCostFunctions` + if/else chains) — the pattern R2 says the graph registry mirrors.
- **Injected-cost boundary** (`DirectOptimizer::CostFunction = std::function<double(const Point6D&)>`) — the pattern BuildGpuCostAdapter keeps; three consumers converge (RunDirectStage lambda :1243–1252, oracle twin oracle_test.cpp:288–291, z-profile probe future).
- **Cumulative-budget accounting:** `SetCallOffset` + guard `(calls + offset) < budget_`; manager write-back keeps the running total; the core's progress mapping is a pure function of (budget struct, calls).
- **Direct-compile test pattern:** compile pure .cpp sources into the test target instead of linking heavy libs (jtml_test_direct_optimizer, etc.); heavy Qt/CUDA surface only where the seam demands it.
- **Signal-relay pattern:** manager signals → shell relays by value (QTBUG-2842 — QSignalSpy must observe on main/test thread); epoch + sender guard for stale relays.

## Documentation Insights

- **AGENTS.md** is current and matches observed reality (pixi tasks, jj workflow, test taxonomy, AUTOMOC gotcha, PBT house rule, headless/oracle labels).
- **Brainstorm** `docs/brainstorms/2026-08-12-optimizer-path-requirements.md`: R1 (graph = ordered stage sequence, edges implicit — stage N+1 seeds from N's optimum), R2 (C++-typed named registry, ablation.json references by name), R3 (per-stage optimizer-variant slot = DirectOptimizer::Options, default = bit-identical classic), R4 (execute through existing seams: driver + RunDirectStage lambda + GPU cost + IoU gates; no new layer, no RunBackend enum, Cut E deferred behind the falsifiable gate), R5 (schema must not preclude biplane/tiered-dilation/polish), R6 (v1 ships one graph `jtml-production`: trunk 20k → 2×branch 5k → leaf 5k, classic DIRECT, dilation 6/4/1 — the feasibility PoC), R7 (behavior-preserving: cumulative caps 20/25/30/35k, same bookkeeping/poses, oracle IoU ≥ 0.85, qml parity unchanged; asserted via the four lineage invariants: group-once dilation across repeats, per-repeat re-seed with recovered-pose *sequence* assertion, asymmetric z-leaf, frame-to-frame seed chaining). R8–R16 cover the measurement-first spine (R10: Cut 0 + meter fix early; R13: gates = headless green, oracle green, parity), the batch seam (R12: DirectOptimizer boundary stays), and parking (R14 caching out, R15/R16 parked studies).
- **Synthesis** `.panoptes/optimizer-deep-dive/synthesis.org` (Finding 7 + Angle 04 + Round-2/3 deltas) — the normative base. All substantive claims verified against current code (details below).

### Synthesis-vs-code verification (contradictions & drift)

**Confirmed exactly:** stage loop shape + cumulative-budget semantics (trunk reset :995–996; `budget_ +=` at :1096/:1151; `SetCallOffset` guard); branch-group single init/dilate/emit before the repeat loop (:1052–1076 vs loop :1080); the error-gated leaf destruct asymmetry (**substance confirmed; line cite drifted** — synthesis :1144–1148 vs actual :1158–1163); Repeat=0/Sym_Trap semantics (:1135 CalculateSymTrap between leaf emit and gated search :1139–1143); epilogue trunk-restore dilate + emit (:1170–1184); driver seam (by-value launch, verbatim forwarding, 8 binds, only adapter constructs manager — driver.cpp:31); torch situs (CMakeLists.txt:77, compute/CMakeLists.txt:67, CostFunctionManager.h:28); `BRANCH_DILATION_DECREASE` vestigial (only settings_constants.h:32); baseline.json caps [20/25/30/35k]; oracle twin at oracle_test.cpp:286–291.

**Drift (line numbers only, from the 2026-08-12 curvature-heatmap owner-fix insertion):** RunDirectStage/DirectOptimizer ctor cite :1234 → actual def :1234, ctor :1243; branch-group cites :1040–1058 → :1052–1076; per-repeat break :1074–1075 → :1083–1085; mainscreen stage-classification cite :4526–4548 → `onUpdateDisplay` at :4582, classification ~:4596+ (the ported `StageLabel` in the core matches verbatim: Trunk / Branch N / "Extra Z-Translation" / Finished).

**Minor doc inconsistencies (not contradictions):** `direct_optimizer.h:19–20` "trunk 10k → branch 20k → leaf 30k" and `oracle_test.cpp:286` "production uses 20k/25k/30k" are loose/illustrative vs the canonical 20/25/30/35k cumulative caps; "stage bookkeeping (costCalls/stageText)" terminology lives in the controller core, not the manager (manager: `budget_`/`cost_function_calls_`/`search_stage_flag_`).

## Recommendations

1. **StageScript TU placement:** `include/coordinator/optimizer_stage_script.h` + `src/coordinator/optimizer_stage_script.cpp` with `StageSpec {StageKind kind, Point6D range, unsigned int budget, unsigned int repeat, unsigned int cfm_index}` + `BuildStageScript(OptimizerSettings, QString directive)` + `DeriveStageCostParams(...)` — one added line in `src/coordinator/CMakeLists.txt`'s explicit list (headers auto-globbed). The schema maps 1:1 onto the current blocks; `{Trunk, trunk_range, trunk_budget, 1, 0}`, `{Branch, branch_range, branch_budget, number_branches, 1}` (if enabled), `{Leaf, leaf_range, leaf_budget, 1, 2}` (if enabled), and `[{Leaf, leaf_range, leaf_budget, 0, 2}]` for Sym_Trap.
2. **Cut B pins:** the four lineage invariants (R7) with assertion homes in a new multi-stage oracle; golden assertions stay verbatim; only the twin body (oracle_test.cpp:288–291) moves into `BuildGpuCostAdapter`; transcribe (don't fix) the leaf-destruct error-gating asymmetry and the epilogue emit order.
3. **New test targets** (`jtml_test_metric_semantics` + `_props` don't exist): follow the direct-compile pattern — pure TU + Catch2 (+ hegel with the libhegel rpath/dl recipe for props), `headless` label, TIMEOUT; `metric_semantics` naming continues the `jtml_test_<name>` / `jtml.<name>` convention.
4. **Do-not-edit constraint:** bug #2 sits inside CostFunctionManager.cpp's wizard banner; the stage-guard fix and the derivation relocation must not be conflated — `DeriveStageCostParams` reproduces the *manager's* parameter scan (optimizer_manager.cpp:237–345), not the CFM internals.
5. **Meter fix (Cut 4) is orthogonal but cheap:** both sites (:1206, :1282) emit ticks/call while the UI consumes ms/call (mainscreen.cpp:4582ff) — a 1000× unit error on Linux; land with Cut 0 per R10.
6. **Bug #1 (distance_map_metric.cu:27) is the critical path** per Finding 1/R8: pin → fix → exactly one single-variable re-baseline before any algorithm delta; the oracle's flat-3000 shape (oracle_test.cpp kBudget=3000) and baseline.json are the pre/post references.

---