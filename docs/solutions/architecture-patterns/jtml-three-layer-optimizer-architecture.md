---
title: "JTML three-layer optimizer architecture: DirectOptimizer, OptimizeCoordinator, and OptimizerManager"
date: 2026-08-25
category: docs/solutions/architecture-patterns
module: JTML optimizer / FFI seam
problem_type: architecture_pattern
component: development_workflow
severity: high
applies_when:
  - "Working on the Rust↔C++ FFI seam for the optimizer backend"
  - "Creating DirectOptimizer instances or tracing how the cost function adapter is wired"
  - "Tracing the production optimization hot path from MainScreen through to RunDirect()"
  - "Deciding where an A/B Rust backend switch fits relative to OptimizerManager"
  - "Modifying RunDirectStage call sites or budget/range parameters"
tags:
  - optimizer
  - direct-optimizer
  - ffi
  - cxx
  - production-path
  - cost-adapter
  - run-direct-stage
  - three-layer-architecture
related_components:
  - DirectOptimizer
  - OptimizerManager
  - OptimizerRunController
  - OptimizerManagerRunDriver
  - CostFunctionManager
  - BuildGpuCostAdapter
  - StageManager
---

# JTML three-layer optimizer architecture

## Context

JTML is a C++17/Qt medical-imaging optimizer that locates optimal 6-DOF joint poses via the DIRECT algorithm. The optimizer runs entirely on the CPU (the `DirectOptimizer` class), while GPU cost evaluation happens through injected lambdas — a seam designed for an A/B switch to a Rust port.

The production hot path is a single synchronous chain from the Qt UI down to the GPU cost kernel. There is exactly one production `DirectOptimizer` construction site (inside `OptimizerManager::RunDirectStage`) and one production cost adapter (`BuildGpuCostAdapter`). Everything else is test infrastructure.

The Rust port lives in `rust/direct-rs/` and uses CXX for the FFI bridge. The FFI-facing cost handle (`CppCost` in `include/domain/cost.h`) wraps a `std::function<double(const Point6D&)>` in an opaque type so Rust can call it through `evaluate()` without passing `std::function` across FFI. The C++ `DirectOptimizer` is pure CPU logic — no GPU, no Qt — making it the natural unit for the Rust port to replace.

## Guidance

### The ONE production call chain

```
MainScreen::on_optimize_button_clicked()          [view/mainscreen.cpp:4175]
  → LaunchOptimizer(Directive::Single)            [view/mainscreen.cpp:4176, :4287]
    → optimizer_run_controller_.start(req)         [view/mainscreen.cpp:4373]
      → OptimizerManagerRunDriver::Initialize()   [coordinator/optimizer_run_driver.cpp:58-78]
        → manager_->Initialize(...)               // wires 15+ args from OptimizerRunLaunch
      → driver->Start()                            [coordinator/optimizer_run_driver.cpp:80-83]
        → thread_->start()
          → Optimize()                             [coordinator/optimizer_manager.cpp:883-1215]
            // script-driven stage loop (stage_script_ built once in Initialize):
            RunDirectStage(trunk_range, trunk_manager)     [:1068]
            RunDirectStage(branch_range, branch_manager)   [:1125] ×N branches
            RunDirectStage(leaf_range, leaf_manager)       [:1174]
              → DirectOptimizer::Run()
```

### What `RunDirectStage` does (`optimizer_manager.cpp:1303-1531`)

This is the single production boundary between the C++ optimizer shell and the GPU cost kernel. It is 228 lines and does exactly seven things:

1. **Build the serial cost adapter** — `BuildGpuCostAdapter(gpu_principal_model_, calibration_, stage_manager)` at `:1315`. This returns a `std::function<double(const Point6D&)>` that captures the GPU model pointer, calibration by value, and stage manager by reference.

2. **Construct DirectOptimizer** — `DirectOptimizer opt(serial_cost, range, starting_point_, budget_, direct_options_)` at `:1317-1318`. This is the single production construction site. The optimizer is fully parameterized through its constructor.

3. **Conditionally install batch path** — `:1324-1353`. Only when `capacity_service_ != null && poolSize > 1 && !biplane && DIRECT_DILATION`. This is the U12 GPU graph execution admission gate.

4. **Set cumulative call offset** — `opt.SetCallOffset(cost_function_calls_)` at `:1485`. DirectOptimizer's loop guard runs `offset_ + its_own_count >= budget_`, so the cumulative budget (20k trunk → +5k per branch → +5k leaf) gates correctly across stages.

5. **Install improvement callback** — `:1489-1492`. Emits `UpdateOptimum` on each improvement. DirectOptimizer owns the optimum tracking internally; this is a display-only hook.

6. **Install iteration callback** — `:1499-1513`. Cooperative stop (`error_occurrred_` → `opt.Stop()`) + ~30Hz display update via `clock()`. Fires after each ConvexHull+Trisect iteration.

7. **Run guarded** — `jta::RunDirectStageGuarded(opt, &stageError)` at `:1517`. Catches coordinator abort and `std::invalid_argument` → `OptimizerError`.

After `Run()` returns, results are read back: `cost_function_calls_`, `current_optimum_location_`, `current_optimum_value_` at `:1528-1530`.

### The cost adapter (`BuildGpuCostAdapter` at `:1745-1788`)

The adapter body is:

1. Construct a `Pose` from the physical `Point6D`
2. `principal_model->SetCurrentPrimaryCameraPose(pose)` — this is the GPU state mutation
3. If biplane: `calibration.convert_Pose_A_to_Pose_B` → `SetCurrentSecondaryCameraPose`
4. `return stage_manager.callActiveCostFunction()` — triggers the GPU render and cost evaluation

Three consumers converge on this single function: the production runner, the Tier-2 oracle (`test/oracle/oracle_test.cpp:286-291`), and the z-profile probe.

### Where the FFI switch goes

**Insertion point:** `OptimizerManager::RunDirectStage` at `:1315-1318`. This is exactly three lines:

```cpp
auto serial_cost = jta::BuildGpuCostAdapter(
    gpu_principal_model_, calibration_, stage_manager);
DirectOptimizer opt(
    serial_cost, range, starting_point_, budget_, direct_options_);
```

The A/B switch replaces these two lines: instead of constructing a C++ `DirectOptimizer` with the GPU cost, you board the cost into `CppCost` (via `build_cpp_cost()`), pass it to the Rust `direct-rs` optimizer, and let Rust run the DIRECT loop. The `CppCost` handle already exists in `include/domain/cost.h:34-55` and the Rust workspace is at `rust/Cargo.toml` (CXX 1.0.199, edition 2024, workspace member `direct-rs`).

### Ownership split

| Class | Owns | Production? |
|---|---|---|
| `OptimizerManager` | `calibration_`, `direct_options_`, 3× `CostFunctionManager`, `gpu_principal_model_`, `capacity_service_`, `evaluation_executor_`, 18+ GPU frame vectors, `stage_script_`, budget/calls/optimum state | Yes — the only production optimizer driver |
| `OptimizeCoordinator` | `cost_` (stub), `range_`, `starting_point_`, `budget_`, `state_`, `worker_thread_` | No — test-only, used only in `coordinator_test.cpp` |

### Construction sites (43 total)

| Site | Count | Notes |
|---|---|---|
| `OptimizerManager::RunDirectStage` | 1 | The ONE production site |
| `OptimizeCoordinator` (test-headless) | 1 | Stub cost, never production |
| Oracle test | 1 | Real GPU cost, Tier-2 validation |
| Unit tests | 28 | `test/unit/test_direct_optimizer.cpp` |
| Lifecycle tests | 6 | `test/lifecycle/coordinator_test.cpp` |
| Test-oracle twin | 1 | Hand-rolled `BuildGpuCostAdapter` clone |

### Three stage sites with budget semantics

| Stage | Line | Range (mm) | Budget behavior | Manager |
|---|---|---|---|---|
| Trunk | `:1068` | (35,35,35,35,35,35) | Reset to 20k | `trunk_manager_` |
| Branch | `:1125` | (15,15,25,25,25,25) | `budget_ += 5k` per branch | `branch_manager_` |
| Leaf | `:1174` | (3,3,15,3,3,3) | `budget_ += 5k` | `leaf_manager_` |

Budget accumulates: 20k → 25k → 30k → 35k total cap across the run.

## Why This Matters

The optimizer architecture has exactly one clean seam — `RunDirectStage` — where the C++ DIRECT implementation is bound to the GPU cost kernel. This seam is the A/B switch point for the Rust port. Understanding this seam is critical because:

- **Wrong insertion point breaks the contract.** The budget accounting (`SetCallOffset`), the improvement/iteration callbacks, the cooperative stop, and the batch-path admission all live *outside* DirectOptimizer but *inside* RunDirectStage. Moving the Rust switch upstream or downstream of this boundary loses one or more of these invariants.

- **The cost adapter is shared across three consumers.** The production runner, the oracle, and the z-profile probe all call `BuildGpuCostAdapter`. The Rust port must board the same cost function that these consumers build — not a parallel implementation.

- **DirectOptimizer is pure CPU, no side effects.** It takes a `std::function<double(const Point6D&)>` and returns results. The Rust port can be drop-in because it only needs to implement the same algorithm over the same cost handle, with the same termination semantics (budget cap, `Stop()`, callbacks).

- **43 construction sites exist, but only 1 is production.** The Rust port targets exactly that one site. The 42 test sites remain as validation: they exercise the C++ `DirectOptimizer` directly and serve as the regression baseline for the A/B comparison.

## When to Apply

- **When implementing the Rust DIRECT optimizer port** — this document locates the exact insertion point (RunDirectStage `:1315-1318`), the cost interface (`CppCost` in `include/domain/cost.h`), and the callback/termination contracts the Rust loop must honor.

- **When modifying the optimizer stage loop** — the script-driven stage pattern (trunk→branch→leaf with cumulative budgets) is the only production path. Any changes to stage ordering, budget semantics, or the cost adapter must be tested against the three `RunDirectStage` call sites.

- **When adding new cost functions** — `BuildGpuCostAdapter` is the shared entry point. New cost functions must produce a compatible `std::function<double(const Point6D&)>` that can be boarded into `CppCost` for the Rust path.

- **When debugging optimizer hangs or incorrect results** — the cooperative stop mechanism (`error_occurrred_` → `opt.Stop()` in the iteration callback) and the cumulative budget gate (`SetCallOffset` + `budget_` comparison) are the two most likely failure modes.

## Examples

### The FFI cost handle (what Rust receives)

```cpp
// include/domain/cost.h:34-55
class CppCost {
public:
    explicit CppCost(CostFunction fn) : fn_(std::move(fn)) {}
    double evaluate(const Point6D& point) const { return fn_(point); }
private:
    CostFunction fn_;
};
```

Rust calls `evaluate()` through CXX. The concrete function is `BuildGpuCostAdapter`'s return value — set-pose, score, return double.

### What DirectOptimizer needs from its caller

The C++ `DirectOptimizer` constructor takes: cost function, range, starting point, budget, options. It exposes: `Run()`, `Stop()`, `SetCallOffset()`, `SetIterationCallback()`, `SetImprovementCallback()`, `SetBatchCost()`, and getters for calls/optimum/location. The Rust port must replicate this interface surface.

### The insertion site (3 lines to swap)

```cpp
// optimizer_manager.cpp:1315-1318 (CURRENT — C++ path)
auto serial_cost = jta::BuildGpuCostAdapter(
    gpu_principal_model_, calibration_, stage_manager);
DirectOptimizer opt(
    serial_cost, range, starting_point_, budget_, direct_options_);
```

The Rust switch boards `serial_cost` into `CppCost`, then calls the Rust DIRECT implementation instead of constructing `DirectOptimizer`. The rest of `RunDirectStage` (callbacks, offset, guarded run, result readback) stays unchanged — the Rust loop must honor the same callback contracts.

## Related

- `docs/rust-direct-rs-ffi-seam-todos.org` — remaining FFI work items (declare `build_cpp_cost` in bridge, export DirectOptimizer surface, install panic firewall)
- `docs/rust-direct-rs-ffi-handshake.org` — the full handshake design between C++ and Rust sides
- `include/domain/cost.h` — the FFI cost handle (`CppCost`) that boards `BuildGpuCostAdapter`'s output
- `rust/Cargo.toml` — Rust workspace (CXX 1.0.199, member `direct-rs`)
- `src/coordinator/optimizer_manager.cpp:1303-1531` — `RunDirectStage`, the production A/B switch point
- `src/coordinator/optimizer_manager.cpp:1745-1788` — `BuildGpuCostAdapter`, the shared cost lambda
- `src/coordinator/optimize_coordinator.cpp:36` — test-only DirectOptimizer site (stub cost)
- `docs/plans/2026-08-14-010-feat-direct-variants-capacity-launch-plan.org` — the U12 batch-path admission plan
