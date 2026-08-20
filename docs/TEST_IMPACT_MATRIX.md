# TEST_IMPACT_MATRIX — CUDA-Graph Greedy Evaluation Executor (U2)

**Plan:** `docs/plans/2026-08-19-011-feat-cuda-graph-greedy-evaluation-executor-plan.md` (U2)
**Date:** 2026-08-19
**Status:** Frozen for U2 — any addition/removal requires re-approval as scope change. U4 must not mutate `src/compute` until this matrix is landed (see plan coherence R10).

This matrix enumerates every test that exists at U2 time and its disposition for the executor work. `retained` means no change; `retained-with-coverage` means the test stays and a new graph-aware test extends coverage without changing the old assertion; `superseded` means a new test replaces the old assertion with a named rationale; `obsolete` would require code-owner sign-off (none in this matrix — no deletions merely because internals change).

## Legend

- **retained** — no mutation, must stay green
- **retained-with-coverage** — stays green, new test adds graph path coverage
- **superseded** — new test replaces old assertion; rationale + replacement name required
- **obsolete** — genuinely obsolete; requires code-owner sign-off and risk acceptance (none here)

## Unit tests (`test/unit/*`, Catch2, `LABELS headless`)

| Test file | Disposition | Rationale / Replacement |
|---|---|---|
| `test/unit/test_data_structures.cpp` | retained | pure data structures, not touched |
| `test/unit/test_direct_data_storage_properties.cpp` | retained | pure DIRECT storage PBT, not touched |
| `test/unit/test_direct_optimizer.cpp` | retained | Tier-1 analytic golden, not touched |
| `test/unit/test_direct_optimizer_properties.cpp` | retained | PBT pure optimizer, not touched |
| `test/unit/test_direct_optimizer_batch.cpp` | **retained-with-coverage** | Tier-0 replay contract — U2 extends with ordered-replay + SetCallOffset + non-finite checks; old assertions remain |
| `test/unit/test_bank_state.cpp` | retained | Stage-1 math, now includes `graph_overhead_bytes` and 3*int counters — existing pins remain, new footprint math is additive |
| `test/unit/test_bank_binding_api.cpp` | retained | API pins for BankState bindings — EvaluationContext overloads are additive shim |
| `test/unit/test_cost_capacity_service.cpp` | retained | pure capacity math, not touched |
| `test/unit/test_cost_function.cpp` | retained | pure CostFunction registry, not touched |
| `test/unit/test_cost_function_properties.cpp` | retained | PBT CostFunction, not touched |
| `test/unit/evaluation_context_test.cpp` | **new** | U2 — EvaluationContext null-init, BankAdmission half-memory + graph_overhead, pool Checkout/Recycle (covers R4/R6) |
| `test/unit/graph_recipe_preflight_test.cpp` | **new** | U2 — GraphRecipeKey equality, registry empty preflight, Layer-C tolerance frozen artifact |
| `test/unit/test_direct_data_storage_properties.cpp` | retained | — |
| `test/unit/test_harness_smoke.cpp` | retained | — |
| `test/unit/test_hegel_smoke.cpp` | retained | — |
| `test/unit/test_location_storage.cpp` / `test_location_storage_properties.cpp` | retained | — |
| `test/unit/pose_copy_test.cpp` / `pose_copy_properties.cpp` | retained | — |
| `test/unit/edge_processor_test.cpp` / `edge_processor_properties.cpp` | retained | — |
| `test/unit/frame_headless.cpp` | retained | Frame twin, not touched |
| `test/unit/ml_orchestrator_test.cpp` | retained | — |
| `test/unit/optimizer_run_controller_core_test.cpp` | retained | — |
| `test/unit/test_model_list_builder.cpp` / `test_model_list_builder_properties.cpp` | retained | — |
| `test/unit/test_optimize_intent_controller.cpp` | retained | — |
| `test/unit/test_pose_file_io.cpp` / `test_pose_file_io_properties.cpp` | retained | — |
| `test/unit/test_session_state.cpp` / `test_session_state_properties.cpp` | retained | — |
| `test/unit/test_stage_script.cpp` / `test_stage_script_properties.cpp` | retained | — |
| `test/unit/test_sym_trap_functions.cpp` / `test_sym_trap_functions_properties.cpp` | retained | — |
| `test/unit/test_ambiguous_pose_processing.cpp` / `test_ambiguous_pose_processing_properties.cpp` | retained | — |
| `test/unit/test_calibration.cpp` / `test_calibration_properties.cpp` | retained | — |
| `test/unit/test_cost_function_properties.cpp` | retained | — |
| `test/unit/test_metric_semantics.cpp` / `test_metric_semantics_properties.cpp` | retained | — |
| `test/unit/session_controller_test.cpp` | retained | — |
| `test/unit/render_pipeline_builder_test.cpp` | retained | U4 will extend with chunk math / overflow guard, but U2 leaves as retained |
| `test/unit/study_load_controller_test.cpp` | retained | — |
| `test/unit/experimental_*` (8 files: selection, settings, file_dialog, optimizer_gate, ml_bridge, pose_bridge) | retained | UI/bridge layer, not touched by compute executor |
| `test/unit/cost_function_registry_test.cpp` | retained | — |
| `test/unit/save_last_pose_test.cpp` | retained | — |
| `test/unit/settings_service_test.cpp` / `settings_service_properties.cpp` | retained | — |
| `test/unit/session_state_controller_test.cpp` / `session_controller_test.cpp` | retained | — |

## Lifecycle tests (`test/lifecycle/*`, QtTest, `LABELS headless`)

| Test file | Disposition | Rationale |
|---|---|---|
| `test/lifecycle/coordinator_test.cpp` | retained | QThread seam, not touched |
| `test/lifecycle/list_models_test.cpp` | retained | — |
| `test/lifecycle/optimizer_run_controller_test.cpp` | **retained-with-coverage** | U6 will extend with QSignalSpy ordered replay + firstSubmission error path; U2 leaves as retained |
| `test/lifecycle/session_state_controller_test.cpp` | retained | — |

## Oracle tests (`test/oracle/*`, `LABELS oracle;gpu`, `TIMEOUT 3600`, not in headless default)

| Test file | Disposition | Rationale |
|---|---|---|
| `test/oracle/oracle_test.cpp` | retained | Tier-2 silhouette IoU≥0.85 — remains the human-visible gate |
| `test/oracle/bit_identity_test.cpp` | **retained-with-coverage** | U7 will add graph vs serial layered diff; old bit-identity baseline stays |
| `test/oracle/cost_capacity_oracle_test.cu` | retained | GPU occupancy query, not touched |
| `test/oracle/multistage_oracle_test.cpp` | **retained-with-coverage** | U7 will drive via graph-admitted DIRECT_DILATION; old staged 20k/25k/30k/35k harness stays |
| `test/oracle/z_profile_test.cpp` | retained | z-profile probe, not touched |
| `test/oracle/probe_vtk.cpp` | retained | VTK probe, not touched |
| `test/oracle/qml_parity_check.cpp` | retained | QML parity, not touched |
| `test/oracle/qml_render_smoke.cpp` | retained | QML render smoke, not touched |
| `test/oracle/render_smoke.cpp` | retained | offscreen VTK smoke, not touched |
| `test/oracle/segmentation_oracle_test.cpp` | retained | segmentation oracle, not touched |
| `test/oracle/evaluation_executor_oracle_test.cu` | **new** | U2 — oracle pool / graph overhead smoke (not in headless; `LABELS oracle;gpu`) |
| `test/oracle/graph_capture_probe_test.cu` | **new (U3)** | U3 — Global capture probe for CUB/ memcpy / atomics |
| `test/oracle/graph_recipe_direct_dilation_test.cu` | **new (U5)** | U5 — direct_dilation_monoplane capture & relaunch |
| `test/oracle/layered_correctness_test.cu` | **new (U7)** | U7 — layered A/B exact, C bounded |
| `test/oracle/graph_throughput_oracle_test.cu` | **new (U8)** | U8 — paired throughput/latency + Nsight artifact |

## QML tests (`test/qml/*`, Qt Quick Test, `LABELS headless` via qml harness)

| Suite | Disposition |
|---|---|
| `test/qml/tst_*` (PoseCell, PosesTable, SettingsPanel, StudyFlows, Theme) + fakes | retained — view layer, no compute executor touch |

## Plan 012 U1 — typed outcome, admission policy, U12 coexistence

**Plan:** `docs/plans/2026-08-20-012-feat-cuda-graph-executor-admission-plan.md` (U1). Dispositions for the U1 change only; retain-by-default rule and the frozen plan-011 rows above are unchanged. No `obsolete`/`superseded` rows.

| Test file | Disposition | Rationale |
|---|---|---|
| `test/unit/test_direct_optimizer_batch.cpp` | **retained-with-coverage** | U1 migrates the six U6 executor cases to the typed `BatchOutcome` API (same ordering/edge assertions) and adds six U1 cases: outcome kinds, `MaterializeOrderedScores`, default-deny policy, `DecideGraphAdmission` deny matrix, null-recipe U12-survival, abort propagation through `DirectOptimizer::Run` |
| `test/lifecycle/optimizer_run_controller_test.cpp` | **retained-with-coverage** | U1 updates `EvaluationExecutorGreedyOrderingMatchesSerial` to `MaterializeOrderedScores` (same ordering assertions) and adds four `RunDirectStageGuarded` slots: CoordinatorAbort / InvalidArgument / WatchdogPoisoned → stage error, and success passthrough |
| `test/oracle/layered_correctness_test.cpp` | retained | compile-only adaptation to the typed `RunBatch` return; assertion semantics unchanged |
| `test/oracle/bit_identity_test.cpp` | retained | compile-only adaptation to the typed `RunBatch` return; assertion semantics unchanged |

## Notes

- No test is marked `obsolete` or `superseded` at U2 — deletions merely because internals change are prohibited (R10). A future `superseded` row would require a named replacement + rationale and code-owner sign-off.
- U4 must not mutate `src/compute` until this matrix is landed (plan coherence R10). A failing U4 gate is `jj abandon` of the functional change only, retaining this matrix and baselines.
- `test/golden/graph_pre_registration.json` is the frozen artifact for workloads, admission, metrics, thresholds, and Layer-C tolerance — see U2 Approach and Key Decisions. Any change requires re-approval as scope change.
