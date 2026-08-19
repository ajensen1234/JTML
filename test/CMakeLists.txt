# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# JTML test harness.
#
# Conventions (from docs/plans/2026-08-07-001-refactor-testability-mvvm-plan.md):
#  - The default (`headless`) label runs with no GUI, VTK render window, or GPU.
#  - The `oracle`/`gpu` label is excluded from the default run; run it
#    explicitly on a GPU machine (U2/U6).
#  - QtTest (`Qt<major>::Test`) is used for QObject/QThread/QSignalSpy seams;
#    Catch2 is used for pure-math / data-structure tests.

# The old nfd / vtk subdirectories were dead code (commented out of the build)
# and were never wired into CTest. They are intentionally NOT registered here so
# the default suite is genuinely headless and buildable without legacy fixtures.

find_package(Catch2 REQUIRED)

# ---------------------------------------------------------------------------
# Hegel property-based testing (experiment, see test_hegel_smoke.cpp).
# hegel-cpp is a static lib pulled in via FetchContent; it is C++20 and
# runner-agnostic (we call hegel::test() inside a Catch2 TEST_CASE). If this
# integration ever headaches the build, drop these two blocks and the
# jtml.test_hegel_smoke target -- the rest of the suite does not depend on it.
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    hegel
    GIT_REPOSITORY https://github.com/hegeldev/hegel-cpp.git
    GIT_TAG v0.11.1
)
FetchContent_MakeAvailable(hegel)

add_executable(jtml_test_hegel_smoke unit/test_hegel_smoke.cpp)
# hegel-cpp needs dladdr/dlsym (libdl) which the conda-forge GCC toolchain
# does not link in by default.
target_link_libraries(jtml_test_hegel_smoke PRIVATE Catch2::Catch2WithMain hegel dl)
# libhegel_c.so is emitted into the _deps build dir at configure time; the
# conda toolchain's BUILD_RPATH is not honoured, so append the dir explicitly
# (GNU ld accumulates -rpath, keeping the pixi lib path too).
target_link_options(jtml_test_hegel_smoke PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.hegel_smoke COMMAND jtml_test_hegel_smoke)
set_tests_properties(jtml.hegel_smoke PROPERTIES LABELS "headless" TIMEOUT 300)

# Hegel property-based tests for the pure DirectOptimizer (PBT layer). Linked
# CUDA-free from the direct-optimizer + data-structure sources, exactly like
# jtml_test_direct_optimizer, but with hegel for auto-generated invariants.
add_executable(jtml_test_direct_optimizer_props
    unit/test_direct_optimizer_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/direct_optimizer.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/direct_data_storage.cpp
)
target_include_directories(jtml_test_direct_optimizer_props PRIVATE ${PROJECT_SOURCE_DIR}/include)
# Same two hegel link/rpath tweaks as the smoke target (kept next to the lib so
# the rationale is co-located for both).
target_link_libraries(jtml_test_direct_optimizer_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_direct_optimizer_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.direct_optimizer_props COMMAND jtml_test_direct_optimizer_props)
set_tests_properties(jtml.direct_optimizer_props PROPERTIES LABELS "headless" TIMEOUT 300)

# ---------------------------------------------------------------------------
# Unit tests (Catch2; pure logic, no Qt event loop)
# ---------------------------------------------------------------------------
add_executable(jtml_test_smoke unit/test_harness_smoke.cpp)
target_link_libraries(jtml_test_smoke PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.smoke COMMAND jtml_test_smoke)
set_tests_properties(jtml.smoke PROPERTIES LABELS "headless" TIMEOUT 60)

# Pure data-structure test. Compiled CUDA-free directly from the two data
# .cpp files (data_structures_6D.cpp + direct_data_storage.cpp) rather than
# linking the CUDA-heavy jtml_core, per plan U3/R11.
add_executable(jtml_test_data_structures
    unit/test_data_structures.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/direct_data_storage.cpp
)
target_include_directories(jtml_test_data_structures PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_data_structures PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.data_structures COMMAND jtml_test_data_structures)
set_tests_properties(jtml.data_structures PROPERTIES LABELS "headless" TIMEOUT 120)

# Pure DIRECT optimizer test (plan U5 / Tier-1 analytic golden). Compiled
# CUDA-free from the optimizer + data-structure sources, no jtml_core.
add_executable(jtml_test_direct_optimizer
    unit/test_direct_optimizer.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/direct_optimizer.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/direct_data_storage.cpp
)
target_include_directories(jtml_test_direct_optimizer PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_direct_optimizer PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.direct_optimizer COMMAND jtml_test_direct_optimizer)
set_tests_properties(jtml.direct_optimizer PROPERTIES LABELS "headless" TIMEOUT 300)

# Plan 010 U11: the tier-0 batch-cost REPLAY test (R12). Compiles the optimizer
# + data-structure sources CUDA-free, exactly like jtml_test_direct_optimizer.
# The batch seam is a pure-domain std::function, so the replay contract is
# fully headless-testable against recording fakes.
add_executable(jtml_test_direct_optimizer_batch
    unit/test_direct_optimizer_batch.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/direct_optimizer.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/direct_data_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/compute/evaluation_context.cpp
    ${PROJECT_SOURCE_DIR}/src/compute/evaluation_executor.cpp)
target_include_directories(jtml_test_direct_optimizer_batch PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_direct_optimizer_batch PRIVATE Catch2::Catch2WithMain CUDA::cudart)
add_test(NAME jtml.direct_optimizer_batch COMMAND jtml_test_direct_optimizer_batch)
set_tests_properties(jtml.direct_optimizer_batch PROPERTIES LABELS "headless" TIMEOUT 300)

# CostCapacityService pure capacity sizing math (plan 010 U9). The service's
# sizing functions are INLINE in the header (cost_capacity_service.cuh), so this
# headless test compiles the header only -- no CUDA link -- against injected fake
# device props. The GPU-only device/occupancy queries live in the oracle below.
add_executable(jtml_test_cost_capacity_service
    unit/test_cost_capacity_service.cpp)
target_include_directories(jtml_test_cost_capacity_service PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_cost_capacity_service PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.cost_capacity_service COMMAND jtml_test_cost_capacity_service)
set_tests_properties(jtml.cost_capacity_service PROPERTIES LABELS "headless" TIMEOUT 120)

# Plan 010 U12 Stage 1: allocation-free bank-state/write-set contract and
# pure footprint/admission arithmetic. Header-only, CUDA-free, headless.
add_executable(jtml_test_bank_state unit/test_bank_state.cpp)
target_include_directories(jtml_test_bank_state PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_bank_state PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.bank_state COMMAND jtml_test_bank_state)
set_tests_properties(jtml.bank_state PROPERTIES LABELS "headless" TIMEOUT 120)

# Plan 010 U12 Stage 2: compile-time compatibility pins for non-owning
# RenderEngine/GPUMetrics bank and stream bindings. No CUDA execution here.
add_executable(jtml_test_bank_binding_api unit/test_bank_binding_api.cpp)
target_include_directories(jtml_test_bank_binding_api PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_bank_binding_api PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.bank_binding_api COMMAND jtml_test_bank_binding_api)
set_tests_properties(jtml.bank_binding_api PROPERTIES LABELS "headless" TIMEOUT 120)

# U2: EvaluationContext pool + half-memory admission with graph overhead (headless, CUDA-free header + BankState math)
add_executable(jtml_test_evaluation_context unit/evaluation_context_test.cpp
    ${PROJECT_SOURCE_DIR}/src/compute/evaluation_context.cpp)
target_include_directories(jtml_test_evaluation_context PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${CUDA_INCLUDE_DIRS})
target_link_libraries(jtml_test_evaluation_context PRIVATE Catch2::Catch2WithMain CUDA::cudart)
add_test(NAME jtml.evaluation_context COMMAND jtml_test_evaluation_context)
set_tests_properties(jtml.evaluation_context PROPERTIES LABELS "headless" TIMEOUT 120
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# U2: GraphRecipe registry + frozen pre-registration artifact (headless, header-only)
add_executable(jtml_test_graph_recipe_preflight unit/graph_recipe_preflight_test.cpp)
target_include_directories(jtml_test_graph_recipe_preflight PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_graph_recipe_preflight PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.graph_recipe_preflight COMMAND jtml_test_graph_recipe_preflight)
set_tests_properties(jtml.graph_recipe_preflight PROPERTIES LABELS "headless" TIMEOUT 120
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# U2 oracle: EvaluationContext pool / graph overhead smoke (oracle;gpu, .cu)
add_executable(jtml_test_evaluation_executor_oracle oracle/evaluation_executor_oracle_test.cu
    ${PROJECT_SOURCE_DIR}/src/compute/evaluation_context.cpp)
target_include_directories(jtml_test_evaluation_executor_oracle PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${CUDA_INCLUDE_DIRS})
target_link_libraries(jtml_test_evaluation_executor_oracle PRIVATE Catch2::Catch2WithMain CUDA::cudart)
set_property(TARGET jtml_test_evaluation_executor_oracle PROPERTY CUDA_ARCHITECTURES native)
add_test(NAME jtml.evaluation_executor_oracle COMMAND jtml_test_evaluation_executor_oracle)
set_tests_properties(jtml.evaluation_executor_oracle PROPERTIES LABELS "oracle;gpu" TIMEOUT 3600
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# Pose / kinematics persistence service round-trip tests (plan U7, R2/R8/R9).
# Pure file-format logic; compiled CUDA/Qt-free alongside the real golden
# fixture with the data-structure source, no event loop, no GPU.
add_executable(jtml_test_pose_file_io
    unit/test_pose_file_io.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/pose_file_io.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_pose_file_io PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_pose_file_io PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.pose_file_io COMMAND jtml_test_pose_file_io)
# TEST_WORKING_DIRECTORY so the fixture test finds test/golden/fem_golden.jts.
set_tests_properties(jtml.pose_file_io PROPERTIES LABELS "headless" TIMEOUT 120
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# App-state / session orchestration tests (plan U7, R8/R9). Pure state holder;
# no Qt, no widgets, headless.
add_executable(jtml_test_session_state
    unit/test_session_state.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/session_state.cpp
)
target_include_directories(jtml_test_session_state PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_session_state PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.session_state COMMAND jtml_test_session_state)
set_tests_properties(jtml.session_state PROPERTIES LABELS "headless" TIMEOUT 120)

# Hegel property-based tests for the pure ModelListBuilder (plan U7, U10).
# Linked CUDA/Qt-free from the builder source, exactly like the direct-optimizer
# PBT target, with the same hegel link/rpath tweaks; runner-agnostic hegel::test()
# inside a Catch2 TEST_CASE.
add_executable(jtml_test_model_list_builder_props
    unit/test_model_list_builder_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/model_list_builder.cpp
)
target_include_directories(jtml_test_model_list_builder_props PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_model_list_builder_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_model_list_builder_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.model_list_builder_props COMMAND jtml_test_model_list_builder_props)
set_tests_properties(jtml.model_list_builder_props PROPERTIES LABELS "headless" TIMEOUT 300)

# Hegel PBT for the pure pose/kinematics file format (src/domain). Locks the
# frame-count-preserving round-trip invariant (see guide "The shape of a PBT").
add_executable(jtml_test_pose_file_io_props
    unit/test_pose_file_io_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/pose_file_io.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_pose_file_io_props PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_pose_file_io_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_pose_file_io_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.pose_file_io_props COMMAND jtml_test_pose_file_io_props)
set_tests_properties(jtml.pose_file_io_props PROPERTIES LABELS "headless" TIMEOUT 300)

# Hegel PBT for the pure SessionState app-state holder (src/domain). Locks the
# selection well-formedness invariant.
add_executable(jtml_test_session_state_props
    unit/test_session_state_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/session_state.cpp
)
target_include_directories(jtml_test_session_state_props PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_session_state_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_session_state_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.session_state_props COMMAND jtml_test_session_state_props)
set_tests_properties(jtml.session_state_props PROPERTIES LABELS "headless" TIMEOUT 300)

# Hegel PBT for the pure sym-trap transform helpers (src/domain): rotation_matrix
# orthonormality (R * R^T ~= I).
add_executable(jtml_test_sym_trap_functions_props
    unit/test_sym_trap_functions_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/sym_trap_functions.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_sym_trap_functions_props PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_sym_trap_functions_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_sym_trap_functions_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.sym_trap_functions_props COMMAND jtml_test_sym_trap_functions_props)
set_tests_properties(jtml.sym_trap_functions_props PROPERTIES LABELS "headless" TIMEOUT 300)

# Hegel PBT for the pure ambiguous-pose post-processing (src/domain): varus/valgus
# range + tibial selector set-membership.
add_executable(jtml_test_ambiguous_pose_props
    unit/test_ambiguous_pose_processing_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/ambiguous_pose_processing.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/sym_trap_functions.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_ambiguous_pose_props PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_ambiguous_pose_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_ambiguous_pose_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.ambiguous_pose_props COMMAND jtml_test_ambiguous_pose_props)
set_tests_properties(jtml.ambiguous_pose_props PROPERTIES LABELS "headless" TIMEOUT 300)

# Hegel PBT for the LocationStorage pose-matrix service (src/services). Needs
# include/compute on the path because services/calibration.h pulls camera_calibration.h.
add_executable(jtml_test_location_storage_props
    unit/test_location_storage_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/services/location_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_location_storage_props PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_location_storage_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_location_storage_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.location_storage_props COMMAND jtml_test_location_storage_props)
set_tests_properties(jtml.location_storage_props PROPERTIES LABELS "headless" TIMEOUT 300)

# Hegel PBT for the header-only biplane Calibration conversion (src/services). The
# test needs include/compute for camera_calibration.h; only data_structures_6D.cpp is
# compiled (calibration.h is header-only).
add_executable(jtml_test_calibration_props
    unit/test_calibration_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_calibration_props PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_calibration_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_calibration_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.calibration_props COMMAND jtml_test_calibration_props)
set_tests_properties(jtml.calibration_props PROPERTIES LABELS "headless" TIMEOUT 300)

# --- post-003 review additions: CostFunction/Parameter registry (U4-merged pure
# surface, zero headless coverage before this) + deterministic twins for the four
# PBT-only modules (house rule: PBT complements, never replaces). ---

# CostFunction parameter registry: hegel PBT (typed round-trip, rejection edges).
# Compiled CUDA/Qt-free from src/compute/CostFunction.cpp (std-only surface).
add_executable(jtml_test_cost_function_props
    unit/test_cost_function_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/compute/CostFunction.cpp
)
target_include_directories(jtml_test_cost_function_props PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_cost_function_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_cost_function_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.cost_function_props COMMAND jtml_test_cost_function_props)
set_tests_properties(jtml.cost_function_props PROPERTIES LABELS "headless" TIMEOUT 300)

# CostFunction parameter registry: deterministic twin.
add_executable(jtml_test_cost_function
    unit/test_cost_function.cpp
    ${PROJECT_SOURCE_DIR}/src/compute/CostFunction.cpp
)
target_include_directories(jtml_test_cost_function PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_cost_function PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.cost_function COMMAND jtml_test_cost_function)
set_tests_properties(jtml.cost_function PROPERTIES LABELS "headless" TIMEOUT 120)

# Deterministic twin: sym_trap_functions (complements sym_trap_functions_props).
add_executable(jtml_test_sym_trap_functions
    unit/test_sym_trap_functions.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/sym_trap_functions.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_sym_trap_functions PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_sym_trap_functions PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.sym_trap_functions COMMAND jtml_test_sym_trap_functions)
set_tests_properties(jtml.sym_trap_functions PROPERTIES LABELS "headless" TIMEOUT 120)

# Deterministic twin: ambiguous_pose_processing (complements ambiguous_pose_props).
add_executable(jtml_test_ambiguous_pose
    unit/test_ambiguous_pose_processing.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/ambiguous_pose_processing.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/sym_trap_functions.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_ambiguous_pose PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_ambiguous_pose PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.ambiguous_pose COMMAND jtml_test_ambiguous_pose)
set_tests_properties(jtml.ambiguous_pose PROPERTIES LABELS "headless" TIMEOUT 120)

# Deterministic twin: location_storage (complements location_storage_props).
add_executable(jtml_test_location_storage
    unit/test_location_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/services/location_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_location_storage PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_location_storage PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.location_storage COMMAND jtml_test_location_storage)
set_tests_properties(jtml.location_storage PROPERTIES LABELS "headless" TIMEOUT 120)

# Deterministic twin: calibration (complements calibration_props; header-only).
add_executable(jtml_test_calibration
    unit/test_calibration.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_calibration PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_calibration PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.calibration COMMAND jtml_test_calibration)
set_tests_properties(jtml.calibration PROPERTIES LABELS "headless" TIMEOUT 120)

# Tier-0 metric-semantics pins + CPU references (plan 008 U1 / the R13
# characterization pass). Every CPU reference encodes the CURRENT kernel code as
# spec (kernel-as-spec); RED pins document pre-fix bugs (U2/U4 flip them).
# Since U2 the stage-guard observable pin constructs the REAL CostFunctionManager
# (constructor + getStage() only — CPU-only, no CUDA calls at runtime) linked
# from jtml_compute, the same fixture recipe (heavy include set, rpath) as
# jtml.cost_function_registry; the sym_trap pin includes the REAL compute-layer
# transform header (compile-time CUDA/OpenCV headers only).
add_executable(jtml_test_metric_semantics
    unit/test_metric_semantics.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_metric_semantics PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${CUDA_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jtml_test_metric_semantics PRIVATE
    Catch2::Catch2WithMain
    Qt6::Core
    ${JTA_LIBS})
# The conda toolchain overrides BUILD_RPATH, so append the runtime lib dirs
# explicitly (GNU ld accumulates -rpath): jtml_compute.so lives in the build
# lib dir; torch/cuda deps in the pixi env (the .so's own RUNPATH covers
# them, this is belt-and-braces for the test binary).
target_link_options(jtml_test_metric_semantics PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib")
add_test(NAME jtml.metric_semantics COMMAND jtml_test_metric_semantics)
set_tests_properties(jtml.metric_semantics PROPERTIES LABELS "headless" TIMEOUT 300)

# Hegel PBT twin for the metric-semantics CPU references (plan 008 U1): IoU/L1/
# chamfer/distance-map/Mahfouz invariants + the CropIndexToGlobal bijectivity pin
# (corrected formula passes, the buggy src/compute/distance_map_metric.cu:27
# formula never satisfies it — the RED characterization). Pure: no CUDA/OpenCV
# headers; same hegel link/rpath recipe as the other props targets.
add_executable(jtml_test_metric_semantics_props
    unit/test_metric_semantics_properties.cpp
)
target_include_directories(jtml_test_metric_semantics_props PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_metric_semantics_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_metric_semantics_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.metric_semantics_props COMMAND jtml_test_metric_semantics_props)
set_tests_properties(jtml.metric_semantics_props PROPERTIES LABELS "headless" TIMEOUT 300)

# Plan 008 U7 (Cut A): the pure stage-script surface pins (StageScript types,
# BuildStageScript, DeriveStageCostParams, CumulativeStageCaps, the
# jtml-production graph registry). The builder TU direct-compiles against the
# real OptimizerSettings (settings_constants.h defaults) + the real
# CostFunctionManager/CostFunction linked from jtml_compute (constructor +
# parameter surface only — CPU-only, no CUDA calls at runtime — the same
# fixture recipe as jtml.cost_function_registry): the DeriveStageCostParams
# integration feeds the manager's own getActiveCostFunctionClass()
# ->getIntParameters()/getBoolParameters() vectors, proving the derivation
# consumes the exact data shape the Optimize() loop's inline scan reads
# (engine runtime dilation 6/4/1).
add_executable(jtml_test_stage_script
    unit/test_stage_script.cpp
    ${PROJECT_SOURCE_DIR}/src/coordinator/optimizer_stage_script.cpp
    ${PROJECT_SOURCE_DIR}/src/services/optimizer_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_stage_script PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${CUDA_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jtml_test_stage_script PRIVATE
    Catch2::Catch2WithMain
    Qt6::Core
    ${JTA_LIBS})
# The conda toolchain overrides BUILD_RPATH, so append the runtime lib dirs
# explicitly (GNU ld accumulates -rpath): jtml_compute.so lives in the build
# lib dir; torch/cuda deps in the pixi env (the .so's own RUNPATH covers
# them, this is belt-and-braces for the test binary).
target_link_options(jtml_test_stage_script PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib")
add_test(NAME jtml.stage_script COMMAND jtml_test_stage_script)
set_tests_properties(jtml.stage_script PROPERTIES LABELS "headless" TIMEOUT 300)

# Hegel PBT twin for the stage-script surface (plan 008 U7): budget-sum /
# order / positivity / directive-invariance / determinism invariants over
# randomized settings + the U6-corrected Sym_Trap leaf-only shape (repeat=0,
# empty caps). Pure: Qt only via the header-only QMetaType macro (the
# services OptimizerSettings ctor compiles direct); same hegel link/rpath
# recipe as the other props targets.
add_executable(jtml_test_stage_script_props
    unit/test_stage_script_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/coordinator/optimizer_stage_script.cpp
    ${PROJECT_SOURCE_DIR}/src/services/optimizer_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_stage_script_props PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_stage_script_props PRIVATE
    Catch2::Catch2WithMain hegel dl Qt6::Core)
target_link_options(jtml_test_stage_script_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.stage_script_props COMMAND jtml_test_stage_script_props)
set_tests_properties(jtml.stage_script_props PROPERTIES LABELS "headless" TIMEOUT 300)

# Hegel PBT for the DIRECT container (src/domain): cached-min consistency with the
# column tail box across random insertions.
add_executable(jtml_test_direct_data_storage_props
    unit/test_direct_data_storage_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/direct_data_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_direct_data_storage_props PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_direct_data_storage_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_direct_data_storage_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.direct_data_storage_props COMMAND jtml_test_direct_data_storage_props)
set_tests_properties(jtml.direct_data_storage_props PROPERTIES LABELS "headless" TIMEOUT 300)

# ModelListBuilder tests (plan U7, U10 / R10). Pure name/dedup + size logic;
# no Qt, no GPU, headless -- mirrors jtml.session_state registration.
add_executable(jtml_test_model_list_builder
    unit/test_model_list_builder.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/model_list_builder.cpp
)
target_include_directories(jtml_test_model_list_builder PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_model_list_builder PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.model_list_builder COMMAND jtml_test_model_list_builder)
set_tests_properties(jtml.model_list_builder PROPERTIES LABELS "headless" TIMEOUT 120)

# Optimize-intent controller tests (plan U7, R8/R9, AE4). Pure decision logic;
# no Qt, no GPU, headless -- mirrors jtml.session_state registration.
add_executable(jtml_test_optimize_intent
    unit/test_optimize_intent_controller.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/optimize_intent_controller.cpp
)
target_include_directories(jtml_test_optimize_intent PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_optimize_intent PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.optimize_intent COMMAND jtml_test_optimize_intent)
set_tests_properties(jtml.optimize_intent PROPERTIES LABELS "headless" TIMEOUT 120)

# Headless optimize-lifecycle seam (plan U4, R4/R5/R7, AE1). QtTest with a real
# QThread + worker and an injected stub cost; observed via QSignalSpy on the
# coordinator (no GPU, no display, QCoreApplication only).
find_package(Qt6 COMPONENTS Core Test REQUIRED)
add_executable(jtml_test_coordinator
    lifecycle/coordinator_test.cpp
    ${PROJECT_SOURCE_DIR}/src/coordinator/optimize_coordinator.cpp
    ${PROJECT_SOURCE_DIR}/include/coordinator/optimize_coordinator.h
    ${PROJECT_SOURCE_DIR}/src/domain/direct_optimizer.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/direct_data_storage.cpp
)
target_include_directories(jtml_test_coordinator PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_coordinator PRIVATE Qt6::Core Qt6::Test)
add_test(NAME jtml.coordinator COMMAND jtml_test_coordinator)
set_tests_properties(jtml.coordinator PROPERTIES LABELS "headless" TIMEOUT 120)

# Shared optimizer-run controller lifecycle tests (plan 006 U5, R7/R13/R15/
# R16, F1, AE1). QtTest with a FAKE OptimizerRunDriver implementing the
# drive seam (the controller depends on the narrow interface —
# OptimizerManager::Initialize is non-virtual, so the seam is the driver,
# not the manager): records Initialize args + drive calls and emits the
# manager's 7+1 signals, so the controller's by-value relays are observed
# on the controller/test thread (QTBUG-2842). Pins the full drive sequence
# (save-last-pose mirror before the gate, seed ordering, Initialize args by
# value, bind set + finished-before-Initialize), the H2 gate-input rule,
# the H1 epoch drops + threadActive Start gate (incl. the widgets
# terminal-frame -> thread-death window rejection), the M6 ghost reaping,
# the M10a seed restore, the gate-rejection statuses + severity, the
# OptimizerError -> terminal-frame unlock pin, the H3 destructor contract,
# and cooperative stop. Same heavy link recipe as
# jtml.experimental_optimizer_gate (the controller header pulls
# optimizer_manager.h -> torch/CUDA).
add_executable(jtml_test_optimizer_run_controller
    lifecycle/optimizer_run_controller_test.cpp
)
target_include_directories(jtml_test_optimizer_run_controller PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${CUDA_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS})
target_link_libraries(jtml_test_optimizer_run_controller PRIVATE
    Qt6::Core
    Qt6::Test
    jtml_coordinator
    ${REQUIRED_LIBS_QUALIFIED}
    ${JTA_LIBS}
    ${OpenCV_LIBRARIES}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES})
target_link_options(jtml_test_optimizer_run_controller PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.optimizer_run_controller
    COMMAND jtml_test_optimizer_run_controller)
set_tests_properties(jtml.optimizer_run_controller PROPERTIES
    LABELS "headless" TIMEOUT 300)

# Shared optimizer-run controller CORE tests (plan 006 U5). Ports the pure
# cases of jtml.experimental_optimizer_gate against the Qt/GPU-free core
# (the SingleModelOnly bridge policy stays in the bridge test): the gate
# Input assembly with previous == current (H2), the two rejection statuses,
# the run-state machine + epoch, the widgets stage-math progress mapping,
# and the seed lifecycle (one-shot, stale guards). Direct-compile pattern
# like jtml.optimize_intent — no Qt, no GPU.
add_executable(jtml_test_optimizer_run_controller_core
    unit/optimizer_run_controller_core_test.cpp
    ${PROJECT_SOURCE_DIR}/src/coordinator/optimizer_run_controller_core.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/optimize_intent_controller.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_optimizer_run_controller_core PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_optimizer_run_controller_core PRIVATE
    Catch2::Catch2WithMain)
add_test(NAME jtml.optimizer_run_controller_core
    COMMAND jtml_test_optimizer_run_controller_core)
set_tests_properties(jtml.optimizer_run_controller_core PROPERTIES
    LABELS "headless" TIMEOUT 120)

# Session-state controller lifecycle tests (plan 006 U6, R5/R6/R10, AE2).
# QtTest over the QObject notification shell (diff-based grouped emission,
# M9): scenarios (a)-(d) — dataset load -> datasetChanged; selection change
# -> selectionChanged with correct rows/primary/mirrors; no actual value
# change -> no signal (repeated syncs don't spam); mirrors consistent at
# emission time (state advanced before the emit — previous == current — and
# the payload's previous values are the PRE-CHANGE selection);
# ResetForDatasetClear -> mirrors reset to -1/empty + injected seed-clear
# (H5/M10b) + datasetChanged, with a pending deferred emission dropped. Plus
# the M7 run-in-flight probe and the selection-only signal split. All
# emissions are same-thread direct connections (no worker thread), so
# QSignalSpy observes the controller directly (QTBUG-2842 does not apply).
# Q_OBJECT header in the source list (AUTOMOC gotcha).
add_executable(jtml_test_session_state_controller
    lifecycle/session_state_controller_test.cpp
    ${PROJECT_SOURCE_DIR}/src/coordinator/session_state_controller.cpp
    ${PROJECT_SOURCE_DIR}/include/coordinator/session_state_controller.h
    ${PROJECT_SOURCE_DIR}/src/domain/session_state.cpp
)
target_include_directories(jtml_test_session_state_controller PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_session_state_controller PRIVATE
    Qt6::Core Qt6::Test)
add_test(NAME jtml.session_state_controller
    COMMAND jtml_test_session_state_controller)
set_tests_properties(jtml.session_state_controller PROPERTIES
    LABELS "headless" TIMEOUT 120)

# Session-state controller sync/diff logic tests (plan 006 U6). Catch2 over
# the same shell (Qt6::Core only, direct lambda-connect counting — no event
# loop): the normalization rules (out-of-range current frame -> -1, rows
# pruned + sorted), the diff-vs-stored semantics (repeated identical syncs
# are silent), the two-phase mirror contract (UpdateSession leaves the
# pre-change selection readable for the view's save-last-pose;
# CommitSelection advances the mirrors to the now-current selection, H2),
# the count-shrink pruning interplay, ResetForDatasetClear (mirrors +
# injected seed-clear), and the runInFlight probe (M7).
add_executable(jtml_test_session_state_controller_unit
    unit/session_state_controller_test.cpp
    ${PROJECT_SOURCE_DIR}/src/coordinator/session_state_controller.cpp
    ${PROJECT_SOURCE_DIR}/include/coordinator/session_state_controller.h
    ${PROJECT_SOURCE_DIR}/src/domain/session_state.cpp
)
target_include_directories(jtml_test_session_state_controller_unit PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_session_state_controller_unit PRIVATE
    Catch2::Catch2WithMain Qt6::Core)
add_test(NAME jtml.session_state_controller_unit
    COMMAND jtml_test_session_state_controller_unit)
set_tests_properties(jtml.session_state_controller_unit PROPERTIES
    LABELS "headless" TIMEOUT 120)

# Headless view-model tests (plan 004 U2, R4/R5/R14/R15). QtTest compiling the
# two list-model .cpp files directly against Qt6::Core/Qt6::Test (no widgets,
# no VTK). Q_OBJECT headers go in the add_executable source list (AUTOMOC
# gotcha); ModelListModel pulls jta::ModelListBuilder for the dedup reuse.
add_executable(jtml_test_list_models
    lifecycle/list_models_test.cpp
    ${PROJECT_SOURCE_DIR}/src/view/frame_list_model.cpp
    ${PROJECT_SOURCE_DIR}/src/view/model_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/frame_list_model.h
    ${PROJECT_SOURCE_DIR}/include/view/model_list_model.h
    ${PROJECT_SOURCE_DIR}/src/domain/model_list_builder.cpp
)
target_include_directories(jtml_test_list_models PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_list_models PRIVATE Qt6::Core Qt6::Test)
add_test(NAME jtml.list_models COMMAND jtml_test_list_models)
set_tests_properties(jtml.list_models PROPERTIES LABELS "headless" TIMEOUT 60)

# SettingsService tests (plan 004 U3, R9/R14/R15). QtCore-only: the service
# .cpp + optimizer_settings.cpp compile directly against Qt6::Core (QSettings
# redirected to a temp ini via the path/format override -- never the real
# registry). Catch2 for the parity contract + error paths; hegel PBT for the
# save -> load round-trip invariant (fractional/negative/±0.0 draws).
add_executable(jtml_test_settings_service
    unit/settings_service_test.cpp
    ${PROJECT_SOURCE_DIR}/src/services/settings_service.cpp
    ${PROJECT_SOURCE_DIR}/src/services/optimizer_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_settings_service PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_settings_service PRIVATE Catch2::Catch2WithMain Qt6::Core)
add_test(NAME jtml.settings_service COMMAND jtml_test_settings_service)
set_tests_properties(jtml.settings_service PROPERTIES LABELS "headless" TIMEOUT 120)

add_executable(jtml_test_settings_service_props
    unit/settings_service_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/services/settings_service.cpp
    ${PROJECT_SOURCE_DIR}/src/services/optimizer_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_settings_service_props PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_settings_service_props PRIVATE
    Catch2::Catch2WithMain hegel dl Qt6::Core)
target_link_options(jtml_test_settings_service_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.settings_service_props COMMAND jtml_test_settings_service_props)
set_tests_properties(jtml.settings_service_props PROPERTIES LABELS "headless" TIMEOUT 300)

# Pose-copy seam tests (plan 004 U4, R8/R13/R14/R15). The Catch2 deterministic
# twin compiles the pure seam + the real LocationStorage service (headless,
# Qt-free) so the no-image boundary-fallback chain (GetPose(-1, ...) -> the
# no-image default pose overwriting frame 0 / the last frame) is pinned
# end-to-end. Needs include/compute because services/calibration.h pulls
# compute/camera_calibration.h (same pattern as jtml.location_storage).
add_executable(jtml_test_pose_copy
    unit/pose_copy_test.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/pose_copy.cpp
    ${PROJECT_SOURCE_DIR}/src/services/location_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_pose_copy PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_pose_copy PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.pose_copy COMMAND jtml_test_pose_copy)
set_tests_properties(jtml.pose_copy PROPERTIES LABELS "headless" TIMEOUT 120)

# Hegel PBT for the pure pose-copy seam (plan 004 U4): index/boundary
# invariants (row 0, last row, single/multi selection, no frames) + the
# primary-vs-current split preservation. Pure index math -- no services/vtk/
# Qt -- with the standard hegel link/rpath recipe.
add_executable(jtml_test_pose_copy_props
    unit/pose_copy_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/pose_copy.cpp
)
target_include_directories(jtml_test_pose_copy_props PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_pose_copy_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_pose_copy_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.pose_copy_props COMMAND jtml_test_pose_copy_props)
set_tests_properties(jtml.pose_copy_props PROPERTIES LABELS "headless" TIMEOUT 300)

# EdgeProcessor tests (plan 004 U5, R7/R13/R14/R15). The processor compiles
# against the real Frame API (include/compute/frame.h) and calls only the
# existing Frame methods; the production Frame impl is compute-layer
# (src/compute/frame.cu -- nvcc + torch/CUDA-linked), so the headless targets
# compile the pure-OpenCV Frame twin (test/unit/frame_headless.cpp, mirrors
# frame.cu's edge-relevant methods against the same header) alongside
# edge_processor.cpp. OpenCV-only: no Qt, no VTK, no CUDA.
add_executable(jtml_test_edge_processor
    unit/edge_processor_test.cpp
    ${PROJECT_SOURCE_DIR}/src/services/edge_processor.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/test/unit/frame_headless.cpp
)
target_include_directories(jtml_test_edge_processor PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jtml_test_edge_processor PRIVATE
    Catch2::Catch2WithMain ${OpenCV_LIBRARIES})
add_test(NAME jtml.edge_processor COMMAND jtml_test_edge_processor)
set_tests_properties(jtml.edge_processor PROPERTIES LABELS "headless" TIMEOUT 120)

# Hegel PBT twin (plan 004 U5): the reset/restore + per-frame-consistency +
# apply-all-equals-per-frame invariants, with the standard hegel link/rpath
# recipe.
add_executable(jtml_test_edge_processor_props
    unit/edge_processor_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/services/edge_processor.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/test/unit/frame_headless.cpp
)
target_include_directories(jtml_test_edge_processor_props PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jtml_test_edge_processor_props PRIVATE
    Catch2::Catch2WithMain hegel dl ${OpenCV_LIBRARIES})
target_link_options(jtml_test_edge_processor_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.edge_processor_props COMMAND jtml_test_edge_processor_props)
set_tests_properties(jtml.edge_processor_props PROPERTIES LABELS "headless" TIMEOUT 300)

# SessionController tests (plan 004 U6, R6/R10/R13/R14). The controller .cpp
# compiles directly against the headless Frame twin (test/unit/
# frame_headless.cpp, pure OpenCV) + the real LocationStorage / Model /
# stl_reader / ModelListBuilder services: QtCore for QString/QFile/QTextStream,
# OpenCV for the Frame twin (imread), VTK for Model's vtkSTLReader (no render
# window, no GPU -- same direct-compile pattern as the other unit targets).
# The golden calibration fixture is read via a repo-root WORKING_DIRECTORY
# (like jtml.pose_file_io); image fixtures are temp PNGs written at runtime.
add_executable(jtml_test_session_controller
    unit/session_controller_test.cpp
    ${PROJECT_SOURCE_DIR}/src/services/session_controller.cpp
    ${PROJECT_SOURCE_DIR}/test/unit/frame_headless.cpp
    ${PROJECT_SOURCE_DIR}/src/services/location_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/services/model.cpp
    ${PROJECT_SOURCE_DIR}/src/services/stl_reader.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/model_list_builder.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_session_controller PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS})
target_link_libraries(jtml_test_session_controller PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    Catch2::Catch2WithMain
    ${OpenCV_LIBRARIES}
    ${VTK_LIBRARIES})
# The conda toolchain overrides BUILD_RPATH, so append the runtime lib dirs
# explicitly (GNU ld accumulates -rpath) -- same recipe as the oracle targets:
# VTK lives in _deps/vtk/install/lib(64), conda libs in the pixi env.
target_link_options(jtml_test_session_controller PRIVATE
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.session_controller COMMAND jtml_test_session_controller)
set_tests_properties(jtml.session_controller PROPERTIES LABELS "headless" TIMEOUT 120
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# StudyLoadController tests (plan 006 U7, R11/R13). The shared study-load
# controller + the real SessionController direct-compile against the headless
# Frame twin + the real LocationStorage/Model/stl_reader/ModelListBuilder
# services; the dedup seam is the real ModelListModel (QtCore-only — the
# AppendModels both views wire; the Q_OBJECT header is in the sources for
# AUTOMOC, like jtml_test_experimental_selection). Same fixture recipe as
# jtml.session_controller (repo-root WORKING_DIRECTORY).
add_executable(jtml_test_study_load_controller
    unit/study_load_controller_test.cpp
    ${PROJECT_SOURCE_DIR}/src/services/study_load_controller.cpp
    ${PROJECT_SOURCE_DIR}/src/services/session_controller.cpp
    ${PROJECT_SOURCE_DIR}/test/unit/frame_headless.cpp
    ${PROJECT_SOURCE_DIR}/src/services/location_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/services/model.cpp
    ${PROJECT_SOURCE_DIR}/src/services/stl_reader.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/model_list_builder.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/view/model_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/model_list_model.h
)
target_include_directories(jtml_test_study_load_controller PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS})
target_link_libraries(jtml_test_study_load_controller PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    Catch2::Catch2WithMain
    ${OpenCV_LIBRARIES}
    ${VTK_LIBRARIES})
target_link_options(jtml_test_study_load_controller PRIVATE
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.study_load_controller COMMAND jtml_test_study_load_controller)
set_tests_properties(jtml.study_load_controller PROPERTIES LABELS "headless" TIMEOUT 120
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# MlOrchestrator tests (plan 006 U8, R12 part/R13/F3/AE4). The shared
# segment/estimate orchestrator direct-compiles against the headless Frame
# twin (the orchestrator is plain C++/OpenCV — no Qt/torch/GPU; the injected
# SegmentOp/EstimateOp/SavePoseFn callables are the torch/CUDA seams, so the
# happy path + segment/estimate-failure scenarios are headless-testable; the
# .pt-load and real CUDA estimate paths stay manual-visual). The missing-.pt
# degradation pins live in the existing jtml.experimental_ml_bridge suite
# (kept green); the stale-seed drop is the run controller's U5 guard
# (optimizer_run_controller_core_test). Same fixture recipe as
# jtml.session_controller (repo-root WORKING_DIRECTORY).
add_executable(jtml_test_ml_orchestrator
    unit/ml_orchestrator_test.cpp
    ${PROJECT_SOURCE_DIR}/src/services/ml_orchestrator.cpp
    ${PROJECT_SOURCE_DIR}/test/unit/frame_headless.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_ml_orchestrator PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jtml_test_ml_orchestrator PRIVATE
    Catch2::Catch2WithMain
    ${OpenCV_LIBRARIES})
add_test(NAME jtml.ml_orchestrator COMMAND jtml_test_ml_orchestrator)
set_tests_properties(jtml.ml_orchestrator PROPERTIES LABELS "headless" TIMEOUT 120
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# StudyBridge + delegate-selection contract tests (plan 005 U4, R3/R17). The
# bridge (AppBridge/StudyBridge/DelegateSelection/ExperimentalScene)
# direct-compiles against the headless Frame twin + the real SessionController
# + LocationStorage/Model/stl_reader/ModelListBuilder services + the
# direct-compiled FrameListModel/ModelListModel: pins the three-action load
# flow on the real Kneel_1 fixtures, the models-before-calibration guard, the
# typed calibration-error mapping, partial-load + dataset-replace semantics,
# and the delegate selection contract (current frame, multi-select set,
# primary = first selected, empty states) incl. the SessionState mirror.
# Same fixture recipe as jtml.session_controller (repo-root WORKING_DIRECTORY).
add_executable(jtml_test_experimental_selection
    unit/experimental_selection_test.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/AppBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/AppBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/MlBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/MlBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/PoseBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/PoseBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/OptimizerBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/OptimizerBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/SettingsBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/SettingsBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/StudyBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/StudyBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/services/study_load_controller.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/DelegateSelection.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/DelegateSelection.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalSession.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalScene.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalScene.cpp
    ${PROJECT_SOURCE_DIR}/src/services/session_controller.cpp
    ${PROJECT_SOURCE_DIR}/test/unit/frame_headless.cpp
    ${PROJECT_SOURCE_DIR}/src/services/location_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/services/model.cpp
    ${PROJECT_SOURCE_DIR}/src/services/stl_reader.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/model_list_builder.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/session_state.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/services/settings_service.cpp
    ${PROJECT_SOURCE_DIR}/src/services/optimizer_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/view/frame_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/frame_list_model.h
    ${PROJECT_SOURCE_DIR}/src/view/model_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/model_list_model.h
)
target_include_directories(jtml_test_experimental_selection PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${PROJECT_SOURCE_DIR}/src/app/experimental
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
    ${TORCH_INCLUDE_DIRS})
target_link_libraries(jtml_test_experimental_selection PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    Catch2::Catch2WithMain
    ${OpenCV_LIBRARIES}
    ${VTK_LIBRARIES}
    jtml_coordinator
    ${JTA_LIBS})
target_link_options(jtml_test_experimental_selection PRIVATE
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}"
    "-Wl,-rpath,$ORIGIN/../lib")
add_test(NAME jtml.experimental_selection COMMAND jtml_test_experimental_selection)
set_tests_properties(jtml.experimental_selection PROPERTIES LABELS "headless" TIMEOUT 120
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# SettingsBridge + replicated registry mapping tests (plan 005 U5, R4/R5/
# R10/R17). The bridge direct-compiles against the real SettingsService
# (temp-ini isolated, never the real registry) + the real
# CostFunctionManager/CostFunction linked from jtml_compute (constructor +
# parameter surface only — CPU-only, no CUDA calls at runtime): pins the
# cost-variant selection, the save->load round-trip, the lossless-double
# invariant, and the registry-mapping parity pin vs the widgets golden
# fixture embedded in the test. The heavy include set mirrors
# SettingsBridge.cpp's TU (CostFunctionManager.h pulls torch/cuda/opencv
# headers).
add_executable(jtml_test_experimental_settings
    unit/experimental_settings_test.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/SettingsBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/SettingsBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/services/cost_function_registry.cpp
    ${PROJECT_SOURCE_DIR}/src/services/settings_service.cpp
    ${PROJECT_SOURCE_DIR}/src/services/optimizer_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_experimental_settings PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${PROJECT_SOURCE_DIR}/src/app/experimental
    ${CUDA_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jtml_test_experimental_settings PRIVATE
    Catch2::Catch2WithMain
    Qt6::Core
    ${JTA_LIBS})
# The conda toolchain overrides BUILD_RPATH, so append the runtime lib dirs
# explicitly (GNU ld accumulates -rpath): jtml_compute.so lives in the build
# lib dir; torch/cuda deps in the pixi env (the .so's own RUNPATH covers
# them, this is belt-and-braces for the test binary).
target_link_options(jtml_test_experimental_settings PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib")
add_test(NAME jtml.experimental_settings COMMAND jtml_test_experimental_settings)
set_tests_properties(jtml.experimental_settings PROPERTIES LABELS "headless" TIMEOUT 300)

# FileDialogBridge directory-memory tests (plan 007 review round, AC1).
# QtCore + QtWidgets only (QFileDialog lives in the .cpp; the tests exercise
# only the QSettings helpers — XDG_CONFIG_HOME is redirected to a temp dir,
# never the user's real config). Q_OBJECT header listed for AUTOMOC (repo
# gotcha).
add_executable(jtml_test_experimental_file_dialog
    unit/experimental_file_dialog_test.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/FileDialogBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/FileDialogBridge.cpp
)
target_include_directories(jtml_test_experimental_file_dialog PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/app/experimental
)
target_link_libraries(jtml_test_experimental_file_dialog PRIVATE
    Catch2::Catch2WithMain
    Qt6::Core
    Qt6::Widgets
)
add_test(NAME jtml.experimental_file_dialog
    COMMAND jtml_test_experimental_file_dialog)
set_tests_properties(jtml.experimental_file_dialog PROPERTIES
    LABELS "headless" TIMEOUT 120)

# Shared cost-function registry mapping tests (plan 006 U1, R8/AE3). The
# shared jta::BuildCostFunctionRegistryEntries (services layer — the one
# mapping both front-ends call) direct-compiles against the real
# SettingsService (temp-ini isolated, never the real registry) + the real
# CostFunctionManager/CostFunction linked from jtml_compute (constructor +
# parameter surface only — CPU-only, no CUDA calls at runtime): pins the
# golden 51-entry table (17 per stage, %.17g exact IEEE-754 round-trip) for
# the widgets first-run manager configuration, the registry write/read-back
# of the exact same entries, and the no-stage/empty-manager robustness. Same
# fixture recipe (heavy include set, rpath) as jtml.experimental_settings.
add_executable(jtml_test_cost_function_registry
    unit/cost_function_registry_test.cpp
    ${PROJECT_SOURCE_DIR}/src/services/cost_function_registry.cpp
    ${PROJECT_SOURCE_DIR}/src/services/settings_service.cpp
    ${PROJECT_SOURCE_DIR}/src/services/optimizer_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_cost_function_registry PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${CUDA_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jtml_test_cost_function_registry PRIVATE
    Catch2::Catch2WithMain
    Qt6::Core
    ${JTA_LIBS})
# The conda toolchain overrides BUILD_RPATH, so append the runtime lib dirs
# explicitly (GNU ld accumulates -rpath): jtml_compute.so lives in the build
# lib dir; torch/cuda deps in the pixi env (the .so's own RUNPATH covers
# them, this is belt-and-braces for the test binary).
target_link_options(jtml_test_cost_function_registry PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib")
add_test(NAME jtml.cost_function_registry COMMAND jtml_test_cost_function_registry)
set_tests_properties(jtml.cost_function_registry PROPERTIES LABELS "headless" TIMEOUT 300)

# Save-last-pose shared-core tests (plan 006 U3, R10 part/R13, H4). The
# parameterized jta::SaveLastPoseToStorage (services layer — the one core all
# four save-last-pose call sites drive) direct-compiles against the real
# LocationStorage + the header-only Calibration: pins the four-row call-site
# table (widgets canonical / camera-A inline / camera-B inline / QML mirror),
# the no-previous-selection no-op, the convert-rule matrix, the
# out-of-range/zero pose-source skip, and the pre-cut storage-dump
# equivalence (characterization oracle). Pure logic: no Qt, no GPU.
add_executable(jtml_test_save_last_pose
    unit/save_last_pose_test.cpp
    ${PROJECT_SOURCE_DIR}/src/services/save_last_pose.cpp
    ${PROJECT_SOURCE_DIR}/src/services/location_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_save_last_pose PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_save_last_pose PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.save_last_pose COMMAND jtml_test_save_last_pose)
set_tests_properties(jtml.save_last_pose PROPERTIES LABELS "headless" TIMEOUT 120)

# Render-pipeline builder pure-config tests (plan 006 U4, R9/AE4). The
# shared widget-free VTK pipeline recipe (jta::render_pipeline, services
# layer) is the ONE recipe both Viewer and QmlVtkRenderer drive — the
# widget-free, value-parameterized extraction (R13 per-cut-type: the render
# smokes are the behavior diff for the VTK-touching calls). The .cpp
# direct-compiles against VTK + OpenCV (same pattern as
# jtml.session_controller; no render window, no GPU): pins the pure
# mat -> import-parameter derivation — gray vs color channel count, extent
# from cols/rows, the spacing/origin recipe constants, and the
# empty/zero-size degenerate (valid params, no crash).
add_executable(jtml_test_render_pipeline_builder
    unit/render_pipeline_builder_test.cpp
    ${PROJECT_SOURCE_DIR}/src/services/render_pipeline_builder.cpp
)
target_include_directories(jtml_test_render_pipeline_builder PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jtml_test_render_pipeline_builder PRIVATE
    Catch2::Catch2WithMain
    ${VTK_LIBRARIES}
    ${OpenCV_LIBRARIES})
# The conda toolchain overrides BUILD_RPATH, so append the runtime lib dirs
# explicitly (GNU ld accumulates -rpath) — same recipe as
# jtml.session_controller: VTK lives in _deps/vtk/install/lib(64), conda libs
# in the pixi env.
target_link_options(jtml_test_render_pipeline_builder PRIVATE
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.render_pipeline_builder COMMAND jtml_test_render_pipeline_builder)
set_tests_properties(jtml.render_pipeline_builder PROPERTIES LABELS "headless" TIMEOUT 120)

# OptimizerBridge QML-policy pins (plan 005 U6 + 006 U5, R6/R13). The
# bridge direct-compiles against the U4/U5 bridges + the app-owned dataset
# and links jtml_coordinator (the REAL seams live there — OptimizerRunController
# + OptimizerManager symbols are pulled through the coordinator) — but never
# touches the GPU: every rejection happens BEFORE any driver/manager/thread
# is created, so run()'s rejection paths are headless-safe. Pins: the
# SingleModelOnly pre-check (after the shared gate's first guard — a QML
# policy, NOT in the controller), the run() rejection message paths against
# the real study fixture (state untouched, no driver created), the stop()
# no-op, and the buildGateInput assembly (previous == current, H2). The
# shared-core cases (gate evaluation, state machine, progress mapping,
# OptimizedFrame handling) live in jtml.optimizer_run_controller_core +
# jtml.optimizer_run_controller (plan 006 U5 port). Same fixture recipe as
# jtml.experimental_selection (repo-root WORKING_DIRECTORY).
add_executable(jtml_test_experimental_optimizer_gate
    unit/experimental_optimizer_gate_test.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/OptimizerBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/OptimizerBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/AppBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/AppBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/MlBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/MlBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/PoseBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/PoseBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/SettingsBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/SettingsBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/StudyBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/StudyBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/services/study_load_controller.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/DelegateSelection.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/DelegateSelection.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalSession.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalScene.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalScene.cpp
    ${PROJECT_SOURCE_DIR}/test/unit/frame_headless.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/session_state.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/services/optimizer_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/view/frame_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/frame_list_model.h
    ${PROJECT_SOURCE_DIR}/src/view/model_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/model_list_model.h
)
target_include_directories(jtml_test_experimental_optimizer_gate PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${PROJECT_SOURCE_DIR}/src/app/experimental
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
    ${TORCH_INCLUDE_DIRS})
target_link_libraries(jtml_test_experimental_optimizer_gate PRIVATE
    Catch2::Catch2WithMain
    jtml_coordinator
    ${REQUIRED_LIBS_QUALIFIED}
    ${JTA_LIBS}
    ${OpenCV_LIBRARIES}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES})
target_link_options(jtml_test_experimental_optimizer_gate PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.experimental_optimizer_gate
    COMMAND jtml_test_experimental_optimizer_gate)
set_tests_properties(jtml.experimental_optimizer_gate PROPERTIES
    LABELS "headless" TIMEOUT 300
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# MlBridge degradation + seed tests (plan 005 U7, R8/R16/AE4). The bridge
# direct-compiles against the U4/U5/U6 bridges + the app-owned dataset and
# links jtml_coordinator (the REAL seams live there — SegmentationController/
# ImplantEstimator symbols are pulled from jtml_services through the
# coordinator's PUBLIC link) — but never touches the GPU: every torch/GPU
# entry in MlBridge is guarded behind the availability flags, so the
# missing-model paths are headless-safe (the torch/GPU paths themselves are
# manual-visual under xcb / oracle-arbitrated by U9). Pins: the .pt path
# surface (env fallback JTML_SEG_PT/JTML_FEM_ESTIMATE_PT, file://
# normalization, availability flags), the AE4 degradation state machine
# (missing .pt -> typed messages + hint, no crash, no frame mutation, the
# plain-optimize path untouched), the estimate -> optimizer seed
# (setSeedPose/applySeedPose lands in LocationStorage + scene, one-shot,
# stale-selection dropped), and the kind-preferred segment-model rule. Same
# fixture recipe as the other experimental tests (repo-root WORKING_DIRECTORY).
add_executable(jtml_test_experimental_ml_bridge
    unit/experimental_ml_bridge_test.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/MlBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/MlBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/PoseBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/PoseBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/OptimizerBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/OptimizerBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/AppBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/AppBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/SettingsBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/SettingsBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/StudyBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/StudyBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/services/study_load_controller.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/DelegateSelection.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/DelegateSelection.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalSession.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalScene.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalScene.cpp
    ${PROJECT_SOURCE_DIR}/src/services/session_controller.cpp
    ${PROJECT_SOURCE_DIR}/test/unit/frame_headless.cpp
    ${PROJECT_SOURCE_DIR}/src/services/location_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/services/model.cpp
    ${PROJECT_SOURCE_DIR}/src/services/stl_reader.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/model_list_builder.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/session_state.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/services/optimizer_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/services/settings_service.cpp
    ${PROJECT_SOURCE_DIR}/src/view/frame_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/frame_list_model.h
    ${PROJECT_SOURCE_DIR}/src/view/model_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/model_list_model.h
)
target_include_directories(jtml_test_experimental_ml_bridge PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${PROJECT_SOURCE_DIR}/src/app/experimental
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
    ${TORCH_INCLUDE_DIRS})
target_link_libraries(jtml_test_experimental_ml_bridge PRIVATE
    Catch2::Catch2WithMain
    jtml_coordinator
    ${REQUIRED_LIBS_QUALIFIED}
    ${JTA_LIBS}
    ${OpenCV_LIBRARIES}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES})
target_link_options(jtml_test_experimental_ml_bridge PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.experimental_ml_bridge
    COMMAND jtml_test_experimental_ml_bridge)
set_tests_properties(jtml.experimental_ml_bridge PROPERTIES
    LABELS "headless" TIMEOUT 300
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# PoseBridge pins (plan 005 U8, R9/R10). The bridge direct-compiles against
# the U4-U7 bridges + the app-owned dataset and links jtml_coordinator (the
# REAL seam symbols — LocationStorage/pose_copy/pose_file_io live in
# jtml_services/jtml_domain through the coordinator's PUBLIC link). The
# fixture builds a SYNTHETIC dataset (storage matrix + frame list model
# driven directly — no images/STLs parsed), so the suite is fast and the
# selection contract is driven through StudyBridge's public delegate API.
# Pins: immediate per-cell SavePose read-back + single-axis semantics + the
# scene-pose sync tail, the pinned pose_copy delegation (no-image fallback
# at frame 0 / the last frame, primary-row writes in v1 multi-select),
# non-numeric/NaN/infinite rejection with the inline message + unchanged
# state, WritePoseFile/WriteKinematicsFile false returns (message + state +
# dirty kept), the save->reload round-trip (pose + kinematics), the real
# golden kinematics fixture (test/golden/fem_oracle_captured.jtak) read
# through loadKinematics, and the guard + typed-file-error messages.
add_executable(jtml_test_experimental_pose_bridge
    unit/experimental_pose_bridge_test.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/PoseBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/PoseBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/AppBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/AppBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/MlBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/MlBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/OptimizerBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/OptimizerBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/SettingsBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/SettingsBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/StudyBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/StudyBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/services/study_load_controller.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/DelegateSelection.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/DelegateSelection.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalSession.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalScene.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalScene.cpp
    ${PROJECT_SOURCE_DIR}/src/services/session_controller.cpp
    ${PROJECT_SOURCE_DIR}/test/unit/frame_headless.cpp
    ${PROJECT_SOURCE_DIR}/src/services/location_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/services/model.cpp
    ${PROJECT_SOURCE_DIR}/src/services/stl_reader.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/model_list_builder.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/session_state.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/services/optimizer_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/services/settings_service.cpp
    ${PROJECT_SOURCE_DIR}/src/view/frame_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/frame_list_model.h
    ${PROJECT_SOURCE_DIR}/src/view/model_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/model_list_model.h
)
target_include_directories(jtml_test_experimental_pose_bridge PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${PROJECT_SOURCE_DIR}/src/app/experimental
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
    ${TORCH_INCLUDE_DIRS})
target_link_libraries(jtml_test_experimental_pose_bridge PRIVATE
    Catch2::Catch2WithMain
    jtml_coordinator
    ${REQUIRED_LIBS_QUALIFIED}
    ${JTA_LIBS}
    ${OpenCV_LIBRARIES}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES})
target_link_options(jtml_test_experimental_pose_bridge PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.experimental_pose_bridge
    COMMAND jtml_test_experimental_pose_bridge)
set_tests_properties(jtml.experimental_pose_bridge PROPERTIES
    LABELS "headless" TIMEOUT 300
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# (Placeholder) QtTest lifecycle / coordinator seam, added at U4.
# The coordinator tests run headless under QCoreApplication; no display needed.
# ---------------------------------------------------------------------------
# find_package(Qt5 COMPONENTS Core Widgets Test QUIET)  # Qt5 until U8
# add_executable(jtml_test_coordinator lifecycle/coordinator_test.cpp)
# target_link_libraries(jtml_test_coordinator PRIVATE Qt5::Test)
# add_test(NAME jtml.coordinator COMMAND jtml_test_coordinator)
# set_tests_properties(jtml.coordinator PROPERTIES LABELS "headless" TIMEOUT 60)

# ---------------------------------------------------------------------------
# Tier-2 GPU oracle (plan U6). Appearance-based: optimize Kneel_1 femur with
# the rewire (DirectOptimizer bound to the real GPU DIRECT_DILATION cost),
# render the implant at the recovered pose, and compare the silhouette to the
# known-good label (Labels/fem/) via IOU. GPU-only: NEVER run in the headless
# default. Run explicitly with: ctest --test-dir .build -L oracle
# ---------------------------------------------------------------------------
add_executable(jtml_test_oracle oracle/oracle_test.cpp)
target_include_directories(jtml_test_oracle PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(jtml_test_oracle PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    jtml_coordinator
    ${JTA_LIBS}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
    Catch2::Catch2WithMain
)
set_property(TARGET jtml_test_oracle PROPERTY CUDA_ARCHITECTURES native)
set_target_properties(jtml_test_oracle PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
)
# The conda toolchain overrides BUILD_RPATH, so append the runtime lib dirs
# explicitly (GNU ld accumulates -rpath): jtml_compute.so lives in the build
# lib dir, VTK + conda libs in their install dirs.
target_link_options(jtml_test_oracle PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.oracle COMMAND jtml_test_oracle)

# Offscreen VTK rendering smoke (post-003 debug): drives the real Viewer's
# display path (calibration -> image -> models -> render) offscreen and saves
# PNGs to render-smoke-output/ so a human/CI can verify the rendering loop.
# Needs a working GL/EGL context - NEVER headless (label: render; also oracle
# so the explicit GPU/real-VTK run picks it up).
add_executable(jtml_test_render_smoke
    oracle/render_smoke.cpp
)
target_include_directories(jtml_test_render_smoke PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(jtml_test_render_smoke PRIVATE
    jtml_view
    ${REQUIRED_LIBS_QUALIFIED}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
)
# VTK 9 requires per-target factory registration: without vtk_module_autoinit,
# vtkRenderWindow::New() returns the BASE class (no GL backend), vtkTextRenderer
# returns null, and vtkWindowToImageFilter segfaults reading the framebuffer.
vtk_module_autoinit(TARGETS jtml_test_render_smoke MODULES ${VTK_LIBRARIES})
set_property(TARGET jtml_test_render_smoke PROPERTY CUDA_ARCHITECTURES native)
target_link_options(jtml_test_render_smoke PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.render_smoke COMMAND jtml_test_render_smoke)
# Renders from the repo root so example_studies/Kneel_1 + render-smoke-output/ resolve.
# Force xcb: the Wayland/EGL stack on this box cannot give QVTKOpenGLNativeWidget
# a GL context (blank render) - the app's historical working platform is xcb.
set_tests_properties(jtml.render_smoke PROPERTIES LABELS "oracle;render" TIMEOUT 120
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ENVIRONMENT "QT_QPA_PLATFORM=xcb")

# Minimal VTK standalone-render probe (post-003 debug): isolates VTK+GL+W2I from
# the Viewer. Also needs a GL context; never headless.
add_executable(jtml_test_probe_vtk
    oracle/probe_vtk.cpp
)
target_include_directories(jtml_test_probe_vtk PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${VTK_INCLUDE_DIRS}
)
target_link_libraries(jtml_test_probe_vtk PRIVATE
    ${VTK_LIBRARIES}
    ${REQUIRED_LIBS_QUALIFIED}
)
vtk_module_autoinit(TARGETS jtml_test_probe_vtk MODULES ${VTK_LIBRARIES})
set_property(TARGET jtml_test_probe_vtk PROPERTY CUDA_ARCHITECTURES native)
target_link_options(jtml_test_probe_vtk PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.probe_vtk COMMAND jtml_test_probe_vtk)
set_tests_properties(jtml.probe_vtk PROPERTIES LABELS "oracle;render" TIMEOUT 120
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ENVIRONMENT "QT_QPA_PLATFORM=xcb")
# The oracle reads its fixtures via repo-relative paths (example_studies/Kneel_1/)
# and writes hm*.png/contour.png diagnostic artifacts, so it must run from the
# repo root -- not the .build dir that ctest defaults to.
set_tests_properties(jtml.oracle PROPERTIES LABELS "oracle" TIMEOUT 3600
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# Tier-2 z-profile probe (plan 008 U5). GPU-only: NEVER in the headless
# default. The cheapest instrument that sees the z-weak axis: a 31-render Δz
# sweep (±15 mm step 1) at fixed truth through the oracle-twin injected-cost
# lambda, per (variant, dilation, frame), with term decomposition
# (full/chamfer/dilated from the same render), yamazaki order-4 fit, and
# coupling (in-plane ±Δ) — records test/golden/z_profiles.json; run 1 records
# the valley metrics/ranking, later runs enforce against baseline.json's
# z_profiles block (pin-first). Same registration recipe as jtml.oracle
# (repo-root WORKING_DIRECTORY, `oracle;gpu` labels, TIMEOUT 3600 nightly-grade
# — measure after run 1 and adjust).
# ---------------------------------------------------------------------------
add_executable(jtml_test_z_profile oracle/z_profile_test.cpp)
target_include_directories(jtml_test_z_profile PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(jtml_test_z_profile PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    jtml_coordinator
    ${JTA_LIBS}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
    Catch2::Catch2WithMain
)
set_property(TARGET jtml_test_z_profile PROPERTY CUDA_ARCHITECTURES native)
set_target_properties(jtml_test_z_profile PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
)
# The conda toolchain overrides BUILD_RPATH, so append the runtime lib dirs
# explicitly (GNU ld accumulates -rpath): jtml_compute.so lives in the build
# lib dir, VTK + conda libs in their install dirs.
target_link_options(jtml_test_z_profile PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.z_profile COMMAND jtml_test_z_profile)
set_tests_properties(jtml.z_profile PROPERTIES LABELS "oracle;gpu" TIMEOUT 3600
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# CostCapacityService GPU-labeled oracle (plan 010 U9). NEVER in the headless
# default: requires a real CUDA device. Compiles the service's GPU queries
# (cost_capacity_service.cu) + a local probe kernel, pins the real-device
# SAFE_CAP == maximum_stride_size*(threads_per_block-1) and that the occupancy
# query returns > 0 blocks/SM. Same registration recipe as jtml.z_profile
# (`oracle;gpu` labels, repo-root WORKING_DIRECTORY, long TIMEOUT).
add_executable(jtml_test_cost_capacity_oracle
    oracle/cost_capacity_oracle_test.cu
    ${PROJECT_SOURCE_DIR}/src/compute/cost_capacity_service.cu
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp)
target_include_directories(jtml_test_cost_capacity_oracle PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_cost_capacity_oracle PRIVATE
    ${JTA_LIBS}
    ${CUDA_LIBRARIES}
    Catch2::Catch2WithMain)
target_link_options(jtml_test_cost_capacity_oracle PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib")
set_property(TARGET jtml_test_cost_capacity_oracle PROPERTY CUDA_ARCHITECTURES native)
add_test(NAME jtml.cost_capacity_oracle COMMAND jtml_test_cost_capacity_oracle)
set_tests_properties(jtml.cost_capacity_oracle PROPERTIES LABELS "oracle;gpu" TIMEOUT 3600
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# U3: Graph capture preflight probe (oracle;gpu, .cu)
add_executable(jtml_test_graph_capture_probe
    oracle/graph_capture_probe_test.cu
    ${PROJECT_SOURCE_DIR}/src/compute/graph_preflight.cu)
target_include_directories(jtml_test_graph_capture_probe PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute)
target_link_libraries(jtml_test_graph_capture_probe PRIVATE
    ${JTA_LIBS}
    ${CUDA_LIBRARIES}
    Catch2::Catch2WithMain)
set_property(TARGET jtml_test_graph_capture_probe PROPERTY CUDA_ARCHITECTURES native)
add_test(NAME jtml.graph_capture_probe COMMAND jtml_test_graph_capture_probe)
set_tests_properties(jtml.graph_capture_probe PROPERTIES LABELS "oracle;gpu" TIMEOUT 3600
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# U5: Monoplane DIRECT_DILATION graph recipe oracle (oracle;gpu, .cu)
add_executable(jtml_test_graph_recipe_direct_dilation
    oracle/graph_recipe_direct_dilation_test.cu
    ${PROJECT_SOURCE_DIR}/src/compute/graph_recipe_direct_dilation.cu
    ${PROJECT_SOURCE_DIR}/src/compute/evaluation_context.cpp)
target_include_directories(jtml_test_graph_recipe_direct_dilation PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${CUDA_INCLUDE_DIRS})
target_link_libraries(jtml_test_graph_recipe_direct_dilation PRIVATE
    Catch2::Catch2WithMain
    CUDA::cudart)
set_property(TARGET jtml_test_graph_recipe_direct_dilation PROPERTY CUDA_ARCHITECTURES native)
add_test(NAME jtml.graph_recipe_direct_dilation COMMAND jtml_test_graph_recipe_direct_dilation)
set_tests_properties(jtml.graph_recipe_direct_dilation PROPERTIES LABELS "oracle;gpu" TIMEOUT 3600
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# Plan 010 U10 bit-identity harness / baseline recorder. GPU-labeled: NEVER in
# the headless default. Drives the production pose->score path over a fixed
# frame and records the deterministic (pose,score) sequence; run against the
# PRE-wiring render_engine and capture stdout as
# test/golden/bit_identity_baseline.txt FIRST (characterization-first, R13),
# then after U10 wires the CostCapacityService into the fill launches re-run
# and the sequence must be BIT-IDENTICAL. Same registration recipe as
# jtml.z_profile (`oracle;gpu` labels, repo-root WORKING_DIRECTORY, TIMEOUT).
add_executable(jtml_test_bit_identity
    oracle/bit_identity_test.cpp)
target_include_directories(jtml_test_bit_identity PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS})
target_link_libraries(jtml_test_bit_identity PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    jtml_coordinator
    ${JTA_LIBS}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
    Catch2::Catch2WithMain)
set_property(TARGET jtml_test_bit_identity PROPERTY CUDA_ARCHITECTURES native)
set_target_properties(jtml_test_bit_identity PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}")
target_link_options(jtml_test_bit_identity PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.bit_identity COMMAND jtml_test_bit_identity)
set_tests_properties(jtml.bit_identity PROPERTIES LABELS "oracle;gpu" TIMEOUT 3600
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
# Cut-0 timing verdict reuses the production bit-identity fixture and executable;
# the Catch2 name filter selects only the GPU-active/CPU-host measurement.
add_test(NAME jtml.cut0_measurement
    COMMAND jtml_test_bit_identity "Cut-0: GPU-active versus CPU host time per production cost call")
set_tests_properties(jtml.cut0_measurement PROPERTIES LABELS "oracle;gpu" TIMEOUT 3600
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# Tier-2 multi-stage oracle (plan 008 U6). GPU-only: NEVER in the headless
# default. Drives OptimizerRunController with the PRODUCTION driver factory
# (real OptimizerManager + QThread per run) at the production shape
# (trunk 20000 -> 2x branch 5000 -> leaf 5000; cumulative caps
# 20/25/30/35k; per-stage dilation 6/4/1; Canny 3/0/150) on all 3
# Kneel_1 frames, plus the tibia-after-femur SymTrap pass (the three sym-trap
# pins: directive, 61 relays, costCalls 0 + stageText Idle), the edge shapes
# (branches=0 / leaf disabled — no phantom stages), and the caps-gate error
# path. Records test/golden/oracle_multistage.json (+_tibia.json) and prints
# the condensed block for baseline.json's oracle_multistage key (run 1
# records; later runs enforce against the recorded band). MEASURED wall time
# on the RTX 3090 Ti (2026-08-12): ~15 s for the whole binary — the plan's
# 10-30 min/frame estimate was conservative; TIMEOUT 7200 is kept as the
# plan-spec bound + headroom for slower GPUs. Same registration recipe as
# jtml.oracle (repo-root WORKING_DIRECTORY, rpath).
# ---------------------------------------------------------------------------
add_executable(jtml_test_oracle_multistage oracle/multistage_oracle_test.cpp)
target_include_directories(jtml_test_oracle_multistage PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(jtml_test_oracle_multistage PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    jtml_coordinator
    ${JTA_LIBS}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
    Catch2::Catch2
)
set_property(TARGET jtml_test_oracle_multistage PROPERTY CUDA_ARCHITECTURES native)
set_target_properties(jtml_test_oracle_multistage PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
)
# The conda toolchain overrides BUILD_RPATH, so append the runtime lib dirs
# explicitly (GNU ld accumulates -rpath): jtml_compute.so lives in the build
# lib dir, VTK + conda libs in their install dirs.
target_link_options(jtml_test_oracle_multistage PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.oracle_multistage COMMAND jtml_test_oracle_multistage)
set_tests_properties(jtml.oracle_multistage PROPERTIES LABELS "oracle;gpu"
    TIMEOUT 7200 WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# U9 parity gate (plan 005 U9, R13): the app's OWN optimizer run path
# (OptimizerBridge: OptimizeIntentController gate -> OptimizerManager +
# QThread -> Initialize with the app's containers by value + a
# QModelIndexList -> the 7 binds -> OptimizedFrame) on Kneel_1 frame 0 with
# the oracle config (trunk budget 3000, branch/leaf budgets 0, ranges
# (12,12,15,15,15,15), DIRECT_DILATION, dilation 6, Canny 3/0/150, start pose
# from fem.jts via pose_file_io) must land within the oracle's band: rendered
# silhouette IoU >= 0.85 vs the empirically pinned label
# (AT_K1_V1_0160_label_fem.tif), backface culling OFF, the label TIFF
# vertical flip -- oracle_test.cpp's render + IoU mechanics. GPU-only: NEVER
# in the headless default. Same registration recipe as jtml.oracle: `oracle`
# label, repo-root WORKING_DIRECTORY, TIMEOUT 3600; xcb env mirrors the
# render smokes (inert for QCoreApplication, pinned for consistency).
add_executable(jtml_test_qml_parity_check
    oracle/qml_parity_check.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/OptimizerBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/OptimizerBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/AppBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/AppBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/MlBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/MlBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/PoseBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/PoseBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/SettingsBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/SettingsBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/StudyBridge.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/StudyBridge.cpp
    ${PROJECT_SOURCE_DIR}/src/services/study_load_controller.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/DelegateSelection.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/DelegateSelection.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalSession.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalScene.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalScene.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/session_state.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/pose_file_io.cpp
    ${PROJECT_SOURCE_DIR}/src/services/optimizer_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/view/frame_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/frame_list_model.h
    ${PROJECT_SOURCE_DIR}/src/view/model_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/model_list_model.h
)
target_include_directories(jtml_test_qml_parity_check PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${PROJECT_SOURCE_DIR}/src/app/experimental
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
    ${TORCH_INCLUDE_DIRS})
target_link_libraries(jtml_test_qml_parity_check PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    jtml_coordinator
    ${JTA_LIBS}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
    ${CUDA_LIBRARIES}
    Catch2::Catch2
)
set_property(TARGET jtml_test_qml_parity_check PROPERTY CUDA_ARCHITECTURES native)
set_target_properties(jtml_test_qml_parity_check PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
)
# The conda toolchain overrides BUILD_RPATH, so append the runtime lib dirs
# explicitly (GNU ld accumulates -rpath): jtml_compute.so lives in the build
# lib dir, VTK + conda libs in their install dirs.
target_link_options(jtml_test_qml_parity_check PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.qml_parity_check COMMAND jtml_test_qml_parity_check)
set_tests_properties(jtml.qml_parity_check PROPERTIES LABELS "oracle" TIMEOUT 3600
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ENVIRONMENT "QT_QPA_PLATFORM=xcb")

# ---------------------------------------------------------------------------
# Tier-2 GPU oracle: segmentation + implant estimate path (plan 004 U8).
# Gates the U8 extraction (SegmentationController + ImplantEstimator) by the
# appearance contract (silhouette IoU vs Labels/ -- golden_oracle.org), never
# raw pose. The .pt torch models are USER-PROVIDED (NOT in test/golden/): the
# test skips cleanly with a message until JTML_SEG_PT / JTML_FEM_ESTIMATE_PT
# are set on the GPU machine. Same registration pattern as jtml.oracle:
# `oracle` label (never headless), repo-root WORKING_DIRECTORY, long TIMEOUT.
# ---------------------------------------------------------------------------
add_executable(jtml_test_segmentation_oracle oracle/segmentation_oracle_test.cpp)
target_include_directories(jtml_test_segmentation_oracle PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(jtml_test_segmentation_oracle PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    jtml_services
    ${JTA_LIBS}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Catch2::Catch2WithMain
)
set_property(TARGET jtml_test_segmentation_oracle PROPERTY CUDA_ARCHITECTURES native)
set_target_properties(jtml_test_segmentation_oracle PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
)
# The conda toolchain overrides BUILD_RPATH, so append the runtime lib dirs
# explicitly (GNU ld accumulates -rpath): jtml_compute.so lives in the build
# lib dir, VTK + conda libs in their install dirs.
target_link_options(jtml_test_segmentation_oracle PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.segmentation_oracle COMMAND jtml_test_segmentation_oracle)
set_tests_properties(jtml.segmentation_oracle PROPERTIES LABELS "oracle" TIMEOUT 3600
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# QML render smoke (plan 005 U1 spike gate / R15): loads spike.qml through
# QQmlApplicationEngine (type registration, qrc/import resolution,
# QQuickVTKItem ownership in a QML scene) and captures via
# QQuickWindow::grabWindow (vtkWindowToImageFilter is a documented segfault
# on this build). Prototype of the U9 smoke, registered at U1 so the gate's
# ctest verification is real. Same registration pattern as jtml.render_smoke:
# `oracle;render` label (never headless), forced xcb, repo-root cwd, TIMEOUT.
# ---------------------------------------------------------------------------
# GUISupportQtQuick is deliberately absent from the root find_package
# COMPONENTS list (root CMakeLists.txt stays untouched): this block() scoped
# re-find adds it here only. find_package sets VTK_LIBRARIES in THIS scope,
# so other targets' link lines are unchanged; the autoinit MODULES stay
# explicit against this scope's set. The spike's SpikeVtkItem.cpp + qrc are
# direct-compiled into the smoke (the established direct-compile pattern).
block()
find_package(Qt6 COMPONENTS Quick Qml Test REQUIRED)
find_package(VTK 9.3 COMPONENTS
    GUISupportQtQuick
    GUISupportQt
    RenderingCore
    RenderingOpenGL2
    InteractionStyle
    InteractionWidgets
    IOCore
    IOGeometry
    REQUIRED
)
add_executable(jtml_test_qml_render_smoke
    oracle/qml_render_smoke.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/QmlVtkRenderer.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/QmlVtkRenderer.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalScene.h
    ${PROJECT_SOURCE_DIR}/src/app/experimental/ExperimentalScene.cpp
    ${PROJECT_SOURCE_DIR}/src/app/experimental/renderer.qrc
)
target_include_directories(jtml_test_qml_render_smoke PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/app/experimental
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(jtml_test_qml_render_smoke PRIVATE
    Qt6::Quick
    Qt6::Qml
    Qt6::Test
    jtml_coordinator
    ${JTA_LIBS}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
)
# VTK 9 requires per-target factory registration (same failure modes as the
# render smoke: base-class vtkRenderWindow::New(), null vtkTextRenderer).
vtk_module_autoinit(TARGETS jtml_test_qml_render_smoke MODULES ${VTK_LIBRARIES})
set_property(TARGET jtml_test_qml_render_smoke PROPERTY CUDA_ARCHITECTURES native)
target_link_options(jtml_test_qml_render_smoke PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.qml_render_smoke COMMAND jtml_test_qml_render_smoke)
# Renders from the repo root so example_studies/Kneel_1 + the output dir
# resolve. Force xcb: same rationale as jtml.render_smoke.
set_tests_properties(jtml.qml_render_smoke PROPERTIES LABELS "oracle;render" TIMEOUT 120
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ENVIRONMENT "QT_QPA_PLATFORM=xcb")
endblock()

# Plan 007 U2: Qt6 qmllint gate over the experimental app's QML files.
# Pins $CONDA_PREFIX/lib/qt6/bin/qmllint (the Qt 6.7.2 binary — the bare
# `qmllint` on PATH is Qt 5.15.8 from opencv's qt5 build and lacks --json).
# Fails on warnings outside the documented accepted set (see
# test/qml_lint.cmake for the accepted ids + their removal paths).
if(EXISTS "$ENV{CONDA_PREFIX}/lib/qt6/bin/qmllint")
    set(QML_LINT_QT6 "$ENV{CONDA_PREFIX}/lib/qt6/bin/qmllint")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/.pixi/envs/default/lib/qt6/bin/qmllint")
    set(QML_LINT_QT6 "${CMAKE_SOURCE_DIR}/.pixi/envs/default/lib/qt6/bin/qmllint")
else()
    set(QML_LINT_QT6 "")
    # Review fix (ce-code-review 2026-08-12, AN2): the gate used to
    # silently vanish when no Qt6 qmllint was resolvable — a configure
    # warning makes the disappearance visible.
    message(WARNING "jtml.qml_lint NOT registered: Qt6 qmllint not found "
                    "($ENV{CONDA_PREFIX}/lib/qt6/bin or "
                    "${CMAKE_SOURCE_DIR}/.pixi/envs/default/lib/qt6/bin). "
                    "The bare 'qmllint' on PATH is Qt 5.15.8 and is never "
                    "used.")
endif()
if(QML_LINT_QT6)
    add_test(NAME jtml.qml_lint
        COMMAND ${CMAKE_COMMAND}
            -D QML_LINT=${QML_LINT_QT6}
            -D QML_DIR=${CMAKE_SOURCE_DIR}/src/app/experimental
            -D QML_OUT=${CMAKE_CURRENT_BINARY_DIR}/qml-lint-out.json
            -P ${CMAKE_CURRENT_SOURCE_DIR}/qml_lint.cmake)
    set_tests_properties(jtml.qml_lint PROPERTIES
        LABELS "headless" TIMEOUT 120
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

# Plan 007 U6: Qt Quick Test coverage for the experimental view layer.
# test/qml/ harness: tst files + fakes on disk (QUICK_TEST_SOURCE_DIR),
# components imported via tests.qrc aliases to src/app/experimental
# (no drift). Headless-safe by construction: no VTK, no bridges, no
# jtml libs — the fakes replace the C++ bridges. Material style env
# matches the app's style pin. The QuickTest ctest target is the SOLE
# run mechanism (bare `qmltestrunner` on PATH is Qt 5.15.8 and would run
# the tst files on the Qt 5 engine).
block()
find_package(Qt6 COMPONENTS QuickTest Quick Qml QuickControls2 REQUIRED)
add_executable(jtml_test_qml_view
    qml/main.cpp
    qml/tests.qrc
)
target_link_libraries(jtml_test_qml_view PRIVATE
    Qt6::QuickTest
    Qt6::Quick
    Qt6::Qml
    Qt6::QuickControls2
)
target_compile_definitions(jtml_test_qml_view PRIVATE
    QUICK_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/qml")
add_test(NAME jtml.qml_view COMMAND jtml_test_qml_view)
set_tests_properties(jtml.qml_view PROPERTIES
    LABELS "headless" TIMEOUT 300
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen;QT_QUICK_CONTROLS_STYLE=Material")
endblock()
