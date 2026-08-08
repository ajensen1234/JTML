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
# include/gpu on the path because services/calibration.h pulls camera_calibration.h.
add_executable(jtml_test_location_storage_props
    unit/test_location_storage_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/services/location_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_location_storage_props PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/gpu)
target_link_libraries(jtml_test_location_storage_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_location_storage_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.location_storage_props COMMAND jtml_test_location_storage_props)
set_tests_properties(jtml.location_storage_props PROPERTIES LABELS "headless" TIMEOUT 300)

# Hegel PBT for the header-only biplane Calibration conversion (src/services). The
# test needs include/gpu for camera_calibration.h; only data_structures_6D.cpp is
# compiled (calibration.h is header-only).
add_executable(jtml_test_calibration_props
    unit/test_calibration_properties.cpp
    ${PROJECT_SOURCE_DIR}/src/domain/data_structures_6D.cpp
)
target_include_directories(jtml_test_calibration_props PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/gpu)
target_link_libraries(jtml_test_calibration_props PRIVATE
    Catch2::Catch2WithMain hegel dl)
target_link_options(jtml_test_calibration_props PRIVATE
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/_deps/hegel-build/libhegel")
add_test(NAME jtml.calibration_props COMMAND jtml_test_calibration_props)
set_tests_properties(jtml.calibration_props PROPERTIES LABELS "headless" TIMEOUT 300)

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
    jtml_core
    ${JTA_LIBS}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Catch2::Catch2WithMain
)
set_property(TARGET jtml_test_oracle PROPERTY CUDA_ARCHITECTURES native)
set_target_properties(jtml_test_oracle PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
)
# The conda toolchain overrides BUILD_RPATH, so append the runtime lib dirs
# explicitly (GNU ld accumulates -rpath): jtml_gpu/JTA_Cost_Functions live in
# the build lib dir, VTK + conda libs in their install dirs.
target_link_options(jtml_test_oracle PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")
add_test(NAME jtml.oracle COMMAND jtml_test_oracle)
# The oracle reads its fixtures via repo-relative paths (example_studies/Kneel_1/)
# and writes hm*.png/contour.png diagnostic artifacts, so it must run from the
# repo root -- not the .build dir that ctest defaults to.
set_tests_properties(jtml.oracle PROPERTIES LABELS "oracle" TIMEOUT 3600
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
