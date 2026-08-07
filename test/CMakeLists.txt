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
    ${PROJECT_SOURCE_DIR}/src/core/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/core/direct_data_storage.cpp
)
target_include_directories(jtml_test_data_structures PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_data_structures PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.data_structures COMMAND jtml_test_data_structures)
set_tests_properties(jtml.data_structures PROPERTIES LABELS "headless" TIMEOUT 120)

# Pure DIRECT optimizer test (plan U5 / Tier-1 analytic golden). Compiled
# CUDA-free from the optimizer + data-structure sources, no jtml_core.
add_executable(jtml_test_direct_optimizer
    unit/test_direct_optimizer.cpp
    ${PROJECT_SOURCE_DIR}/src/core/direct_optimizer.cpp
    ${PROJECT_SOURCE_DIR}/src/core/data_structures_6D.cpp
    ${PROJECT_SOURCE_DIR}/src/core/direct_data_storage.cpp
)
target_include_directories(jtml_test_direct_optimizer PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(jtml_test_direct_optimizer PRIVATE Catch2::Catch2WithMain)
add_test(NAME jtml.direct_optimizer COMMAND jtml_test_direct_optimizer)
set_tests_properties(jtml.direct_optimizer PROPERTIES LABELS "headless" TIMEOUT 300)

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
# (Placeholder) GPU / oracle target, added at U6. Labeled so it is NOT in the
# default headless run.
# ---------------------------------------------------------------------------
# add_executable(jtml_test_oracle oracle/oracle_test.cpp)
# add_test(NAME jtml.oracle COMMAND jtml_test_oracle)
# set_tests_properties(jtml.oracle PROPERTIES LABELS "oracle" TIMEOUT 1800)
