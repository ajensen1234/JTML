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
