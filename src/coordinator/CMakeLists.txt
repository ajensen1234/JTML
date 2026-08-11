# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# 003 U3/U4: jtml_coordinator is the Qt orchestration layer (optimize_coordinator,
# optimizer_manager). It consumes the single jtml_compute SHARED lib (via JTA_LIBS) --
# the compute sources moved out of the coordinator into jtml_compute in U4.
#
# Plan 006 U5/U6 (the shared VM layer): the OptimizerRunController surface joins
# this layer -- optimizer_run_controller_core.cpp (Qt/GPU-free run core: gate,
# run-state machine, epoch, progress mapping, seed lifecycle) +
# optimizer_run_driver.cpp (the OptimizerRunDriver seam + the production adapter
# wrapping OptimizerManager UNTOUCHED) + optimizer_run_controller.cpp (the QObject
# shell: the drive sequence, thread lifecycle + destructor contract, the 8
# by-value relay binds). U6 adds session_state_controller.cpp (the QObject
# notification shell over jta::SessionState: diff-based grouped emission, M9;
# previous-mirror advance on CommitSelection, H2; ResetForDatasetClear + seed
# drop, H5/M10b). Both front-ends drive the controllers; the view-specific
# mappers stay view-side.
#
# AUTOMOC: the Q_OBJECT headers (optimize_coordinator.h, optimizer_manager.h,
# optimizer_run_controller.h, session_state_controller.h) MUST be in the source
# list (via the coordinator header GLOB below) or AUTOMOC never mocs them ->
# undefined-vtable link error.
set(CMAKE_AUTOMOC ON)

file(GLOB HEADER_FILES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/include/coordinator/*.h
)

add_library(jtml_coordinator STATIC
    optimize_coordinator.cpp
    optimizer_manager.cpp
    optimizer_run_controller_core.cpp
    optimizer_run_driver.cpp
    optimizer_run_controller.cpp
    session_state_controller.cpp
    ${HEADER_FILES}
)

# Downward deps: services (-> domain) + the single SHARED compute lib
# (optimizer_manager uses gpu_*.cuh + CostFunctionManager.h => jtml_compute, merged U4).
target_link_libraries(jtml_coordinator PUBLIC jtml_services ${JTA_LIBS})
# Qt + VTK (via optimizer_manager/services), Torch + OpenCV + CUDA (compute).
target_link_libraries(jtml_coordinator PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Adding Include Directories. include/compute resolves the bare/resolver gpu/cost
# headers (gpu_*.cuh, pose_matrix.h, camera_calibration.h) in optimizer_manager + the
# cost managers; CUDA includes for the compute path.
target_include_directories(jtml_coordinator PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${CUDA_INCLUDE_DIRS}
)

# The compute + GPU/CUDA surface now lives in the jtml_compute SHARED lib, linked
# through ${JTA_LIBS}; no per-target CUDA flags or separable compilation are needed
# here (that is jtml_compute only).
