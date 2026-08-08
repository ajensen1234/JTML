# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# 003 U3/U4: jtml_coordinator is the Qt orchestration layer (optimize_coordinator,
# optimizer_manager). It consumes the single jtml_compute SHARED lib (via JTA_LIBS) --
# the compute sources moved out of the coordinator into jtml_compute in U4.
#
# AUTOMOC: the Q_OBJECT headers (optimize_coordinator.h, optimizer_manager.h)
# MUST be in the source list (via the coordinator header GLOB below) or AUTOMOC
# never mocs them -> undefined-vtable link error.
set(CMAKE_AUTOMOC ON)

file(GLOB HEADER_FILES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/include/coordinator/*.h
)

add_library(jtml_coordinator STATIC
    optimize_coordinator.cpp
    optimizer_manager.cpp
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
