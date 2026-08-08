# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# 003 U3: jtml_coordinator is the Qt orchestration layer (optimize_coordinator,
# optimizer_manager) plus, TEMPORARILY, the compute sources (frame.cu,
# curvature_utilities, machine_learning_tools) until src/compute/ owns its own
# lib in U4. It links services+domain (down) and the GPU/CUDA/Torch/OpenCV stack.
#
# AUTOMOC: the Q_OBJECT headers (optimize_coordinator.h, optimizer_manager.h)
# MUST be in the source list (via the coordinator header GLOB below) or AUTOMOC
# never mocs them -> undefined-vtable link error.
set(CMAKE_AUTOMOC ON)

file(GLOB HEADER_FILES
    ${PROJECT_SOURCE_DIR}/include/coordinator/*.h
    ${PROJECT_SOURCE_DIR}/include/compute/*.h
)

add_library(jtml_coordinator STATIC
    optimize_coordinator.cpp
    optimizer_manager.cpp
    # Compute sources are housed here until src/compute/ lands in U4.
    ${PROJECT_SOURCE_DIR}/src/compute/curvature_utilities.cpp
    ${PROJECT_SOURCE_DIR}/src/compute/frame.cu
    ${PROJECT_SOURCE_DIR}/src/compute/machine_learning_tools.cpp
    ${HEADER_FILES}
)

# Downward deps: services (-> domain) + the GPU/cost shared libs (optimizer_manager
# uses bare gpu_*.cuh + CostFunctionManager.h => jtml_gpu + JTA_Cost_Functions).
target_link_libraries(jtml_coordinator PUBLIC jtml_services ${JTA_LIBS})
# Qt + VTK (via optimizer_manager/services), Torch + OpenCV + CUDA (compute).
target_link_libraries(jtml_coordinator PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Adding Include Directories. include/gpu resolves the bare gpu_*.cuh/pose_matrix.h
# in optimizer_manager + camera_calibration.h; CUDA includes for machine_learning_tools.
target_include_directories(jtml_coordinator PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/gpu
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/lib/
)

# frame.cu compiles in a STATIC lib (proven: jtml_core did the same) and inherits
# the project-wide CMAKE_CUDA_ARCHITECTURES=native; no per-target CUDA flags or
# separable compilation are required here (that is jtml_gpu only).
