# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# 003 U4: jtml_compute is the SINGLE merged GPU/compute SHARED lib, combining the
# former jtml_gpu (src/gpu) + JTA_Cost_Functions (src/cost_functions) with the
# 003 U2 compute sources (frame.cu, curvature_utilities.cpp,
# machine_learning_tools.cpp). The merge is acyclic (cost functions depend one-way
# on the GPU primitives), so one SHARED lib preserves the link order with no break
# (R15). JTA_LIBS now == this lib.
#
# Naming (directory provenance): gpu_*.cuh / snake_case <- ex-jtml_gpu,
# CostFunction*/...CustomVariables.h <- ex-JTA_Cost_Functions, and frame.h/
# pose_matrix.h/camera_calibration.h from the 003 U2 compute layer. No Qt/Q_OBJECT
# here - AUTOMOC stays OFF.
set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/compute)
file(GLOB HEADER_FILES CONFIGURE_DEPENDS ${HEADER_DIR}/*.h ${HEADER_DIR}/*.cuh)

add_library(jtml_compute SHARED
    # --- GPU primitives (ex-jtml_gpu) ---
    dilate_edge_detected_image.cu
    edge_detect_rendered_implant_model.cu
    gpu_dilated_frame.cu
    gpu_edge_frame.cu
    gpu_frame.cu
    gpu_image.cu
    gpu_image_functions.cu
    gpu_intensity_frame.cu
    gpu_metrics.cu
    gpu_model.cu
    implant_mahfouz_metric.cu
    iou.cu
    l_1_1_matrix_diff_norm.cu
    metric_toolbox.cu
    pose_matrix.cpp
    registration_metric.cu
    render_drr_engine.cu
    render_engine.cu
    capture_coordinator.cpp
    distance_map_metric.cu
    fast_implant_dilation_metric.cu
    gpu_heatmaps.cu
    curvature_hausdorf_metric.cu
    # --- compute layer (003 U2) ---
    frame.cu
    curvature_utilities.cpp
    machine_learning_tools.cpp
    # --- cost functions (ex-JTA_Cost_Functions) ---
    CostFunction.cpp
    CostFunctionManager.cpp
    DD_NEW_POLE_CONSTRAINT.cpp
    DIRECT_DILATION.cpp
    DIRECT_DILATION_POLE_CONSTRAINT.cpp
    DIRECT_DILATION_SAME_Z.cpp
    DIRECT_DILATION_T1.cpp
    DIRECT_MAHFOUZ.cpp
    sym_trap_function.cpp
    ${HEADER_FILES})

set_target_properties(jtml_compute PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

target_link_libraries(jtml_compute
    PUBLIC
        ${OpenCV_LIBRARIES}
        CUDA::cublas
        CUDA::cudart
        CUDA::curand
    PRIVATE
        ${TORCH_LIBRARIES}
)

target_include_directories(jtml_compute
    PUBLIC
      ${HEADER_DIR}
      ${PROJECT_SOURCE_DIR}/include
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
      ${OpenCV_INCLUDE_DIRS}

)

target_link_options(jtml_compute PRIVATE
    "-Wl,--no-undefined"
)
