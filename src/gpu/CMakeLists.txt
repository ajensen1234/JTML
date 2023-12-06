# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/gpu)
file(GLOB HEADER_FILES ${HEADER_DIR}/*.cuh ${HEADER_DIR}/*.h)


add_library(jtml_gpu SHARED
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
        distance_map_metric.cu
        fast_implant_dilation_metric.cu
        ${HEADER_DIR}/fast_implant_dilation_metric.cuh
        ${HEADER_FILES})



set_target_properties(jtml_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(jtml_gpu ${VTK_LIBRARIES})
target_link_libraries(jtml_gpu ${OpenCV_LIBRARIES})
target_link_libraries(jtml_gpu ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(jtml_gpu ${CUDA_LIBRARIES})

target_include_directories(jtml_gpu PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(jtml_gpu PUBLIC ${HEADER_DIR})
if (MSVC)
        target_link_libraries(jtml_gpu curand.lib)
endif()
get_target_property(TARGET_LIBRARIES jtml_gpu LINK_LIBRARIES)
set_property(TARGET jtml_gpu PROPERTY LINK_LIBRARIES ${TARGET_LIBRARIES})

target_include_directories(jtml_gpu PUBLIC ${PROJECT_SOURCE_DIR}/include)
