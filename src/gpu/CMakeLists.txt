set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/gpu)
file(GLOB HEADER_FILES ${HEADER_DIR}/*.cuh ${HEADER_DIR}/*.h)


add_library(jtml_gpu SHARED
        dilate_edge_detected_image.cu
        edge_detect_rendered_implant_model.cu
        fast_implant_dilation_metric.cu
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
        ${HEADER_FILES})



set_target_properties(jtml_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
include(${VTK_USE_FILE})
#target_link_libraries(jtml_gpu ${VTK_LIBRARIES})
target_link_libraries(jtml_gpu ${OpenCV_LIBRARIES})
target_link_libraries(jtml_gpu ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(jtml_gpu ${CUDA_LIBRARIES})
target_link_libraries(jtml_gpu curand.lib)
target_include_directories(jtml_gpu PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(jtml_gpu PUBLIC ${HEADER_DIR})


get_target_property(TARGET_LIBRARIES jtml_gpu LINK_LIBRARIES)
list(REMOVE_ITEM TARGET_LIBRARIES vtkRenderingQt vtkViewsGeovis vtkViewsQt)
set_property(TARGET jtml_gpu PROPERTY LINK_LIBRARIES ${TARGET_LIBRARIES})

target_include_directories(jtml_gpu PUBLIC ${PROJECT_SOURCE_DIR}/include)