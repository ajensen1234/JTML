# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# 003 U3: jtml_services is the headless, non-pure service layer (model/stl_reader/
# optimizer_settings/location_storage/calibration). It is NOT Qt-free on this cut
# (model.h uses VTK+QString, optimizer_settings.h uses QMetaType, stl readers use
# QtCore) -- the purity decouples are deferred follow-up. It depends downward on
# jtml_domain (optimizer_settings.cpp -> domain/settings_constants.h).
#
# 004 U8: layer-character change -- jtml_services becomes GPU-linked. The
# SegmentationController + ImplantEstimator (the per-frame segment/estimate ops
# of the MainScreen block) wrap compute::segment_image and call the CUDA cache
# clear + the estimate math, so ${TORCH_LIBRARIES} and jtml_compute (${JTA_LIBS})
# join the link line below. Headless consumers are unaffected: tests compile
# layer .cpps directly and never link this lib as a whole.
set(CMAKE_AUTOMOC ON)

file(GLOB HEADER_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/services/*.h)

add_library(jtml_services STATIC
    location_storage.cpp
    model.cpp
    optimizer_settings.cpp
    render_pipeline_builder.cpp
    session_controller.cpp
    study_load_controller.cpp
    settings_service.cpp
    cost_function_registry.cpp
    save_last_pose.cpp
    stl_reader.cpp
    STLReader.cpp
    edge_processor.cpp
    segmentation_controller.cpp
    implant_estimator.cpp
    ${HEADER_FILES}
)

# Downward dep on the pure core.
target_link_libraries(jtml_services PUBLIC jtml_domain)
# Qt (model.cpp QString, optimizer_settings.h QMetaType, stl QtCore) + VTK
# (model.h vtkSTLReader/vtkSmartPointer) + OpenCV (edge_processor.cpp calls the
# Frame methods; frame.h needs the OpenCV headers).
# 004 U8: Torch (segmentation_controller.cpp: c10 CUDA cache clear + torch
# module param) + jtml_compute (segment_image / GPUModel / the estimate math).
# The services lib is now GPU-linked; the SHARED jtml_compute's PUBLIC torch/
# CUDA deps propagate to the final link (app already links torch + JTA_LIBS).
target_link_libraries(jtml_services PRIVATE ${REQUIRED_LIBS_QUALIFIED} ${VTK_LIBRARIES} ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES} ${JTA_LIBS})

# 004 U8: CUDA include dirs for the GPU-linked TUs (implant_estimator.cpp pulls
# compute/gpu_model.cuh -> cuda_runtime.h; machine_learning_tools.h pulls cuda.h).
target_include_directories(jtml_services PRIVATE ${CUDA_INCLUDE_DIRS})

# Adding Include Directories. include/compute is PUBLIC because calibration.h -> the
# bare camera_calibration.h include (in include/compute) is consumed through
# location_storage.h by any jtml_services consumer. OpenCV include dirs are PUBLIC
# because edge_processor.h (a public header) pulls compute/frame.h -> OpenCV.
target_include_directories(jtml_services PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${OpenCV_INCLUDE_DIRS})
