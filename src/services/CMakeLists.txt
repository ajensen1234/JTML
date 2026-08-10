# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# 003 U3: jtml_services is the headless, non-pure service layer (model/stl_reader/
# optimizer_settings/location_storage/calibration). It is NOT Qt-free on this cut
# (model.h uses VTK+QString, optimizer_settings.h uses QMetaType, stl readers use
# QtCore) -- the purity decouples are deferred follow-up. It depends downward on
# jtml_domain (optimizer_settings.cpp -> domain/settings_constants.h).
set(CMAKE_AUTOMOC ON)

file(GLOB HEADER_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/services/*.h)

add_library(jtml_services STATIC
    location_storage.cpp
    model.cpp
    optimizer_settings.cpp
    session_controller.cpp
    settings_service.cpp
    stl_reader.cpp
    STLReader.cpp
    edge_processor.cpp
    ${HEADER_FILES}
)

# Downward dep on the pure core.
target_link_libraries(jtml_services PUBLIC jtml_domain)
# Qt (model.cpp QString, optimizer_settings.h QMetaType, stl QtCore) + VTK
# (model.h vtkSTLReader/vtkSmartPointer) + OpenCV (edge_processor.cpp calls the
# Frame methods; frame.h needs the OpenCV headers).
target_link_libraries(jtml_services PRIVATE ${REQUIRED_LIBS_QUALIFIED} ${VTK_LIBRARIES} ${OpenCV_LIBRARIES})

# Adding Include Directories. include/compute is PUBLIC because calibration.h -> the
# bare camera_calibration.h include (in include/compute) is consumed through
# location_storage.h by any jtml_services consumer. OpenCV include dirs are PUBLIC
# because edge_processor.h (a public header) pulls compute/frame.h -> OpenCV.
target_include_directories(jtml_services PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/compute
    ${OpenCV_INCLUDE_DIRS})
