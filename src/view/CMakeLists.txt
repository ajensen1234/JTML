# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# 003 U5: jtml_view is the STATIC Qt Widgets view layer (the classes that used
# to live in src/gui). It is kept as a separable static lib so the view is
# QML-swappable later; the thin composition root (main.cpp) + Study2Grid live in
# src/app and link this lib. .ui/.qrc/Resources are co-located under include/view
# and consumed atomically (AUTOUIC/AUTORCC/AUTOMOC).
#
# QML DECISION RECORD (2026-08-11, plan 005 / panoptes angle 04): the seams are
# QML-ready (models/services/domain widget-free; VTK 9.3 GUISupportQtQuick built
# in-tree; the QML spike passed its go/no-go gate). The full QML front-end is
# DEFERRED — the experimental app (src/app/experimental, jtml_experimental) is
# the sanctioned parallel vehicle. Revisit a full QML front-end on a concrete UI
# need QWidgets cannot deliver, or Qt 6.8+/VTK 9.4+ integration maturity.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/view)
file(GLOB HEADER_FILES CONFIGURE_DEPENDS ${HEADER_DIR}/*.ui ${HEADER_DIR}/*.h)
set(RESOURCE_DIR ${PROJECT_SOURCE_DIR}/res)

add_library(jtml_view STATIC
    about.cpp
    controls.cpp
    drr_tool.cpp
    frame_list_model.cpp
    mainscreen.cpp
    model_list_model.cpp
    settings_control.cpp
    viewer.cpp
    ${HEADER_FILES}
    ${HEADER_DIR}/mainscreen.qrc
)
# NOTE: no interactor.cpp is compiled - include/view/interactor.h defines its VTK
# interactor classes inline (vtkStandardNewMacro in the header). The orphan .cpp was
# deleted in 003 U6; compiling a copy would redefine New() (link error).

# View consumes the layered stack (coordinator -> services -> domain, + compute)
# and the Qt/VTK/OpenCV/Eigen toolchain.
target_link_libraries(jtml_view PUBLIC
    ${REQUIRED_LIBS_QUALIFIED}
    jtml_coordinator
    ${JTA_LIBS}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
)

target_include_directories(jtml_view PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${RESOURCE_DIR}
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# AUTOUIC produces ui_*.h (e.g. ui_mainscreen.h) into this target's autogen
# include dir; mainscreen.h references ui_mainscreen.h, so consumers (the app
# composition root) inherit the generated-UI include dir.
target_include_directories(jtml_view PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/jtml_view_autogen/include
)

# CUDA for any view TU that pulls compute headers.
set_property(TARGET jtml_view PROPERTY CUDA_ARCHITECTURES native)
