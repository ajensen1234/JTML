# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# 003 U5: jtml_view is the STATIC Qt Widgets view layer (the classes that used
# to live in src/gui). It is kept as a separable static lib so the view is
# QML-swappable later; the thin composition root (main.cpp) + Study2Grid live in
# src/app and link this lib. .ui/.qrc/Resources are co-located under include/view
# and consumed atomically (AUTOUIC/AUTORCC/AUTOMOC).
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/view)
file(GLOB HEADER_FILES ${HEADER_DIR}/*.ui ${HEADER_DIR}/*.h)
set(RESOURCE_DIR ${PROJECT_SOURCE_DIR}/res)

add_library(jtml_view STATIC
    about.cpp
    controls.cpp
    drr_tool.cpp
    mainscreen.cpp
    settings_control.cpp
    viewer.cpp
    ${HEADER_FILES}
    ${HEADER_DIR}/mainscreen.qrc
)
# NOTE: interactor.cpp is NOT compiled - include/view/interactor.h defines its VTK
# classes inline (vtkStandardNewMacro in the header), and the old gui target also
# omitted interactor.cpp (orphan). Compiling it redefines New() -> link redefinition
# error. It is dead weight, flagged for the U6 cleanup sweep.

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
