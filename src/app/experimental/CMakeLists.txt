# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# 005 U1: QML+VTK spike (go/no-go gate). Scratch target proving
# QQuickVTKItem renders + interacts + captures under this box's
# xcb + Qt 6.7.2 + VTK 9.3 before any real UI work (plan 005 U1).
#
# GUISupportQtQuick is deliberately NOT in the root find_package COMPONENTS
# list (root CMakeLists.txt stays untouched): this directory-scoped re-find
# adds it here only. find_package sets VTK_LIBRARIES in THIS directory scope
# (and below), so the widgets app's link line in the parent scope is
# unchanged. The autoinit MODULES must stay explicit against THIS scope's
# VTK_LIBRARIES (the directory-scoped set), not the root one.

find_package(Qt6 COMPONENTS Quick Qml REQUIRED)
find_package(VTK 9.3 COMPONENTS
    GUISupportQtQuick
    GUISupportQt
    RenderingCore
    RenderingOpenGL2
    InteractionStyle
    InteractionWidgets
    IOCore
    IOGeometry
    REQUIRED
)

add_executable(jtml_experimental_spike
    spike_main.cpp
    SpikeVtkItem.h
    SpikeVtkItem.cpp
    spike.qrc
)

target_link_libraries(jtml_experimental_spike PRIVATE
    Qt6::Quick
    Qt6::Qml
    jtml_coordinator
    ${JTA_LIBS}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
)

# VTK 9 requires per-target factory registration: without vtk_module_autoinit,
# vtkRenderWindow::New() returns the BASE class (no GL backend), vtkTextRenderer
# returns null, and vtkWindowToImageFilter segfaults reading the framebuffer.
vtk_module_autoinit(TARGETS jtml_experimental_spike MODULES ${VTK_LIBRARIES})

# CUDA Configuration for the executable (compute linkage).
set_property(TARGET jtml_experimental_spike PROPERTY CUDA_ARCHITECTURES native)

# RPATH mirrors src/app/CMakeLists.txt (the conda toolchain overrides
# BUILD_RPATH, so the explicit -rpath entries below are the actual link
# recipe; GNU ld accumulates).
set_target_properties(jtml_experimental_spike PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib:$ORIGIN/../lib/vtk:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../lib/vtk:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
target_link_options(jtml_experimental_spike PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")

target_include_directories(jtml_experimental_spike PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
