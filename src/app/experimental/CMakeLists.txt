# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# 005 U3: jtml_experimental — the final QML app target. Replaces the U1
# scratch target jtml_experimental_spike (spike_main.cpp is superseded by
# main.cpp and removed). U3 replaced the U2 placeholder SpikeVtkItem with
# QmlVtkRenderer + ExperimentalScene (the render seam); the U1 smoke in
# test/oracle/ now drives the new renderer (its target compiles
# QmlVtkRenderer.cpp + ExperimentalScene.cpp + renderer.qrc). U4 added
# StudyBridge + DelegateSelection + ExperimentalSession (the study-load
# adapter, the delegate selection contract, and the app-owned dataset). U5
# added SettingsBridge (the session-local settings adapter: OptimizerSettings
# + three per-stage CostFunctionManager(Stage) instances + the replicated
# registry mapping; Save/Load/Reset through SettingsService) + SettingsPanel.qml.
#
# GUISupportQtQuick is deliberately NOT in the root find_package COMPONENTS
# list (root CMakeLists.txt stays untouched): this directory-scoped re-find
# adds it here only. find_package sets VTK_LIBRARIES in THIS directory scope
# (and below), so the widgets app's link line in the parent scope is
# unchanged. The autoinit MODULES must stay explicit against THIS scope's
# VTK_LIBRARIES (the directory-scoped set), not the root one.
#
# Link set = the widgets-app set minus jtml_view (R1): jtml_coordinator
# PUBLIC-pulls jtml_services -> jtml_domain + jtml_compute (JTA_LIBS).
# FrameListModel/ModelListModel are widget-free QAbstractListModels
# direct-compiled into the target (the established direct-compile pattern;
# jtml_view is NOT linked). AUTOMOC: every Q_OBJECT header (AppBridge.h,
# SettingsBridge.h, QmlVtkRenderer.h, the two model headers) is in the
# add_executable source list (repo gotcha: a globbed header without its impl in the target is an
# undefined-vtable link error). AUTORCC is global (root CMakeLists.txt), so
# listing resources.qrc in the sources is enough.

find_package(Qt6 COMPONENTS Quick Qml QuickControls2 REQUIRED)
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

add_executable(jtml_experimental
    main.cpp
    AppBridge.h
    AppBridge.cpp
    SettingsBridge.h
    SettingsBridge.cpp
    StudyBridge.h
    StudyBridge.cpp
    DelegateSelection.h
    DelegateSelection.cpp
    ExperimentalSession.h
    QmlVtkRenderer.h
    QmlVtkRenderer.cpp
    ExperimentalScene.h
    ExperimentalScene.cpp
    resources.qrc
    ${PROJECT_SOURCE_DIR}/src/view/frame_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/frame_list_model.h
    ${PROJECT_SOURCE_DIR}/src/view/model_list_model.cpp
    ${PROJECT_SOURCE_DIR}/include/view/model_list_model.h
)

target_link_libraries(jtml_experimental PRIVATE
    Qt6::Quick
    Qt6::Qml
    Qt6::QuickControls2
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
vtk_module_autoinit(TARGETS jtml_experimental MODULES ${VTK_LIBRARIES})

# CUDA Configuration for the executable (compute linkage).
set_property(TARGET jtml_experimental PROPERTY CUDA_ARCHITECTURES native)

# RPATH mirrors src/app/CMakeLists.txt (the conda toolchain overrides
# BUILD_RPATH, so the explicit -rpath entries below are the actual link
# recipe; GNU ld accumulates).
set_target_properties(jtml_experimental PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib:$ORIGIN/../lib/vtk:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../lib/vtk:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
target_link_options(jtml_experimental PRIVATE
    "-Wl,-rpath,$ORIGIN/../lib"
    "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib"
    "-Wl,-rpath,${VTK_LIB_DIR}")

target_include_directories(jtml_experimental PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
