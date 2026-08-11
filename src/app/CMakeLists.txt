# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# 003 U5: src/app is the composition root. It holds the thin GUI main() (which
# instantiates MainScreen from the jtml_view lib) and the Study2Grid tool. It is
# the only place that wires the layered libs + view + toolchain into executables.

# Study2Grid: self-contained tool, relocated verbatim into app/Study2Grid.
add_subdirectory(Study2Grid)

# 005 U1: QML+VTK spike (go/no-go gate for the experimental QML front-end).
# Scratch target, isolated in its own directory scope (directory-scoped VTK
# re-find for GUISupportQtQuick; the widgets app's link line is untouched).
add_subdirectory(experimental)

# Principal GUI executable (composition root).
add_executable(${PROJECT_NAME}
    main.cpp
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    jtml_view
    jtml_coordinator
    ${JTA_LIBS}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
)

# CUDA Configuration for the executable (QVTK/OpenGL + compute linkage).
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES native)

set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib:$ORIGIN/../lib/vtk:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../lib/vtk:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

if(MSVC)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _WIN32 _WIN64 JTML_EXPORTS)
endif()
