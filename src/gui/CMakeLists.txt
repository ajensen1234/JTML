# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

#=============================
# GUI Configuration
#=============================
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/gui)
file(GLOB HEADER_FILES ${HEADER_DIR}/*.ui ${HEADER_DIR}/*.h)
set(RESOURCE_DIR ${PROJECT_SOURCE_DIR}/res)

#=============================
# Executable Target
#=============================
add_executable(${PROJECT_NAME}
    about.cpp
    controls.cpp
    drr_tool.cpp
    main.cpp
    mainscreen.cpp
    settings_control.cpp
    viewer.cpp
    ${HEADER_FILES}
    ${HEADER_DIR}/mainscreen.qrc
)

# Link Required Libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${REQUIRED_LIBS_QUALIFIED}
    jtml_core
    ${JTA_LIBS}
    ${VTK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
# CUDA Configuration for GUI
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES native)

set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib:$ORIGIN/../lib/vtk:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../lib/vtk:$ENV{CONDA_PREFIX}/lib:${VTK_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Include Directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${RESOURCE_DIR}
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
add_dependencies(${PROJECT_NAME} vtk_external)

#=============================
# Platform-Specific Definitions
#=============================
if(MSVC)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _WIN32 _WIN64 JTML_EXPORTS)
endif()
