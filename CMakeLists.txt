# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

#=============================
# Path Configuration
#=============================
# Force CMake to look in conda environment first
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
set(CMAKE_FIND_ROOT_PATH "$ENV{CONDA_PREFIX}")
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)

# Explicitly ignore system paths that might conflict
list(APPEND CMAKE_IGNORE_PATH
    "/usr/local/cuda"
    "/usr/local/cuda/lib64"
    "/usr/lib64"
    "/usr/lib"
    "/usr/local/cuda-12.6/compat"  # Common CUDA compat directory
)

# RPATH settings
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:$ENV{CONDA_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#=============================
# Basic Project Setup
#=============================
cmake_minimum_required(VERSION 3.26)
project(joint-track-machine-learning LANGUAGES CXX CUDA)
message(STATUS ${CUDA_INCLUDE_DIRS})

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build the test suite (headless default; GPU/oracle cases are a separate label)
option(JTML_BUILD_TESTS "Build and enable JTML tests" ON)

#=============================
# Global Settings
#=============================
set(USE_CUDNN 1)
set(REQUIRED_LIBS Core Gui Widgets)
set(REQUIRED_LIBS_QUALIFIED Qt6::Core Qt6::Gui Qt6::Widgets)

# Qt setup
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

#=============================
# CUDA Setup - Add this before other find_package calls
#=============================
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Set CUDA paths based on conda environment
set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "Cuda installation root")
set(CUDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
set(CUDA_TOOLKIT_INCLUDE "$ENV{CONDA_PREFIX}/include")
set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc")
set(CMAKE_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib")

# Force CMake to recognize CUDA paths
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})

#=============================
# Find Required Packages
#=============================
find_package(Qt6 COMPONENTS Core Gui Widgets OpenGL Quick Qml REQUIRED)
#find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# JTA_LIBS is defined ONCE below (before add_subdirectory) as the single handle
# to the merged compute SHARED lib; do not define it twice here (a one-sided
# edit of a second occurrence would silently diverge every consumer's links).


# VTK setup - assuming VTK was built using the separate build script
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/_deps/vtk/install/lib64")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/_deps/vtk/install/lib64/cmake/vtk-9.3/")
    set(VTK_DIR "${CMAKE_CURRENT_LIST_DIR}/_deps/vtk/install/lib64/cmake/vtk-9.3/")
    set(VTK_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/_deps/vtk/install/lib64/")
else()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/_deps/vtk/install/lib/cmake/vtk-9.3/")
    set(VTK_DIR "${CMAKE_CURRENT_LIST_DIR}/_deps/vtk/install/lib/cmake/vtk-9.3/")
    set(VTK_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/_deps/vtk/install/lib/")
endif()

find_package(VTK 9.3 REQUIRED COMPONENTS
    GUISupportQt
    RenderingCore
    RenderingOpenGL2
    IOCore
    IOImage
    IOGeometry
    ChartsCore
    ViewsContext2D
    RenderingAnnotation
    InteractionStyle
)
#=============================
# Output Directories
#=============================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

#=============================
# Main Target
#=============================

set(JTA_LIBS jtml_compute)

#=============================
# Subdirectories
#=============================
add_subdirectory(src)
if(JTML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
#add_subdirectory(lib)
add_subdirectory(packaging)
