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

#=============================
# Global Settings
#=============================
set(USE_CUDNN 1)
set(REQUIRED_LIBS Core Gui Widgets)
set(REQUIRED_LIBS_QUALIFIED Qt5::Core Qt5::Gui Qt5::Widgets)

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
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)
#find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(JTA_LIBS jtml_gpu JTA_Cost_Functions)
#set(JTA_INCLUDES ${JTA_COST_FUNCTIONS_INCLUDE_PATH} ${COSTFUNCTIONTOOLS_INCLUDE_PATH})


include(ExternalProject)

# Define VTK build
ExternalProject_Add(vtk_external
    PREFIX "${CMAKE_BINARY_DIR}/external/vtk"
    GIT_REPOSITORY https://github.com/Kitware/VTK.git
    GIT_TAG v9.3.0
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DVTK_GROUP_ENABLE_Qt=YES
        -DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES
        -DQt5_DIR=${Qt5_DIR}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DVTK_MODULE_ENABLE_VTK_RenderingCore=YES
        -DVTK_MODULE_ENABLE_VTK_RenderingOpenGL2=YES
        -DVTK_MODULE_ENABLE_VTK_IOCore=YES
        -DVTK_MODULE_ENABLE_VTK_IOImage=YES
        -DVTK_MODULE_ENABLE_VTK_IOGeometry=YES
        -DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES
        -DVTK_MODULE_ENABLE_VTK_ChartsCore=YES
        -DVTK_MODULE_ENABLE_VTK_ViewsContext2D=YES
        -DVTK_MODULE_ENABLE_VTK_RenderingAnnotation=YES
        -DVTK_MODULE_ENABLE_VTK_InteractionStyle=YES
        -DVTK_USE_X=OFF
        -DVTK_USE_SDL2=ON
    BUILD_ALWAYS ON
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
)
# Create imported target for VTK after external project install
ExternalProject_Get_Property(vtk_external INSTALL_DIR)
set(VTK_DIR "${INSTALL_DIR}/lib/cmake/vtk-9.3")
set(VTK_LIB_DIR "${INSTALL_DIR}/lib64/")
list(APPEND CMAKE_PREFIX_PATH "${INSTALL_DIR}")
find_package(VTK REQUIRED)

# Add an interface target to link VTK

#=============================
# Output Directories
#=============================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

#=============================
# Main Target
#=============================

set(JTA_LIBS jtml_gpu JTA_Cost_Functions)

#=============================
# Subdirectories
#=============================
add_subdirectory(src)
#add_subdirectory(test)
#add_subdirectory(lib)
add_subdirectory(packaging)

#=============================
# Testing
#=============================
#enable_testing()
