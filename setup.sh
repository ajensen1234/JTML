#!/bin/bash

# Set environment variables for CMake to find packages
export Qt5_DIR="$CONDA_PREFIX/lib/cmake/Qt5"
export VTK_DIR="$CONDA_PREFIX/lib/cmake/vtk-9.3"
export Torch_DIR="$CONDA_PREFIX/lib/cmake/Torch"
export OpenCV_DIR="$CONDA_PREFIX/lib/cmake/opencv4"

# CUDA setup
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"

# Library path for runtime
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
