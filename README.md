# Important information for compilation, building, and running on Linux!

## Dependencies

1. [Qt 5.15.2](https://doc.qt.io/qt-5/gettingstarted.html)
2. [VTK 9.2.6](https://vtk.org/download/)
3. [OpenCV 4.7](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
4. [GCC 12](https://itslinuxfoss.com/install-gcc-ubuntu-22-04/#2)
5. [CUDA 12.0](https://developer.nvidia.com/cuda-12-0-0-download-archive)
6. NVIDIA driver 525
7. [Libtorch 1.13](https://pytorch.org/get-started/locally/)
8. CCMake (not required, but is what I used to compile and generate, it made things easier than the GUI or VSCode)

## Environment Variables

### Add these paths to .bashrc, and .profiles if ccmake cannot find them still

*NOTE: Install stuff in their default directories to make this easier*

export CUDAToolkit_ROOT=/usr/local/cuda-12/bin

export Qt5Core_DIR=\~/Qt/5.15.2/gcc_64/lib/cmake/Qt5

export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12

export Torch_DIR=\~/libtorch/share/cmake/Torch

export OpenCV_DIR=\~/opencv-4.x/build

export VTK_DIR=\~/VTK-9.2.6/build

export CUDA_HOME="/usr/local/cuda-12/"

export PATH="/usr/local/cuda-12/bin:$PATH"

export PATH="~/opencv-4.x/modules/core/include:$PATH"


I'm not 100% sure on what *is* and what *isn't* required, but these are what I added and everything managed to run well. Exclude the above at your own risk.

## Setup

Assuming you have the above installed correctly:

1. Ensure that your [nvcc version](https://linuxhint.com/update_alternatives_ubuntu/) that is used in terminal is 12.0
2. In CCMake, enable advanced mode by hitting 't': 
    1. Ensure that CMAKE_CXX_COMPILER_AR and CMAKE_CXX_COMPILER_RANLIB are pointing to wherever gcc-ar-12 and gcc-ranlib-12 are on your machine. Mine were located at /usr/bin/
    2. Ensure all CUDA-related environment variables are pointing to the contents of the cuda-12 folder. Mine was located in /usr/local/cuda-12
    3. Ensure that the CUDA version being seen by ccmake (after configuring the first time) is 12.1. Not sure why it doesn't say 12.0, I think this is an artifact of the nvcc version included in CUDA 12.
    4. Ensure the rest of the environment variables are pointing to the right places (libtorch, Qt5, Torch, OpenCV, VTK)
3. Hit 'c' to configure, and if anything comes up red, review it and modify as needed. Configure until no *required* environment variables are missing.
4. Hit 'g' to generate, cross your fingers, and pray to a god of your choosing. 
5. Navigate to JTA-CMake/src/gui/ and run Joint-Track-Machine-Learning on Linux!