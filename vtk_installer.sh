#!/bin/bash
# Exit on any error
set -e

# Get absolute path of script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration variables with absolute paths
VTK_VERSION="9.7.0"
BASE_DIR="$SCRIPT_DIR/_deps/vtk"
SOURCE_DIR="$BASE_DIR/source"
BUILD_DIR="$BASE_DIR/build"
INSTALL_DIR="$BASE_DIR/install"
GIT_REPO="https://github.com/Kitware/VTK.git"

# Function to clean up on error
cleanup() {
	local exit_code=$?
	echo "An error occurred. Cleaning up..."
	if [ $exit_code -ne 0 ]; then
		echo "Build failed with exit code $exit_code"
		echo "Check the logs above for errors"
	fi
	exit $exit_code
}

# Set up error handling
trap cleanup ERR

# Create directory structure
echo "Creating build directory structure..."
mkdir -p "$SOURCE_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_DIR"

# Clone VTK if needed
if [ ! -d "$SOURCE_DIR/.git" ]; then
	echo "Cloning VTK repository..."
	git clone "$GIT_REPO" "$SOURCE_DIR"
	cd "$SOURCE_DIR"
	git checkout "v$VTK_VERSION"
	cd "$SCRIPT_DIR"
fi

# Enter build directory
cd "$BUILD_DIR"

# Detect number of CPU cores for parallel build
if command -v nproc >/dev/null 2>&1; then
	NUM_CORES=$(nproc)
else
	NUM_CORES=4 # Default if nproc not available
fi

echo "Configuring VTK with CMake..."
cmake "$SOURCE_DIR" \
	-DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
	-DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
	-DVTK_GROUP_ENABLE_Qt=YES \
	-DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES \
	-DQt6_DIR="$CONDA_PREFIX/lib/cmake/Qt6" \
	-DCMAKE_BUILD_TYPE=Release \
	-DVTK_MODULE_ENABLE_VTK_RenderingCore=YES \
	-DVTK_MODULE_ENABLE_VTK_RenderingOpenGL2=YES \
	-DVTK_MODULE_ENABLE_VTK_IOCore=YES \
	-DVTK_MODULE_ENABLE_VTK_IOImage=YES \
	-DVTK_MODULE_ENABLE_VTK_IOGeometry=YES \
	-DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES \
	-DVTK_MODULE_ENABLE_VTK_ChartsCore=YES \
	-DVTK_MODULE_ENABLE_VTK_ViewsContext2D=YES \
	-DVTK_MODULE_ENABLE_VTK_RenderingAnnotation=YES \
	-DVTK_MODULE_ENABLE_VTK_InteractionStyle=YES \
	-DVTK_USE_X=ON \
	-DVTK_USE_SDL2=ON \
	-DVTK_GROUP_ENABLE_Qt6=YES \
	-DVTK_USE_QT6=ON \
	-DVTK_QT_VERSION=6 \
	-DVTK_OPENGL_HAS_OSMESA=OFF \
	-DVTK_OPENGL_HAS_EGL=OFF \
	-DCMAKE_CXX_COMPILER=clang++ \
	-DCMAKE_C_COMPILER=clang \
	-G Ninja

echo "Building VTK..."
cmake --build "$BUILD_DIR" --parallel "$NUM_CORES"

echo "Installing VTK..."
cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR"

echo "VTK build and installation completed successfully!"
