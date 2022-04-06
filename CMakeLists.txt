cmake_minimum_required(VERSION 3.19)
project(Joint-Track-Auto-GPU-CMake
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)


add_subdirectory(src)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) #fix to overall source directory
	add_subdirectory(packaging)
endif()
