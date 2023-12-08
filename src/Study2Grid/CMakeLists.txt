# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.
add_executable (Study2Grid-Cmake 
					"BasicLinearAlgebraStructures.cpp"
					"BasicLinearAlgebraStructures.h"
					"FileParsers.cpp"
					"FileParsers.h"
					"ImageInfo.cpp"
					"ImageInfo.h"
					"KPReader.cpp"
					"KPReader.h"
					"main.cpp"
					"STLReader.cpp"
					"STLReader.h"
					"Study.cpp"
					"Study.h")

# TODO: Add tests and install targets if needed.
set(QT_VERSION 5)
set(REQUIRED_LIBS Core Gui Widgets)
set(REQUIRED_LIBS_QUALIFIED Qt5::Core Qt5::Gui Qt5::Widgets)

find_package(Qt${QT_VERSION} COMPONENTS ${REQUIRED_LIBS} REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

target_link_libraries(Study2Grid-Cmake ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(Study2Grid-Cmake ${OpenCV_LIBRARIES})
target_link_libraries(Study2Grid-Cmake ${CUDA_LIBRARIES})
target_compile_definitions(Study2Grid-Cmake PRIVATE _HAS_STD_BYTE=0)



#set(S2G_LIBS ${COSTFUNCTIONTOOLS_LIBRARY_PATH} ${jtml_gpu_library_path})
#set(S2G_INCLUDES ${COSTFUNCTIONTOOLS_INCLUDE_PATH} ${jtml_gpu_include_path} ${jtml_core_include_path})

#target_link_libraries(Study2Grid-Cmake ${S2G_LIBS})
target_link_libraries(Study2Grid-Cmake ${JTA_LIBS})
#target_include_directories(Study2Grid-Cmake PUBLIC ${S2G_INCLUDES})
target_include_directories(Study2Grid-Cmake PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
