# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/core)
file(GLOB HEADER_FILES 
    ${HEADER_DIR}/*.h
    ${HEADER_DIR}/io/*.h
)

file(GLOB SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/io/*.cpp
)

add_library(jtml_core STATIC
    ${SOURCE_FILES}
    ${HEADER_FILES}
)

# Linking Necessary Libraries
target_link_libraries(jtml_core ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(jtml_core ${TORCH_LIBRARIES})
target_link_libraries(jtml_core ${VTK_LIBRARIES})
target_link_libraries(jtml_core ${OpenCV_LIBRARIES})
target_link_libraries(jtml_core ${CUDA_LIBRARIES})

# Linking In House Libs and Directories
target_link_libraries(jtml_core ${JTA_LIBS})

# Adding Include Directories
target_include_directories(jtml_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(jtml_core PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(jtml_core PUBLIC ${PROJECT_SOURCE_DIR}/lib/)