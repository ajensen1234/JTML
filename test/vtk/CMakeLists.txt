# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

add_executable(vtk-tester
    vtk_tester.cpp
    "../../src/gui/viewer.cpp" 
    ${PROJECT_SOURCE_DIR}/include/gui/viewer.h 

)
target_link_libraries(vtk-tester PRIVATE ${VTK_LIBRARIES})
target_link_libraries(vtk-tester PRIVATE jtml_gpu jtml_core)
target_link_libraries(vtk-tester PRIVATE ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(vtk-tester PRIVATE ${OpenCV_LIBRARIES})



target_include_directories(vtk-tester PUBLIC ${PROJECT_SOURCE_DIR}/include)