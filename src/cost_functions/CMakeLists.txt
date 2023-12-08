# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0


set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/cost_functions)
file(GLOB HEADER_FILES ${HEADER_DIR}/*.h)

add_library(JTA_Cost_Functions
		SHARED
		CostFunction.cpp
		CostFunctionManager.cpp
		DD_NEW_POLE_CONSTRAINT.cpp
		DIRECT_DILATION.cpp
		DIRECT_DILATION_POLE_CONSTRAINT.cpp
		DIRECT_DILATION_SAME_Z.cpp
		DIRECT_DILATION_T1.cpp
		DIRECT_MAHFOUZ.cpp
		sym_trap_function.cpp
		${HEADER_FILES})




target_link_libraries(JTA_Cost_Functions ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(JTA_Cost_Functions ${TORCH_LIBRARIES})
#target_link_libraries(JTA_Cost_Functions ${VTK_LIBRARIES})
target_link_libraries(JTA_Cost_Functions ${OpenCV_LIBRARIES})
target_link_libraries(JTA_Cost_Functions ${CUDA_LIBRARIES})


target_link_libraries(JTA_Cost_Functions jtml_gpu)
target_include_directories(JTA_Cost_Functions PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(JTA_Cost_Functions PUBLIC ${HEADER_DIR})




