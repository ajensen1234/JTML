
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




include(${VTK_USE_FILE})
target_link_libraries(JTA_Cost_Functions ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(JTA_Cost_Functions ${TORCH_LIBRARIES})
#target_link_libraries(JTA_Cost_Functions ${VTK_LIBRARIES})
target_link_libraries(JTA_Cost_Functions ${OpenCV_LIBRARIES})
target_link_libraries(JTA_Cost_Functions ${CUDA_LIBRARIES})


get_target_property(TARGET_LIBRARIES JTA_Cost_Functions LINK_LIBRARIES)
message(${TARGET_LIBRARIES})
list(REMOVE_ITEM ${TARGET_LIBRARIES} vtkRenderingQt vtkViewsGeovis vtkViewsQt vtkRenderingQt.lib vtkRenderingQt.lib)
set_property(TARGET JTA_Cost_Functions PROPERTY LINK_LIBRARIES ${TARGET_LIBRARIES})
message(${TARGET_LIBRARIES})

target_link_libraries(JTA_Cost_Functions jtml_gpu)
target_include_directories(JTA_Cost_Functions PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(JTA_Cost_Functions PUBLIC ${HEADER_DIR})




