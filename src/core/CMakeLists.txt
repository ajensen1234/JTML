set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/core)
file(GLOB HEADER_FILES ${HEADER_DIR}/*.h)


add_library(jtml_core STATIC
	data_structures_6D.cpp
	direct_data_storage.cpp
	frame.cpp
	location_storage.cpp
	model.cpp
	optimizer_manager.cpp
	optimizer_settings.cpp
	stl_reader.cpp
	STLReader.cpp
	${HEADER_FILES}
	)

# Linking Necessary Libraries
target_link_libraries(jtml_core ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(jtml_core ${TORCH_LIBRARIES})
target_link_libraries(jtml_core ${VTK_LIBRARIES})
target_link_libraries(jtml_core ${OpenCV_LIBRARIES})
target_link_libraries(jtml_core ${cuda_LIBRARIES})
target_link_libraries(jtml_core ${JTA_LIBS})
#target_link_libraries(jtml_core ${PROJECT_NAME})

# Linking In House LIbs and Directories
target_include_directories(jtml_core
	PUBLIC ${JTA_INCLUDES})
	target_link_libraries(jtml_core ${JTA_LIBS})

# Adding Include Directories
target_include_directories(jtml_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(jtml_core PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(jtml_core PUBLIC ${CMAKE_BINARY_DIR}/src/gui/${PROJECT_NAME}_autogen/include_Release)
message(${CMAKE_BINARY_DIR}/src/gui/${PROJECT_NAME}_autogen/include_Release)