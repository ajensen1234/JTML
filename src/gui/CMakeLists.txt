set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/gui)
file(GLOB HEADER_FILES ${HEADER_DIR}/*.ui ${HEADER_DIR}/*.h)
set(res ${PROJECT_SOURCE_DIR}/res)
message("HELLO")


add_executable(${PROJECT_NAME}
	about.cpp
	controls.cpp
	drr_tool.cpp
	main.cpp
	mainscreen.cpp
	settings_control.cpp
	sym_trap.cpp
	viewer.cpp
	${HEADER_FILES}
	${HEADER_DIR}/mainscreen.qrc)


target_link_libraries(${PROJECT_NAME} ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(${PROJECT_NAME} jtml_core)
target_link_libraries(${PROJECT_NAME} jtml_nfd)
target_link_libraries(${PROJECT_NAME} ${JTA_LIBS})
#target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} vtkRenderingLabel)


get_target_property(TARGET_LIBRARIES ${PROJECT_NAME} LINK_LIBRARIES)
message(${TARGET_LIBRARIES})
list(REMOVE_ITEM ${TARGET_LIBRARIES} vtkRenderingQt vtkViewsGeovis vtkViewsQt)
message(${TARGET_LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY LINK_LIBRARIES ${TARGET_LIBRARIES})

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/res)