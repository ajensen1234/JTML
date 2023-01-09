set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/gui)
file(GLOB HEADER_FILES ${HEADER_DIR}/*.ui ${HEADER_DIR}/*.h)
set(res ${PROJECT_SOURCE_DIR}/res)


add_executable(${PROJECT_NAME}
	about.cpp
	controls.cpp
	drr_tool.cpp
	main.cpp
	mainscreen.cpp
	settings_control.cpp
	
	viewer.cpp
	${HEADER_FILES}
	${HEADER_DIR}/mainscreen.qrc)


target_link_libraries(${PROJECT_NAME} ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(${PROJECT_NAME} jtml_core)
target_link_libraries(${PROJECT_NAME} jtml_nfd)
target_link_libraries(${PROJECT_NAME} ${JTA_LIBS})
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})

vtk_module_autoinit(
    TARGETS ${PROJECT_NAME}
    MODULES ${VTK_LIBRARIES}
)




target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/res)