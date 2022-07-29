set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/gui)
file(GLOB HEADER_FILES ${HEADER_DIR}/*.ui ${HEADER_DIR}/*.h)
set(res ${PROJECT_SOURCE_DIR}/res)

qt_add_resources(${PROJECT_NAME} "app_images"
	PREFIX "Menu Icons"
	FILES
		${res}/normaluppink.png
		${res}/normalup.png
		${res}/camerared.png
		${res}/cameragreen.png
		${res}/camerapink.png
		${res}/camerayellow.png
		${res}/save_icon.png
		${res}/stop_icon.png
		${res}/selection_icon.png
		${res}/load_icon.png
		${res}/controls_icon.png
		${res}/center_placement.png
		${res}/help_icon.png
		${res}/optimizer_settings_icon.png
		${res}/quit_icon.png
	PREFIX "Desktop Icon"
	FILES
		${res}/jta_icon.png
		${res}/jta_dime_icon.png
)

add_executable(${PROJECT_NAME}
	about.cpp
	controls.cpp
	drr_tool.cpp
	main.cpp
	mainscreen.cpp
	settings_control.cpp
	sym_trap.cpp
	${HEADER_FILES}
	${HEADER_DIR}/mainscreen.qrc)

target_link_libraries(${PROJECT_NAME} ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(${PROJECT_NAME} jtml_core)
target_link_libraries(${PROJECT_NAME} jtml_nfd)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/res)