# Set up Qt
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Set source and header directories
set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include/gui)
file(GLOB HEADER_FILES ${HEADER_DIR}/*.ui ${HEADER_DIR}/*.h)
set(RESOURCE_DIR ${PROJECT_SOURCE_DIR}/res)

# Define source files
set(SOURCES
    about.cpp
    calibration_manager.cpp
    controls.cpp
    drr_tool.cpp
    main.cpp
    mainscreen.cpp
    settings_control.cpp
    viewer.cpp
)

add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${HEADER_FILES}
    ${HEADER_DIR}/mainscreen.qrc
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/res
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/include/gpu
    ${CMAKE_SOURCE_DIR}/include/cost_functions
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    Qt5::OpenGL
    ${VTK_LIBRARIES}
    jtml_core
    jtml_gpu
)
