set(QT_VERSION 5)
set(REQUIRED_LIBS Core Gui Widgets)
set(REQUIRED_LIBS_QUALIFIED Qt5::Core Qt5::Gui Qt5::Widgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

add_executable(${PROJECT_NAME}
        direct_data_storage.h
        direct_data_storage.cpp
        data_structures_6D.cpp
        mainscreen.qrc
        location_storage.h
        calibration.h
        settings_control.ui
        about.ui
        stl_reader.cpp
        PropertySheet.props
        STLReader.h
        resource3.h
        controls.h
        mainscreen.h
        model.h
        optimizer_settings.cpp
        mainscreen_size_constants.h
        JointTrackAuto.rc
        optimizer_manager.cpp
        about.h
        stl_reader.h
        controls.ui
        resource4.h
        drr_interactor.h
        settings_control.cpp
        data_structures_6D.h
        frame.h
        settings_control.h
        resource2.h
        drr_tool.ui
        resource.h
        settings_window_size_constants.h
        resource1.h
        settings_constants.h
        mainscreen.cpp
        mainscreen.ui
        drr_tool.cpp
        drr_tool.h
        model.cpp
        metric_enum.h
        optimizer_manager.h
        controls.cpp
        main.cpp
        optimizer_settings.h
        location_storage.cpp
        about.cpp
        frame.cpp
        STLReader.cpp
        interactor.h
        cost_function_selection.ui
        sym_trap.cpp
        sym_trap.h
        sym_trap.ui)



if (NOT CMAKE_PREFIX_PATH)
    message(WARNING "CMAKE_PREFIX_PATH is not defined, you may need to set it "
            "(-DCMAKE_PREFIX_PATH=\"path/to/Qt/lib/cmake\" or -DCMAKE_PREFIX_PATH=/usr/include/{host}/qt{version}/ on Ubuntu)")
endif ()

find_package(Qt${QT_VERSION} COMPONENTS ${REQUIRED_LIBS} REQUIRED)
find_package(Torch REQUIRED)
find_package(VTK 7.1.1 REQUIRED)
find_package(cuda REQUIRED)
find_package(OpenCV REQUIRED)
include(${VTK_USE_FILE})



target_link_libraries(${PROJECT_NAME} ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${cuda_LIBRARIES})


find_path(COSTFUNCTIONTOOLS_INCLUDE_PATH
	NO_DEFAULT_PATH
	REQUIRED)
find_path(JTA_COST_FUNCTIONS_INCLUDE_PATH
	NO_DEFAULT_PATH)

find_file(COSTFUNCTIONTOOLS_LIBRARY_PATH
	NO_DEFAULT_PATH
	REQUIRED)
find_file(JTA_COST_FUNCTIONS_LIBRARY_PATH
	NO_DEFAULT_PATH
	REQUIRED)

set(JTA_LIBS ${COSTFUNCTIONTOOLS_LIBRARY_PATH} ${JTA_COST_FUNCTIONS_LIBRARY_PATH})
set(JTA_INCLUDES ${JTA_COST_FUNCTIONS_INCLUDE_PATH} ${COSTFUNCTIONTOOLS_INCLUDE_PATH})

target_link_libraries(${PROJECT_NAME} ${JTA_LIBS})

target_include_directories(${PROJECT_NAME}
	PUBLIC ${JTA_INCLUDES})


# need to remove a pervasive vtk lib
get_target_property(LINKED_LIBS ${PROJECT_NAME} LINK_LIBRARIES)

message("Linked Libs: ${LINKED_LIBS}")
list(REMOVE_ITEM LINKED_LIBS vtkViewsGeovis)
set_property(TARGET ${PROJECT_NAME} PROPERTY LINK_LIBRARIES ${LINKED_LIBS})
get_target_property(NEW_LINKED ${PROJECT_NAME} LINK_LIBRARIES)
message("Linked Libs: ${NEW_LINKED}")




#comment
