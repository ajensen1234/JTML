cmake_minimum_required(VERSION 3.19)
project(Joint-Track-Auto-GPU-CMake
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(JTA_Cost_Functions_DIR "C:/JTA_libraries/JTA_Cost_Functions/JTA_Cost_Functions")
set(CostFunctionTools_DIR "C:/JTA_Libraries/CostFunctionTools/CostFunctionTools")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(QT_VERSION 5)
set(REQUIRED_LIBS Core Gui Widgets)
set(REQUIRED_LIBS_QUALIFIED Qt5::Core Qt5::Gui Qt5::Widgets)

add_executable(${PROJECT_NAME}
        direct_data_storage.h
        direct_data_storage.cpp
        gpu_dilated_frame.cu
        data_structures_6D.cpp
        mainscreen.qrc
        location_storage.h
        calibration.h
        settings_control.ui
        about.ui
        gpu_model.cu
        stl_reader.cpp
        PropertySheet.props
        gpu_image_functions.cu
        gpu_intensity_frame.cu
        STLReader.h
        resource3.h
        controls.h
        mainscreen.h
        model.h
        optimizer_settings.cpp
        mainscreen_size_constants.h
        JointTrackAuto.rc
        optimizer_manager.cpp
        gpu_edge_frame.cu
        about.h
        stl_reader.h
        controls.ui
        resource4.h
        drr_interactor.h
        settings_control.cpp
        data_structures_6D.h
        frame.h
        settings_control.h
        gpu_image.cu
        resource2.h
        drr_tool.ui
        resource.h
        settings_window_size_constants.h
        resource1.h
        gpu_metrics.cu
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
        gpu_frame.cu
        STLReader.cpp
        interactor.h
        cost_function_selection.ui)

if (NOT CMAKE_PREFIX_PATH)
    message(WARNING "CMAKE_PREFIX_PATH is not defined, you may need to set it "
            "(-DCMAKE_PREFIX_PATH=\"path/to/Qt/lib/cmake\" or -DCMAKE_PREFIX_PATH=/usr/include/{host}/qt{version}/ on Ubuntu)")
endif ()

find_package(Qt${QT_VERSION} COMPONENTS ${REQUIRED_LIBS} REQUIRED)
find_package(Torch REQUIRED)
find_package(VTK REQUIRED)
find_package(cuda REQUIRED)
find_package(OpenCV REQUIRED)
include(${VTK_USE_FILE})
target_link_libraries(${PROJECT_NAME} ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${cuda_LIBRARIES})
target_link_libraries(${PROJECT_NAME} C:/Users/pflood/Desktop/JTAML/CostFunctionTools/x64/Release/CostFunctionTools.lib
        C:/Users/pflood/Desktop/JTAML/JTA_Cost_Functions/x64/Release/JTA_Cost_Functions.lib)
include_directories(BEFORE ${CostFunctionTools_DIR} ${JTA_Cost_Functions_DIR})