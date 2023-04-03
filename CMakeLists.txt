cmake_minimum_required(VERSION 3.19)
project(Joint-Track-Machine-Learning
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(QT_VERSION 5)
set(REQUIRED_LIBS Core Gui Widgets)
set(REQUIRED_LIBS_QUALIFIED Qt5::Core Qt5::Gui Qt5::Widgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Finding Required Packages
find_package(Qt${QT_VERSION} COMPONENTS ${REQUIRED_LIBS} REQUIRED)
find_package(Torch REQUIRED PATHS ~/libtorch/share/cmake/Torch/)
#find_package(VTK 7.1.1 REQUIRED)
#find_package(cuda REQUIRED)
find_package(OpenCV REQUIRED)

find_package(VTK 9 REQUIRED COMPONENTS
RenderingCore
RenderingOpenGL2
IOCore
IOImage
IOGeometry
GUISupportQt
ChartsCore
ViewsContext2D
RenderingAnnotation
InteractionStyle
)


set(JTA_LIBS jtml_gpu JTA_Cost_Functions)
#set(JTA_INCLUDES ${JTA_COST_FUNCTIONS_INCLUDE_PATH} ${COSTFUNCTIONTOOLS_INCLUDE_PATH})

# Adding Subdirectories
add_subdirectory(src)
#add_subdirectory(test)
add_subdirectory(lib)
add_subdirectory(packaging)

#enable_testing()

