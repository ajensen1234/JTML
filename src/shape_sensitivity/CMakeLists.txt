add_executable(shape-sensitivity
  "main.cpp"
  "descriptors.h"
  "descriptors.cpp"
  "art.cu"
  "art.cuh"
)


set(QT_VERSION 5)
set(REQUIRED_LIBS Core Gui Widgets)
set(REQUIRED_LIBS_QUALIFIED Qt5::Core Qt5::Gui Qt5::Widgets)

find_package(Qt${QT_VERSION} COMPONENTS ${REQUIRED_LIBS} REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

target_link_libraries(shape-sensitivity ${JTA_LIBS})

# Adding Include Directories
target_include_directories(shape-sensitivity PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(shape-sensitivity PUBLIC ${CUDA_INCLUDE_DIRS})
