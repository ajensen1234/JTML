add_executable(shape-sensitivity
  "main.cpp"
  "descriptors.h"
  "descriptors.cpp"
  "art.cu"
  "art.cuh"
  "matrix_vector_utils.cpp"
  "matrix_vector_utils.h"
)


set(QT_VERSION 5)
set(REQUIRED_LIBS Core Gui Widgets)
set(REQUIRED_LIBS_QUALIFIED Qt5::Core Qt5::Gui Qt5::Widgets)


target_link_libraries(shape-sensitivity ${VTK_LIBRARIES})

target_link_libraries(shape-sensitivity ${JTA_LIBS})
target_link_libraries(shape-sensitivity jtml_core)
target_link_libraries(shape-sensitivity jtml_gpu)
# Adding Include Directories
target_include_directories(shape-sensitivity PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(shape-sensitivity PUBLIC ${CUDA_INCLUDE_DIRS})
