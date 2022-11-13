add_library(DIRECT_DILATION SHARED
    DIRECT_DILATION.cpp
    DIRECT_DILATION.h)

	
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
target_link_libraries(DIRECT_DILATION ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})



target_link_libraries(DIRECT_DILATION JTA_Cost_Functions jtml_gpu)

target_include_directories(DIRECT_DILATION
	PUBLIC ${PROJECT_SOURCE_DIR}/include)

