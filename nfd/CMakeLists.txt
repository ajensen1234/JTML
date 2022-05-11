set(QT_VERSION 5)
set(REQUIRED_LIBS Core)
set(REQUIRED_LIBS_QUALIFIED Qt5::Core)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

add_library(JTML_NFD STATIC
	nfd.cpp
	nfd.h
	nfd_instance.h
	nfd_instance.cpp
	nfd_library.cpp
	nfd_library.h
	)

set(CMAKE_CXX_STANDARD 17)

	
	
find_package(OpenCV)
find_package(Qt${QT_VERSION} COMPONENTS ${REQUIRED_LIBS} REQUIRED)
find_package(VTK 7.1.1 REQUIRED)
find_package(CUDA REQUIRED)
find_package(fitpackpp)
include(${VTK_USE_FILE})
find_path(COSTFUNCTIONTOOLS_INCLUDE_PATH
	NO_DEFAULT_PATH
	REQUIRED)

find_file(COSTFUNCTIONTOOLS_LIBRARY_PATH
	NO_DEFAULT_PATH
	REQUIRED)

target_link_libraries(JTML_NFD ${REQUIRED_LIBS_QUALIFIED})
target_link_libraries(JTML_NFD ${VTK_LIBRARIES})
target_link_libraries(JTML_NFD ${COSTFUNCTIONTOOLS_LIBRARY_PATH})
target_link_libraries(JTML_NFD ${OpenCV_LIBRARIES})
target_link_libraries(JTML_NFD ${CUDA_LIBRARIES})
target_link_libraries(JTML_NFD fitpackpp::fitpackpp)
target_include_directories(JTML_NFD 
			 PUBLIC ${COSTFUNCTIONTOOLS_INCLUDE_PATH})