add_library(jtml_nfd STATIC
	nfd.cpp
	nfd_instance.cpp
	nfd_library.cpp)
	
target_link_libraries(jtml_nfd jtml_core)
target_link_libraries(jtml_nfd alglib)

get_target_property(TARGET_LIBRARIES jtml_nfd LINK_LIBRARIES)
list(REMOVE_ITEM TARGET_LIBRARIES vtkRenderingQt vtkViewsGeovis vtkViewsQt)
set_property(TARGET jtml_nfd PROPERTY LINK_LIBRARIES ${TARGET_LIBRARIES})

target_include_directories(jtml_nfd PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(jtml_nfd PUBLIC ${PROJECT_SOURCE_DIR}/lib)