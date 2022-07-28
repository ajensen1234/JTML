add_library(jtml_nfd STATIC
	nfd.cpp
	nfd_instance.cpp
	nfd_library.cpp)
	
target_link_libraries(jtml_nfd jtml_core)
target_link_libraries(jtml_nfd alglib)

target_include_directories(jtml_nfd PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(jtml_nfd PUBLIC ${PROJECT_SOURCE_DIR}/lib)