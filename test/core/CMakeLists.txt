add_executable(direct-data-storage-ownership-test
    direct_data_storage_ownership_test.cpp
)

target_link_libraries(direct-data-storage-ownership-test PRIVATE jtml_core)

target_include_directories(direct-data-storage-ownership-test PUBLIC ${PROJECT_SOURCE_DIR}/include)
