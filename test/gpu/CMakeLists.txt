add_executable(render-engine-ownership-test
    render_engine_ownership_test.cpp
)

target_link_libraries(render-engine-ownership-test PRIVATE jtml_gpu)

target_include_directories(render-engine-ownership-test PUBLIC ${PROJECT_SOURCE_DIR}/include)
