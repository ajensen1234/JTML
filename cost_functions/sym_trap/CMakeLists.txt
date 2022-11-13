add_library(sym_trap_fn SHARED
    sym_trap_fn.cpp
    sym_trap_fn.h)


	

target_link_libraries(sym_trap_fn ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(sym_trap_fn JTA_Cost_Functions jtml_gpu)

target_include_directories(sym_trap_fn PUBLIC ${PROJECT_SOURCE_DIR}/include)

