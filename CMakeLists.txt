cmake_minimum_required(VERSION 3.19)
project(Joint-Track-Machine-Learning
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
add_subdirectory(nfd)
add_subdirectory(JTML)

target_link_libraries(Joint-Track-Machine-Learning JTML_NFD)
add_subdirectory(packaging)