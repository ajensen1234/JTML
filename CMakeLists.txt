cmake_minimum_required(VERSION 3.19)
project(Joint-Track-Machine-Learning
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(nfd)
add_subdirectory(JTML)
add_subdirectory(alglib)

target_link_libraries(Joint-Track-Machine-Learning JTML_NFD)
target_link_libraries(JTML_NFD alglib)
add_subdirectory(packaging)