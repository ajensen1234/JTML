# General Template for Tests
add_executable(nfd_speed_test
    nfd_speed_test.cpp)

target_link_libraries(nfd_speed_test
    jtml_core
    jtml_nfd)
