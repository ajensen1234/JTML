# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# General Template for Tests
add_executable(nfd_speed_test
    nfd_speed_test.cpp)


target_link_libraries(nfd_speed_test PRIVATE
    jtml_nfd)
