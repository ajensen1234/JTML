# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

# 003 U3: jtml_domain is the PURE (Qt/GPU-free) core layer -- the intended
# future Rust/FFI surface. Grep-gate: this lib must contain no QObject/vtk/
# cuda/torch/opencv dependency (it is the domain-purity boundary). It links
# nothing heavy.
set(CMAKE_AUTOMOC ON)

file(GLOB HEADER_FILES ${PROJECT_SOURCE_DIR}/include/domain/*.h)

add_library(jtml_domain STATIC
    ambiguous_pose_processing.cpp
    data_structures_6D.cpp
    direct_data_storage.cpp
    direct_optimizer.cpp
    model_list_builder.cpp
    optimize_intent_controller.cpp
    pose_file_io.cpp
    session_state.cpp
    sym_trap_functions.cpp
    ${HEADER_FILES}
)

# Adding Include Directories
target_include_directories(jtml_domain PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(jtml_domain PUBLIC ${PROJECT_SOURCE_DIR}/lib/)
