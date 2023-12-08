// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "core/ambiguous_pose_processing.h"

Point6D tibial_pose_selector(Point6D& femur_pose, Point6D& tibia_pose) {
    Point6D tibial_dual_pose = compute_mirror_pose(
        tibia_pose);  // calculate the mirror pose for the tibia

    float vv_original = varus_valgus_calculation(femur_pose, tibia_pose);
    float vv_mirror = varus_valgus_calculation(femur_pose, tibial_dual_pose);

    std::cout << vv_original << std::endl;
    std::cout << vv_mirror << std::endl;
    // if normal pose has a greater VV, return the dual pose.
    if (vv_original > vv_mirror) return tibial_dual_pose;
    // if normal pose has lower VV than mirror pose, return the normal pose.
    return tibia_pose;
}

float varus_valgus_calculation(Point6D& femur_pose, Point6D& tibia_pose) {
    float T_x_fem[4][4], T_x_tib[4][4], T_fem_x[4][4],
        T_fem_tib[4][4];  // declaring transformation matrices
    create_312_transform(T_x_fem,
                         femur_pose);  // defining transformation matrices
    create_312_transform(T_x_tib, tibia_pose);
    invert_transform(T_fem_x, T_x_fem);  // inverting the femoral transformation
    matmult4(T_fem_tib, T_fem_x, T_x_tib);
    // multiplying transformation matrices to get tibial coordinates in the
    // femoral reference frame
    return abs(asin(T_fem_tib[2][1]));  // returning the absolute value of the
                                        // Varus/valgus rotation
}
