/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#pragma once

#include <cmath>
#include <sstream>
#include <vector>

#include "core/data_structures_6D.h"

Point6D compute_mirror_pose(Point6D point);
void matmult4(float ans[4][4], float matrix1[4][4], float matrix2[4][4]);
void matmult3(float ans[3][3], const float matrix1[3][3],
              const float matrix2[3][3]);
void invert_transform(float result[4][4], const float tran[4][4]);
void equivalent_axis_angle_rotation(float rot[3][3], const float m[3],
                                    const float angle);
void cross_product(float CP[3], const float v1[3], const float v2[3]);
void dot_product(float& result, const float vector1[3], const float vector2[3]);
void rotation_matrix(float R[3][3], Point6D pose);
void create_312_transform(float transform[4][4], Point6D pose);
void getRotations312(float& xr, float& yr, float& zr, const float Rot[3][3]);

void copy_matrix_by_value(float (&new_matrix)[3][3],
                          const float (&old_matrix)[3][3]);
void create_vector_of_poses(std::vector<Point6D>& pose_list, Point6D pose,
                            int numPoses);

template <typename T>
std::vector<double> static linspace(T start_in, T end_in, int num_in);
