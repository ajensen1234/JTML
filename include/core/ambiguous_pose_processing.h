/**
 * @file ambiguous_pose_processing.h
 * @author Andrew Jensen (andrewjensen321@gmail.com)
 * @brief This is a file that handles all of the different types of
 * post-processing that we can do for each of the input poses. The main goal is
 * to help with symmetry traps.
 * @version 0.1
 * @date 2022-09-30
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once
#include "core/data_structures_6D.h"
#include "core/sym_trap_functions.h"
/**
 * @brief This function will take a relative pose between a femur and tibia and
 * choose either the current tibial pose or it's mirror.
 *
 * @param femur_pose This is a 6D representation of the femur pose for the
 * current frame.
 * @param tibia_pose This is a 6D representation of the tibial pose for the
 * current frame.
 * @return Point6D. This function will return the (hopefully) correct tibial
 * pose from the current pose and the mirror.
 */
Point6D tibial_pose_selector(Point6D& femur_pose, Point6D& tibia_pose);

/**
 * @brief This is a function that calculates the varus/valgus angle between
 * tibia and the femur.
 *
 * @param femur_pose The femur's 6D pose.
 * @param tibia_pose  The tibia's 6D pose.
 * @return float The varus/valgus angle between the two.
 */
float varus_valgus_calculation(Point6D& femur_pose, Point6D& tibia_pose);