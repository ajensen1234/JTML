#pragma once
#include "core/sym_trap_functions.h"
#include "core/data_structures_6D.h"

Point6D tibial_pose_selector(Point6D& femur_pose, Point6D& tibia_pose);
float varus_valgus_calculation(Point6D& femur_pose, Point6D& tibia_pose);