/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*Constants for optimizer default settings and version*/

#ifndef SETTINGS_CONSTANTS_H
#define SETTINGS_CONSTANTS_H

/*Data Structures Used by All*/
#include "data_structures_6D.h"

/*Metric Type Enumerator*/
#include "metric_enum.h"

/*Version Numbers*/
const int VER_FIRST_NUM = 3;
const int VER_MIDDLE_NUM = 4;
const int VER_LAST_NUM = 0;

/*Variables*/
/*Trunk*/
const Point6D TRUNK_RANGE = Point6D(35, 35, 35, 35, 35, 35);
const int TRUNK_BUDGET = 20000;
const int TRUNK_DILATION = 6;

/*BrancheS*/
const Point6D BRANCH_RANGE = Point6D(15, 15, 25, 25, 25, 25);
const int BRANCH_BUDGET = 5000;
const int NUMBER_BRANCHES = 2;
const int BRANCH_DILATION_DECREASE = 2;

/*Z- SeaRCH*/
const Point6D Z_SEARCH_RANGE = Point6D(3, 3, 15, 3, 3, 3);
const int Z_SEARCH_BUDGET = 5000;
const int Z_SEARCH_DILATION = 1;

/*Display Current Optimum During Optimization*/
const bool DISPLAY_CURRENT_OPTIMUM = true;

/*Optimizer Settings Control Window Other Stuff*/
const bool ENABLE_BRANCH = true;
const bool ENABLE_Z = true;
const bool SCALE_TRUNK = false;
const double SCALE_TRUNK_VALUE = 0.5;

/*Edge Constants Save*/
const int APERTURE = 3;
const int LOW_THRESH = 40;
const int HIGH_THRESH = 120;

/*Intensity Image Uses Black Silhouette? (True = Black, False = White)*/
const bool BLACK_SILHOUETTE = true;

/*Edge and Intensity Weights If Combined*/
const double INTENSITY_WEIGHT = 1;
const double EDGE_WEIGHT = 1;

#endif /* SETTINGS_CONSTANTS_H */