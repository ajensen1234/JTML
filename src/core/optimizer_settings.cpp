// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*Optimizer Settings Structure*/
#include "core/optimizer_settings.h"

/*Default Values*/
#include "core/settings_constants.h"

OptimizerSettings::OptimizerSettings() {
    /*Variables*/
    /*Trunk*/
    trunk_range = TRUNK_RANGE;
    trunk_budget = TRUNK_BUDGET;

    /*BrancheS*/
    branch_range = BRANCH_RANGE;
    branch_budget = BRANCH_BUDGET;
    number_branches = NUMBER_BRANCHES;

    /*Leaf Search*/
    leaf_range = Z_SEARCH_RANGE;
    leaf_budget = Z_SEARCH_BUDGET;

    /*Optimizer Settings Control Window Other Stuff*/
    enable_branch_ = ENABLE_BRANCH;
    enable_leaf_ = ENABLE_Z;
}

OptimizerSettings::~OptimizerSettings() {}
