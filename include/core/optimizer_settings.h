/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#ifndef OPTIMIZER_SETTINGS_H
#define OPTIMIZER_SETTINGS_H
/*Data Structures Used by All*/
#include "data_structures_6D.h"

/*Declare as MetaType So Can Send*/
#include <QMetaType>

/*Structure that Stores the Optimizer Settings Except for the Cost Function
 * Information which is stored in the 3 Cost Function Managers*/
struct OptimizerSettings {
    /*Constructor Destructor*/
    OptimizerSettings();
    ~OptimizerSettings();

    /*Variables*/
    /*Trunk*/
    Point6D trunk_range;
    int trunk_budget;

    /*Branch*/
    Point6D branch_range;
    int branch_budget;
    int number_branches;

    /*Leaf*/
    Point6D leaf_range;
    int leaf_budget;

    /*Optimizer Settings Which Stages Are on (trunk always on)*/
    bool enable_branch_;
    bool enable_leaf_;
};

Q_DECLARE_METATYPE(OptimizerSettings);

#endif /*OPTIMIZER_SETTINGS_H*/
