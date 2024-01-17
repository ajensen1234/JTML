/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#pragma once

/*Render Engine Header for Pose Class*/
#include "core/preprocessor-defs.h"
#include "gpu/render_engine.cuh"
/*Standard Library*/
#include <string>
#include <vector>

/*Class for Storing and Retrieving Pose linked to a unique Frame/Model Pair*/
class PoseMatrix {
   public:
    /*Blank Constructor/Destructor*/
    JTML_DLL PoseMatrix();
    JTML_DLL ~PoseMatrix();

    /*Add New Model to Pose Matrix*/
    JTML_DLL void AddModel(std::vector<gpu_cost_function::Pose>,
                           std::string model_name, bool is_principal_model);

    /*Get Model Pose (True if Successful, Else False) - Pose is Returned by
     * Passing via reference*/
    JTML_DLL bool GetModelPose(std::string model_name, int frame_index,
                               gpu_cost_function::Pose* pose_container);
    /*Get Principal Model Pose*/
    JTML_DLL bool GetModelPose(int frame_index,
                               gpu_cost_function::Pose* pose_container);
    /*Update Stored Pose for Principal Model at given frame*/
    JTML_DLL bool UpdatePrincipalModelPose(
        int frame_index, gpu_cost_function::Pose pose_container);

   private:
    /*Principal Model Name*/
    std::string principal_model_name_;

    /*Principal Model Lookup Index*/
    int principal_model_index_;

    /*Vector of All Models (Order Implies Index)*/
    std::vector<std::string> model_names_;

    /*Vector of All Model's Vector of Frame Poses
    Size of pose_matrix_ = # of models by # of frames*/
    std::vector<std::vector<gpu_cost_function::Pose>> pose_matrix_;
};
