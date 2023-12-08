/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#pragma once

#include <QModelIndex>
#include <QString>
#include <gpu_dilated_frame.cuh>
#include <gpu_edge_frame.cuh>
#include <gpu_intensity_frame.cuh>
#include <gpu_metrics.cuh>
#include <gpu_model.cuh>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "core/calibration.h"
#include "core/frame.h"
#include "core/model.h"
#include "cufft.h"
#include "nfd_instance.h"
#include "nfd_library.h"

using namespace gpu_cost_function;

class JTML_NFD {
   public:
    JTML_NFD();
    ~JTML_NFD();
    bool Initialize(Calibration cal_file, std::vector<Model> model_list,
                    std::vector<Frame> frames_list,
                    QModelIndexList selected_models,
                    unsigned int primary_model_index, QString error_message);

    void Run();

   private:
    bool successful_initialization_;
    Calibration calibration_;
    std::vector<Frame> frames_;

    GPUModel* gpu_principal_model_;

    Model primary_model_;
    std::vector<Model> all_models_;
};