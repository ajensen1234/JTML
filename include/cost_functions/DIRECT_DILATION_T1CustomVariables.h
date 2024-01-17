/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#pragma once
/****************Headers*************/
/*Cost Function Tools Library*/
#include "gpu/gpu_dilated_frame.cuh"
#include "gpu/gpu_edge_frame.cuh"
#include "gpu/gpu_frame.cuh"
#include "gpu/gpu_image.cuh"
#include "gpu/gpu_intensity_frame.cuh"
#include "gpu/gpu_metrics.cuh"
#include "gpu/gpu_model.cuh"
#include "gpu/render_engine.cuh"
/*Stage Enum*/
#include "Stage.h"
/*Parameter Class*/
#include "Parameter.h"
#include "core/preprocessor-defs.h"

/****************Begin Custom Variables*************/
/*Sum of the white pixels in the current dilation comparison image*/
int DIRECT_DILATION_T1_current_white_pix_sum_dilated_comparison_image_A_;
int DIRECT_DILATION_T1_current_white_pix_sum_dilated_comparison_image_B_;
int DIRECT_DILATION_T1_current_dilation_parameter;
