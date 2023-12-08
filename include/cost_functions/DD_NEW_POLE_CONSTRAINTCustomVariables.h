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
double x_loc_non;
double z_loc_non;
