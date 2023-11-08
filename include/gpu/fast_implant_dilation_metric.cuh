#pragma once

#include "gpu/gpu_metrics.cuh"

/*Cuda*/
#include "cuda_runtime.h"
#include "cuda.h"

/*Grayscale Colors*/
#include "pixel_grayscale_colors.h"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"


__global__ void FastImplantDilationMetric_ResetPixelScoreKernel(int* dev_pixel_score);

__global__ void FastImplantDilationMetric_EdgeKernel(unsigned char* dev_image, int sub_left_x, int sub_bottom_y,
                                                     int sub_right_x, int sub_top_y, int width, int dilation);
