#pragma once

#include "compute/gpu_metrics.cuh"

/*Cuda*/
#include "cuda.h"
#include "cuda_runtime.h"

/*Grayscale Colors*/
#include "pixel_grayscale_colors.h"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"

__global__ void FastImplantDilationMetric_ResetPixelScoreKernel(
    int* dev_pixel_score);

__global__ void FastImplantDilationMetric_EdgeKernel(
    unsigned char* dev_image,
    int sub_left_x,
    int sub_bottom_y,
    int sub_right_x,
    int sub_top_y,
    int width,
    int dilation);

__global__ void FastImplantDilationMetric_DilateKernel(
    unsigned char* dev_image,
    int width,
    int height,
    int sub_left_x,
    int sub_bottom_y,
    int sub_cropped_width,
    int dilation);

__global__ void FastImplantDilationMetric_DifferenceKernel(
    unsigned char* dev_image,
    unsigned char* dev_comparison_image,
    int* result,
    int width,
    int height,
    int diff_kernel_left_x,
    int diff_kernel_bottom_y,
    int diff_kernel_cropped_width);

// Distance map metric kernels (declared for U4 device-AABB path)
__global__ void DistanceMapMetric_Kernel(
    unsigned char* projected_image,
    unsigned char* distance_map,
    int* distance_map_score,
    int* edge_pixel_count,
    int width,
    int height,
    int diff_kernel_left_x,
    int diff_kernel_bottom_y,
    int diff_kernel_cropped_width);

__global__ void DistanceMapMetric_ResetPixelScoreKernel(int* dev_pixel_score_);
