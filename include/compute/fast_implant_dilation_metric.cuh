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

// U4: device-side metric crop derivation from device AABB
__global__ void ComputeMetricCropKernel(
    const int* dev_bounding_box,
    int dilation,
    int width,
    int height,
    gpu_cost_function::MetricCropParams* dev_crop);

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

// U4: graph-capturable kernel variants — read crop from device
// MetricCropParams*, fixed-max grid, in-kernel out-of-crop guards.  Legacy
// host-scalar kernels above are unchanged.

__global__ void FastImplantDilationMetric_EdgeKernel_Graph(
    unsigned char* dev_image,
    const gpu_cost_function::MetricCropParams* crop,
    int width,
    int height,
    int dilation);

__global__ void FastImplantDilationMetric_DilateKernel_Graph(
    unsigned char* dev_image,
    int width,
    int height,
    const gpu_cost_function::MetricCropParams* crop,
    int dilation);

__global__ void FastImplantDilationMetric_DifferenceKernel_Graph(
    unsigned char* dev_image,
    unsigned char* dev_comparison_image,
    int* result,
    int width,
    int height,
    const gpu_cost_function::MetricCropParams* crop);

__global__ void DistanceMapMetric_Kernel_Graph(
    unsigned char* projected_image,
    unsigned char* distance_map,
    int* distance_map_score,
    int* edge_pixel_count,
    int width,
    int height,
    const gpu_cost_function::MetricCropParams* crop);
