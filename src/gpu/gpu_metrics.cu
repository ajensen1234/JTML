/*GPU Metrics Header*/
#include "gpu/gpu_metrics.cuh"

/*CUDA Error Checking*/
#include "gpu/cuda_check.cuh"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"

/*Grayscale Colors*/
#include "pixel_grayscale_colors.h"

namespace gpu_cost_function {

/*Constructor and Destructor for GPU Metrics Class*/
GPUMetrics::GPUMetrics() {

    /*Initialized Correctly?*/
    initialized_correctly_ = true;

    /*Initialize Pinned Memory for Slightly Faster Transfer if Using Mismatched
     * Pixel Count*/
    CUDA_CHECK(cudaHostAlloc((void**)&pixel_score_, sizeof(int), cudaHostAllocDefault));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Initialize Pinned Memory for IOU Intermediary */
    CUDA_CHECK(cudaHostAlloc(
        (void**)&intersection_score_, sizeof(int), cudaHostAllocDefault));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Initialize Pinned Memory for IOU Intermediary */
    CUDA_CHECK(cudaHostAlloc((void**)&union_score_, sizeof(int), cudaHostAllocDefault));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Allocate GPU buffers for pixel score.*/
    CUDA_CHECK(cudaMalloc((void**)&dev_pixel_score_, sizeof(int)));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Allocate GPU buffers for IOU Intermediary*/
    CUDA_CHECK(cudaMalloc((void**)&dev_intersection_score_, sizeof(int)));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Allocate GPU buffers for IOU Intermediary*/
    CUDA_CHECK(cudaMalloc((void**)&dev_union_score_, sizeof(int)));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Allocate GPU buffers for comparison white pixel count.*/
    CUDA_CHECK(cudaMalloc((void**)&dev_white_pix_count_, sizeof(int)));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Upload (Reset) white pixel count for comparison image from Host to
     * Device.*/
    white_pix_count_ = 0;
    CUDA_CHECK(cudaMemcpy(
        dev_white_pix_count_,
        &white_pix_count_,
        sizeof(int),
        cudaMemcpyHostToDevice));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    CUDA_CHECK(cudaHostAlloc(
        (void**)&distance_map_score_, sizeof(int), cudaHostAllocDefault));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;
    // Allocate some memory for the dev dm score
    CUDA_CHECK(cudaMalloc((void**)&dev_distance_map_score_, sizeof(int)));
    if (cudaGetLastError() != cudaSuccess) {
        initialized_correctly_ = false;
    }

    // Allocating memory for the edge pixels count (GPU and CPU)
    CUDA_CHECK(cudaMalloc((void**)&dev_edge_pixels_count_, sizeof(int)));
    if (cudaGetLastError() != cudaSuccess) {
        initialized_correctly_ = false;
    }
    CUDA_CHECK(cudaHostAlloc(
        (void**)&edge_pixels_count_, sizeof(int), cudaHostAllocDefault));
    if (cudaGetLastError() != cudaSuccess) {
        initialized_correctly_ = false;
    }
};

GPUMetrics::~GPUMetrics() {
    /*Free CUDA*/
    CUDA_CHECK(cudaFree(dev_pixel_score_));
    CUDA_CHECK(cudaFree(dev_intersection_score_));
    CUDA_CHECK(cudaFree(dev_union_score_));
    CUDA_CHECK(cudaFree(dev_edge_pixels_count_));
    CUDA_CHECK(cudaFree(dev_distance_map_score_));
    CUDA_CHECK(cudaFree(dev_curvature_hausdorf_score_));

    /*Free Host*/
    CUDA_CHECK(cudaFreeHost(pixel_score_));
    CUDA_CHECK(cudaFreeHost(intersection_score_));
    CUDA_CHECK(cudaFreeHost(union_score_));
    CUDA_CHECK(cudaFreeHost(distance_map_score_));
    CUDA_CHECK(cudaFreeHost(edge_pixels_count_));
    CUDA_CHECK(cudaFreeHost(curvature_hausdorf_score_));
};

void GPUMetrics::AllocateCurvatureHausdorfScore(int num_keypoints) {
    CUDA_CHECK(cudaMalloc(
        (void**)&dev_curvature_hausdorf_score_, num_keypoints * sizeof(int)));
    if (cudaGetLastError() != cudaSuccess) {
        initialized_correctly_ = false;
    }

    CUDA_CHECK(cudaHostAlloc(
        (void**)&curvature_hausdorf_score_,
        num_keypoints * sizeof(int),
        cudaHostAllocDefault));
    if (cudaGetLastError() != cudaSuccess) {
        initialized_correctly_ = false;
    }
};

/*Reset White Pixel Count*/
__global__ void
ComputeSumWhitePixels__ResetWhitePixelScoreKernel(int* dev_white_pix_count_) {
    dev_white_pix_count_[0] = 0;
}

/*CUDA White Pixel Sum Function*/
__global__ void WhitePixelSum(
    unsigned char* dev_dilation_comparison_image,
    int* dev_comparison_white_pix_count,
    int width,
    int height) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < width * height) {
        if (dev_dilation_comparison_image[i] == WHITE_PIXEL)
            atomicAdd(&dev_comparison_white_pix_count[0], 1);
    }
};

/*Computes Sum of White Pixels in Image*/
int GPUMetrics::ComputeSumWhitePixels(GPUImage* image, cudaError* error) {

    /*Reset Errors*/
    cudaGetLastError();

    /*Reset White Pixel Count*/
    CUDA_CHECK_KERNEL(ComputeSumWhitePixels__ResetWhitePixelScoreKernel<<<1, 1>>>(
        dev_white_pix_count_));

    /*Get Sum of White Pixels in Dilation Comparison Image and Total Pixel Sum*/
    auto dim_grid_comparison_white_pix = dim3(
        ceil(sqrt(
            static_cast<double>(
                image->GetFrameWidth() * image->GetFrameHeight()) /
            static_cast<double>(256))),
        ceil(sqrt(
            static_cast<double>(
                image->GetFrameWidth() * image->GetFrameHeight()) /
            static_cast<double>(256))));
    CUDA_CHECK_KERNEL(WhitePixelSum<<<dim_grid_comparison_white_pix, 256>>>(
        image->GetDeviceImagePointer(),
        dev_white_pix_count_,
        image->GetFrameWidth(),
        image->GetFrameHeight()));
    CUDA_CHECK(cudaMemcpy(
        &white_pix_count_,
        dev_white_pix_count_,
        sizeof(int),
        cudaMemcpyDeviceToHost));
    /*Get Errors*/
    *error = cudaGetLastError();
    return white_pix_count_;
};

/*Get Initialized Correctly*/
bool GPUMetrics::IsInitializedCorrectly() {
    return initialized_correctly_;
}
} // namespace gpu_cost_function
