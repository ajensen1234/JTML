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
    pixel_score_ = make_cuda_host_unique<int>(1);
    if (!pixel_score_) initialized_correctly_ = false;

    /*Initialize Pinned Memory for IOU Intermediary */
    intersection_score_ = make_cuda_host_unique<int>(1);
    if (!intersection_score_) initialized_correctly_ = false;

    /*Initialize Pinned Memory for IOU Intermediary */
    union_score_ = make_cuda_host_unique<int>(1);
    if (!union_score_) initialized_correctly_ = false;

    /*Allocate GPU buffers for pixel score.*/
    dev_pixel_score_ = make_cuda_unique<int>(1);
    if (!dev_pixel_score_) initialized_correctly_ = false;

    /*Allocate GPU buffers for IOU Intermediary*/
    dev_intersection_score_ = make_cuda_unique<int>(1);
    if (!dev_intersection_score_) initialized_correctly_ = false;

    /*Allocate GPU buffers for IOU Intermediary*/
    dev_union_score_ = make_cuda_unique<int>(1);
    if (!dev_union_score_) initialized_correctly_ = false;

    /*Allocate GPU buffers for comparison white pixel count.*/
    dev_white_pix_count_ = make_cuda_unique<int>(1);
    if (!dev_white_pix_count_) initialized_correctly_ = false;

    /*Upload (Reset) white pixel count for comparison image from Host to
     * Device.*/
    white_pix_count_ = 0;
    CUDA_CHECK(cudaMemcpy(
        dev_white_pix_count_.get(),
        &white_pix_count_,
        sizeof(int),
        cudaMemcpyHostToDevice));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    distance_map_score_ = make_cuda_host_unique<int>(1);
    if (!distance_map_score_) initialized_correctly_ = false;

    // Allocate some memory for the dev dm score
    dev_distance_map_score_ = make_cuda_unique<int>(1);
    if (!dev_distance_map_score_) initialized_correctly_ = false;

    // Allocating memory for the edge pixels count (GPU and CPU)
    dev_edge_pixels_count_ = make_cuda_unique<int>(1);
    if (!dev_edge_pixels_count_) initialized_correctly_ = false;

    edge_pixels_count_ = make_cuda_host_unique<int>(1);
    if (!edge_pixels_count_) initialized_correctly_ = false;
};

GPUMetrics::~GPUMetrics() = default;

void GPUMetrics::AllocateCurvatureHausdorfScore(int num_keypoints) {
    dev_curvature_hausdorf_score_ = make_cuda_unique<int>(num_keypoints);
    if (!dev_curvature_hausdorf_score_) {
        initialized_correctly_ = false;
    }

    curvature_hausdorf_score_ = make_cuda_host_unique<int>(num_keypoints);
    if (!curvature_hausdorf_score_) {
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
        dev_white_pix_count_.get()));

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
        dev_white_pix_count_.get(),
        image->GetFrameWidth(),
        image->GetFrameHeight()));
    CUDA_CHECK(cudaMemcpy(
        &white_pix_count_,
        dev_white_pix_count_.get(),
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
