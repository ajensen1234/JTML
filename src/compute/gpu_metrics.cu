/*GPU Metrics Header*/
#include "compute/gpu_metrics.cuh"

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
    cudaHostAlloc((void**)&pixel_score_, sizeof(int), cudaHostAllocDefault);
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Initialize Pinned Memory for IOU Intermediary */
    cudaHostAlloc(
        (void**)&intersection_score_, sizeof(int), cudaHostAllocDefault);
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Initialize Pinned Memory for IOU Intermediary */
    cudaHostAlloc((void**)&union_score_, sizeof(int), cudaHostAllocDefault);
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Allocate GPU buffers for pixel score.*/
    cudaMalloc((void**)&dev_pixel_score_, sizeof(int));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Allocate GPU buffers for IOU Intermediary*/
    cudaMalloc((void**)&dev_intersection_score_, sizeof(int));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Allocate GPU buffers for IOU Intermediary*/
    cudaMalloc((void**)&dev_union_score_, sizeof(int));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Allocate GPU buffers for comparison white pixel count.*/
    cudaMalloc((void**)&dev_white_pix_count_, sizeof(int));
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    /*Upload (Reset) white pixel count for comparison image from Host to
     * Device.*/
    white_pix_count_ = 0;
    cudaMemcpy(
        dev_white_pix_count_,
        &white_pix_count_,
        sizeof(int),
        cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;

    cudaHostAlloc(
        (void**)&distance_map_score_, sizeof(int), cudaHostAllocDefault);
    if (cudaGetLastError() != cudaSuccess) initialized_correctly_ = false;
    // Allocate some memory for the dev dm score
    cudaMalloc((void**)&dev_distance_map_score_, sizeof(int));
    if (cudaGetLastError() != cudaSuccess) {
        initialized_correctly_ = false;
    }

    // Allocating memory for the edge pixels count (GPU and CPU)
    cudaMalloc((void**)&dev_edge_pixels_count_, sizeof(int));
    if (cudaGetLastError() != cudaSuccess) {
        initialized_correctly_ = false;
    }
    cudaHostAlloc(
        (void**)&edge_pixels_count_, sizeof(int), cudaHostAllocDefault);
    if (cudaGetLastError() != cudaSuccess) {
        initialized_correctly_ = false;
    }
    CaptureBank0Metrics();
};

GPUMetrics::~GPUMetrics() {
    /* External bank pointers are non-owning. Always restore the original
     * allocation set before freeing members. */
    RestoreBank0Metrics();
    /*Free CUDA*/
    cudaFree(dev_pixel_score_);
    cudaFree(dev_intersection_score_);
    cudaFree(dev_union_score_);
    cudaFree(dev_edge_pixels_count_);
    cudaFree(dev_distance_map_score_);
    cudaFree(dev_curvature_hausdorf_score_);

    /*Free Host*/
    cudaFreeHost(pixel_score_);
    cudaFreeHost(intersection_score_);
    cudaFreeHost(union_score_);
    cudaFreeHost(distance_map_score_);
    cudaFreeHost(edge_pixels_count_);
    cudaFreeHost(curvature_hausdorf_score_);
};

void GPUMetrics::AllocateCurvatureHausdorfScore(int num_keypoints) {
    /*Owner fix (2026-08-12): 0-keypoint heatmaps (a study loaded without
     * ML segmentation) must not allocate — a 0-size cudaHostAlloc is not
     * guaranteed to succeed on every driver. The curvature metrics are
     * no-ops with 0 keypoints.*/
    if (num_keypoints <= 0) {
        return;
    }

    cudaMalloc(
        (void**)&dev_curvature_hausdorf_score_, num_keypoints * sizeof(int));
    if (cudaGetLastError() != cudaSuccess) {
        initialized_correctly_ = false;
    }

    cudaHostAlloc(
        (void**)&curvature_hausdorf_score_,
        num_keypoints * sizeof(int),
        cudaHostAllocDefault);
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
    ComputeSumWhitePixels__ResetWhitePixelScoreKernel<<<1, 1>>>(
        dev_white_pix_count_);

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
    WhitePixelSum<<<dim_grid_comparison_white_pix, 256>>>(
        image->GetDeviceImagePointer(),
        dev_white_pix_count_,
        image->GetFrameWidth(),
        image->GetFrameHeight());
    cudaMemcpy(
        &white_pix_count_,
        dev_white_pix_count_,
        sizeof(int),
        cudaMemcpyDeviceToHost);
    /*Get Errors*/
    *error = cudaGetLastError();
    return white_pix_count_;
};

/*Get Initialized Correctly*/
bool GPUMetrics::IsInitializedCorrectly() {
    return initialized_correctly_;
}

void GPUMetrics::CaptureBank0Metrics() {
    bank0_metrics_.host_pixel_score = pixel_score_;
    bank0_metrics_.dev_pixel_score = dev_pixel_score_;
    bank0_metrics_.host_intersection = intersection_score_;
    bank0_metrics_.host_union = union_score_;
    bank0_metrics_.dev_intersection = dev_intersection_score_;
    bank0_metrics_.dev_union = dev_union_score_;
    bank0_metrics_.host_white_count = &white_pix_count_;
    bank0_metrics_.dev_white_count = dev_white_pix_count_;
    bank0_metrics_.host_distance_score = distance_map_score_;
    bank0_metrics_.dev_distance_score = dev_distance_map_score_;
    bank0_metrics_.host_edge_count = edge_pixels_count_;
    bank0_metrics_.dev_edge_count = dev_edge_pixels_count_;
    bank0_metrics_.host_curvature = curvature_hausdorf_score_;
    bank0_metrics_.dev_curvature = dev_curvature_hausdorf_score_;
    bank0_metrics_.curvature_capacity = 0;
    bank0_metrics_captured_ = true;
}

void GPUMetrics::RestoreBank0Metrics() {
    if (!bank0_metrics_captured_) return;
    pixel_score_ = static_cast<int*>(bank0_metrics_.host_pixel_score);
    dev_pixel_score_ = static_cast<int*>(bank0_metrics_.dev_pixel_score);
    intersection_score_ = static_cast<int*>(bank0_metrics_.host_intersection);
    union_score_ = static_cast<int*>(bank0_metrics_.host_union);
    dev_intersection_score_ = static_cast<int*>(bank0_metrics_.dev_intersection);
    dev_union_score_ = static_cast<int*>(bank0_metrics_.dev_union);
    dev_white_pix_count_ = static_cast<int*>(bank0_metrics_.dev_white_count);
    distance_map_score_ = static_cast<int*>(bank0_metrics_.host_distance_score);
    dev_distance_map_score_ = static_cast<int*>(bank0_metrics_.dev_distance_score);
    edge_pixels_count_ = static_cast<int*>(bank0_metrics_.host_edge_count);
    dev_edge_pixels_count_ = static_cast<int*>(bank0_metrics_.dev_edge_count);
    curvature_hausdorf_score_ = static_cast<int*>(bank0_metrics_.host_curvature);
    dev_curvature_hausdorf_score_ = static_cast<int*>(bank0_metrics_.dev_curvature);
    active_bank_ = nullptr;
    execution_stream_ = nullptr;
}

bool GPUMetrics::BindMetricBank(const BankState& bank) {
    const auto& m = bank.metrics;
    if (!m.host_pixel_score || !m.dev_pixel_score ||
        !m.host_distance_score || !m.dev_distance_score ||
        !m.host_edge_count || !m.dev_edge_count ||
        !m.host_intersection || !m.host_union ||
        !m.dev_intersection || !m.dev_union ||
        !m.host_white_count || !m.dev_white_count) {
        return false;
    }
    pixel_score_ = static_cast<int*>(m.host_pixel_score);
    dev_pixel_score_ = static_cast<int*>(m.dev_pixel_score);
    intersection_score_ = static_cast<int*>(m.host_intersection);
    union_score_ = static_cast<int*>(m.host_union);
    dev_intersection_score_ = static_cast<int*>(m.dev_intersection);
    dev_union_score_ = static_cast<int*>(m.dev_union);
    dev_white_pix_count_ = static_cast<int*>(m.dev_white_count);
    distance_map_score_ = static_cast<int*>(m.host_distance_score);
    dev_distance_map_score_ = static_cast<int*>(m.dev_distance_score);
    edge_pixels_count_ = static_cast<int*>(m.host_edge_count);
    dev_edge_pixels_count_ = static_cast<int*>(m.dev_edge_count);
    curvature_hausdorf_score_ = static_cast<int*>(m.host_curvature);
    dev_curvature_hausdorf_score_ = static_cast<int*>(m.dev_curvature);
    active_bank_ = const_cast<BankState*>(&bank);
    execution_stream_ = bank.stream ? reinterpret_cast<cudaStream_t>(bank.stream) : nullptr;
    return true;
}

bool GPUMetrics::TrySetActiveBank(BankState* bank) {
    if (bank == nullptr) {
        RestoreBank0Metrics();
        return true;
    }
    if (!bank0_metrics_captured_) CaptureBank0Metrics();
    if (!BindMetricBank(*bank)) {
        RestoreBank0Metrics();
        return false;
    }
    return true;
}

void GPUMetrics::SetActiveBank(BankState* bank) {
    (void)TrySetActiveBank(bank);
}

void GPUMetrics::SetExecutionStream(cudaStream_t stream) {
    execution_stream_ = stream;
}

BankState* GPUMetrics::GetActiveBank() const {
    return active_bank_;
}

cudaStream_t GPUMetrics::GetExecutionStream() const {
    return execution_stream_;
}

} // namespace gpu_cost_function

