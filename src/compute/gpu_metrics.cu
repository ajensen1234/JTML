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
    dev_intersection_score_ =
        static_cast<int*>(bank0_metrics_.dev_intersection);
    dev_union_score_ = static_cast<int*>(bank0_metrics_.dev_union);
    dev_white_pix_count_ = static_cast<int*>(bank0_metrics_.dev_white_count);
    distance_map_score_ = static_cast<int*>(bank0_metrics_.host_distance_score);
    dev_distance_map_score_ =
        static_cast<int*>(bank0_metrics_.dev_distance_score);
    edge_pixels_count_ = static_cast<int*>(bank0_metrics_.host_edge_count);
    dev_edge_pixels_count_ = static_cast<int*>(bank0_metrics_.dev_edge_count);
    curvature_hausdorf_score_ =
        static_cast<int*>(bank0_metrics_.host_curvature);
    dev_curvature_hausdorf_score_ =
        static_cast<int*>(bank0_metrics_.dev_curvature);
    active_bank_ = nullptr;
    execution_stream_ = nullptr;
}

bool GPUMetrics::BindMetricBank(const BankState& bank) {
    const auto& m = bank.metrics;
    if (!m.host_pixel_score || !m.dev_pixel_score || !m.host_distance_score ||
        !m.dev_distance_score || !m.host_edge_count || !m.dev_edge_count ||
        !m.host_intersection || !m.host_union || !m.dev_intersection ||
        !m.dev_union || !m.host_white_count || !m.dev_white_count) {
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
    execution_stream_ =
        bank.stream ? reinterpret_cast<cudaStream_t>(bank.stream) : nullptr;
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

// ── U4: EvaluationContext metric overloads — graph-capturable paths ────
//
// These overloads are fully graph-capturable:
//   1. ComputeMetricCropKernel derives crop params on-device from
//      ctx.primary.dev_bounding_box (no host bbox read).
//   2. Graph-path kernel variants (EdgeKernel_Graph etc.) read crop
//      params from device MetricCropParams* and use fixed-max grids
//      sized from frame dims (max(2048, width) x max(2048, height)).
//   3. In-kernel guards retire threads beyond the actual crop.
//   4. Score D2H tail copies are cudaMemcpyAsync on ctx.stream.
//
// Between RenderPhase(EvaluationContext&) and these enqueues there is:
//   - NO host_bounding_box read, NO cudaMemcpy (sync),
//   - NO cudaStreamSynchronize, NO cudaEventSynchronize,
//   - NO dynamic host grid sizing.
//
// Completion happens later after one stream/event completion.
// Legacy BankState host-scalar overloads are untouched.

#include "compute/evaluation_context.h"
#include "compute/fast_implant_dilation_metric.cuh"
#include <algorithm>

namespace gpu_cost_function {

namespace {

// Fixed-max grid constants for graph-capturable metric launches.
// The grid covers max(2048, frameDims) in each dimension so that
// launch parameters are capturable in a CUDA graph.
constexpr int kMaxMetricDim = 2048;

} // anonymous namespace

cudaError_t GPUMetrics::EnqueueFastImplantDilationMetric(
    GPUImage* rendered_image,
    GPUDilatedFrame* cf,
    int dilation,
    EvaluationContext& ctx) {
    if (!ctx.initialized_correctly || !ctx.in_flight || ctx.stream == nullptr ||
        ctx.completion_event == nullptr) {
        return cudaErrorInvalidResourceHandle;
    }
    if (!cf) return cudaErrorInvalidValue;
    auto stream = reinterpret_cast<cudaStream_t>(ctx.stream);

    // Resolve frame dims from context (set once at pool init).
    const int height =
        ctx.height > 0
            ? ctx.height
            : (rendered_image ? rendered_image->GetFrameHeight() : 0);
    const int width =
        ctx.width > 0 ? ctx.width
                      : (rendered_image ? rendered_image->GetFrameWidth() : 0);
    unsigned char* image =
        ctx.primary.output != nullptr
            ? static_cast<unsigned char*>(ctx.primary.output)
            : (rendered_image ? rendered_image->GetDeviceImagePointer()
                              : nullptr);
    if (!image || width == 0 || height == 0) return cudaErrorInvalidValue;

    // Device pointers for crop derivation and metric score reduction.
    auto* dev_bb = static_cast<const int*>(ctx.primary.dev_bounding_box);
    auto* dev_crop =
        static_cast<MetricCropParams*>(ctx.primary.dev_metric_crop);
    if (!dev_bb || !dev_crop) return cudaErrorInvalidValue;

    // Bind ctx.metrics into GPUMetrics member aliases, saving bank-0.
    BankState temp_bank{};
    temp_bank.metrics = ctx.metrics;
    temp_bank.stream = ctx.stream;
    if (!TrySetActiveBank(&temp_bank)) return cudaErrorInvalidValue;
    // On return, dev_pixel_score_ / pixel_score_ etc. alias ctx.metrics.

    // 1. ComputeMetricCropKernel — single-thread, writes dev_crop on device.
    ComputeMetricCropKernel<<<1, 1, 0, stream>>>(
        dev_bb, dilation, width, height, dev_crop);

    // 2. Reset pixel score on device.
    FastImplantDilationMetric_ResetPixelScoreKernel<<<1, 1, 0, stream>>>(
        dev_pixel_score_);

    // 3. Fixed-max launch grids (graph-capturable).
    const int max_grid_w = std::max(kMaxMetricDim, width);
    const int max_grid_h = std::max(kMaxMetricDim, height);
    const unsigned long long max_crop_pixels =
        static_cast<unsigned long long>(max_grid_w) * max_grid_h;

    // EdgeKernel_Graph: 2D grid, 16x16 blocks (256 threads).
    constexpr unsigned kEdgeBlock = 16;
    dim3 edgeBlock(kEdgeBlock, kEdgeBlock);
    dim3 edgeGrid(
        static_cast<unsigned>((max_grid_w + kEdgeBlock - 1) / kEdgeBlock),
        static_cast<unsigned>((max_grid_h + kEdgeBlock - 1) / kEdgeBlock));

    FastImplantDilationMetric_EdgeKernel_Graph<<<
        edgeGrid,
        edgeBlock,
        0,
        stream>>>(image, dev_crop, width, height, dilation);

    // DilateKernel_Graph: 1D grid, 4 threads per crop pixel.
    const unsigned dilate_threads = static_cast<unsigned>(threads_per_block);
    const unsigned long long dilate_total = 4 * max_crop_pixels;
    const unsigned dilate_grid = static_cast<unsigned>(
        (dilate_total + dilate_threads - 1) / dilate_threads);

    FastImplantDilationMetric_DilateKernel_Graph<<<
        dilate_grid,
        dilate_threads,
        0,
        stream>>>(image, width, height, dev_crop, dilation);

    // DifferenceKernel_Graph: 1D grid, 1 thread per diff-crop pixel.
    const unsigned diff_grid = static_cast<unsigned>(
        (max_crop_pixels + dilate_threads - 1) / dilate_threads);

    FastImplantDilationMetric_DifferenceKernel_Graph<<<
        diff_grid,
        dilate_threads,
        0,
        stream>>>(
        image,
        static_cast<unsigned char*>(cf->GetDeviceImagePointer()),
        dev_pixel_score_,
        width,
        height,
        dev_crop);

    // 4. D2H tail copy — async on ctx.stream; completion checked later.
    cudaError_t err = cudaMemcpyAsync(
        pixel_score_,
        dev_pixel_score_,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream);

    // Restore bank-0 aliases.
    RestoreBank0Metrics();
    return err != cudaSuccess ? err : cudaGetLastError();
}

cudaError_t GPUMetrics::EnqueueDistanceMapMetric(
    GPUImage* projected_image,
    GPUFrame* dm,
    int dilation,
    EvaluationContext& ctx) {
    if (!ctx.initialized_correctly || !ctx.in_flight || ctx.stream == nullptr ||
        ctx.completion_event == nullptr) {
        return cudaErrorInvalidResourceHandle;
    }
    if (!dm) return cudaErrorInvalidValue;
    auto stream = reinterpret_cast<cudaStream_t>(ctx.stream);

    // Resolve frame dims from context (set once at pool init).
    const int height =
        ctx.height > 0
            ? ctx.height
            : (projected_image ? projected_image->GetFrameHeight() : 0);
    const int width =
        ctx.width > 0
            ? ctx.width
            : (projected_image ? projected_image->GetFrameWidth() : 0);
    unsigned char* image =
        ctx.primary.output != nullptr
            ? static_cast<unsigned char*>(ctx.primary.output)
            : (projected_image ? projected_image->GetDeviceImagePointer()
                               : nullptr);
    if (!image || width == 0 || height == 0) return cudaErrorInvalidValue;

    // Device pointers for crop derivation and metric score reduction.
    auto* dev_bb = static_cast<const int*>(ctx.primary.dev_bounding_box);
    auto* dev_crop =
        static_cast<MetricCropParams*>(ctx.primary.dev_metric_crop);
    if (!dev_bb || !dev_crop) return cudaErrorInvalidValue;

    // Bind ctx.metrics into GPUMetrics member aliases, saving bank-0.
    BankState temp_bank{};
    temp_bank.metrics = ctx.metrics;
    temp_bank.stream = ctx.stream;
    if (!TrySetActiveBank(&temp_bank)) return cudaErrorInvalidValue;
    // On return, dev_distance_map_score_ / distance_map_score_ etc. alias
    // ctx.metrics.

    // 1. ComputeMetricCropKernel — single-thread, writes dev_crop on device.
    ComputeMetricCropKernel<<<1, 1, 0, stream>>>(
        dev_bb, dilation, width, height, dev_crop);

    // 2. Reset distance-map score and edge-pixel count on device.
    DistanceMapMetric_ResetPixelScoreKernel<<<1, 1, 0, stream>>>(
        dev_distance_map_score_);
    DistanceMapMetric_ResetPixelScoreKernel<<<1, 1, 0, stream>>>(
        dev_edge_pixels_count_);

    // 3. Fixed-max launch grid (graph-capturable).
    const int max_grid_w = std::max(kMaxMetricDim, width);
    const int max_grid_h = std::max(kMaxMetricDim, height);
    const unsigned long long max_crop_pixels =
        static_cast<unsigned long long>(max_grid_w) * max_grid_h;
    const unsigned km_threads = static_cast<unsigned>(threads_per_block);
    const unsigned km_grid =
        static_cast<unsigned>((max_crop_pixels + km_threads - 1) / km_threads);

    DistanceMapMetric_Kernel_Graph<<<km_grid, km_threads, 0, stream>>>(
        image,
        static_cast<unsigned char*>(dm->GetDeviceImagePointer()),
        dev_distance_map_score_,
        dev_edge_pixels_count_,
        width,
        height,
        dev_crop);

    // 4. D2H tail copies — async on ctx.stream; completion checked later.
    cudaError_t err = cudaMemcpyAsync(
        distance_map_score_,
        dev_distance_map_score_,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream);
    if (err != cudaSuccess) {
        RestoreBank0Metrics();
        return err;
    }

    err = cudaMemcpyAsync(
        edge_pixels_count_,
        dev_edge_pixels_count_,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream);

    // Restore bank-0 aliases.
    RestoreBank0Metrics();
    return err != cudaSuccess ? err : cudaGetLastError();
}

} // namespace gpu_cost_function
