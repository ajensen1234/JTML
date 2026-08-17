/*GPU Metrics Header*/
#include "compute/gpu_metrics.cuh"

/*Cuda*/
#include "cuda.h"
#include "cuda_runtime.h"

/*Grayscale Colors*/
#include "compute/pixel_grayscale_colors.h"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"

#include "compute/fast_implant_dilation_metric.cuh"

__global__ void DistanceMapMetric_Kernel(
    unsigned char* projected_image,
    unsigned char* distance_map,
    int* distance_map_score,
    int* edge_pixel_count,
    int width,
    int height,
    int diff_kernel_left_x,
    int diff_kernel_bottom_y,
    int diff_kernel_cropped_width) {
    // Global Thread
    // Plan 008 U4 (the ONE live kernel bug, Finding 1): the grid->pixel index
    // formula was (blockIdx.y + gridDim.x + blockIdx.x) * blockDim.x, which
    // mis-maps every block row (crop rows 0..blockDim.x-1 never visited, the
    // middle band visited with multiplicity). Corrected to the standard
    // row-major block offset (matches iou.cu:37, l_1_1_matrix_diff_norm.cu:24,
    // fast_implant_dilation_metric.cu:129). The CPU reference in
    // test/unit/test_metric_semantics.cpp (CropIndexToGlobal) is the spec.
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    // Now, we need to convert the threadIdx (from
    // 0->Num_pixels_in_bounding_box) into something that we can use relative to
    // our entire image array. First, note that for any global thread (i),
    // i/bb_width (no remainder) gives the row, and i%width will give the
    // column. These row/column pairs can be converted into original pixel
    // coordinates, then further converted into row-major order (i.e. how we
    // access them in memory)

    int bb_row = i / diff_kernel_cropped_width;
    int bb_col = i % diff_kernel_cropped_width;
    int orig_row = bb_row + diff_kernel_bottom_y;
    int orig_col = bb_col + diff_kernel_left_x;

    int orig_loc = orig_col + orig_row * width;

    int pixel;
    if (orig_loc < width * height) {
        if (projected_image[orig_loc] == EDGE_PIXEL) {
            atomicAdd(&edge_pixel_count[0], 1);
            atomicAdd(&distance_map_score[0], distance_map[orig_loc]);
        }
    }
}

__global__ void DistanceMapMetric_ResetPixelScoreKernel(int* dev_pixel_score_) {
    dev_pixel_score_[0] = 0;
}

namespace gpu_cost_function {

double GPUMetrics::DistanceMapMetric(
    GPUImage* projected_image, GPUFrame* distance_map, int dilation) {
    /*
    This is the distance map metric with the hopes to "convexify" the
    search space a little bit more.
    Instead of just using the Hamming/Dilation overlap as a metric, which has
    many local min. We employ a distance map and calculate the average distance
    of each projected pixel to the template.
        * */

    int height = projected_image->GetFrameHeight();
    int width = projected_image->GetFrameWidth();
    int* bounding_box = projected_image->GetBoundingBox();

    DistanceMapMetric_ResetPixelScoreKernel<<<1, 1>>>(dev_distance_map_score_);
    DistanceMapMetric_ResetPixelScoreKernel<<<1, 1>>>(dev_edge_pixels_count_);

    int diff_kernel_left_x = max(bounding_box[0] - dilation, dilation);
    int diff_kernel_bottom_y = max(bounding_box[1] - dilation, dilation);
    int diff_kernel_right_x =
        min(bounding_box[2] + dilation, width - dilation - 1);
    int diff_kernel_top_y =
        min(bounding_box[3] + dilation, height - dilation - 1);
    int diff_kernel_cropped_width =
        diff_kernel_right_x - diff_kernel_left_x + 1;
    int diff_kernel_cropped_height =
        diff_kernel_top_y - diff_kernel_bottom_y + 1;

    // Construct some grid/block/thread dimensions
    // that fill out the whole of the bounding box
    // (maybe go a little bit more just to make sure
    // we grab everything)

    dim3 dim_grid_bounding_box = dim3(
        ceil(
            static_cast<double>(diff_kernel_cropped_width) /
            sqrt(static_cast<double>(threads_per_block))),
        ceil(
            static_cast<double>(diff_kernel_cropped_height) /
            sqrt(static_cast<double>(threads_per_block))));

    DistanceMapMetric_Kernel<<<dim_grid_bounding_box, threads_per_block>>>(
        projected_image->GetDeviceImagePointer(),
        distance_map->GetDeviceImagePointer(),
        dev_distance_map_score_,
        dev_edge_pixels_count_,
        width,
        height,
        diff_kernel_left_x,
        diff_kernel_bottom_y,
        diff_kernel_cropped_width);

    cudaMemcpy(
        distance_map_score_,
        dev_distance_map_score_,
        sizeof(int),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        edge_pixels_count_,
        dev_edge_pixels_count_,
        sizeof(int),
        cudaMemcpyDeviceToHost);

    // std::cout << "Distance Map Total Score : " << distance_map_score_[0] <<
    // std::endl; std::cout << "Number of Edge Pixels " << edge_pixels_count_[0]
    // << std::endl; std::cout << "Distance Metric Score :" << score <<
    // std::endl; std::cout << "==============================" << std::endl;
    return distance_map_score_[0] /
           (edge_pixels_count_[0] + 0.1); // adding a 0.1 to avoid singularities
}

cudaError_t GPUMetrics::EnqueueDistanceMapMetric(
    GPUImage* projected_image, GPUFrame* distance_map, int dilation,
    cudaStream_t stream) {
    if (!stream || !active_bank_) {
        return cudaErrorInvalidResourceHandle;
    }
    const int height = active_bank_->height > 0 ? active_bank_->height : projected_image->GetFrameHeight();
    const int width = active_bank_->width > 0 ? active_bank_->width : projected_image->GetFrameWidth();
    int* bounding_box = active_bank_->primary.host_bounding_box != nullptr
                            ? static_cast<int*>(active_bank_->primary.host_bounding_box)
                            : projected_image->GetBoundingBox();
    unsigned char* image = active_bank_->primary.output != nullptr
                               ? static_cast<unsigned char*>(active_bank_->primary.output)
                               : projected_image->GetDeviceImagePointer();
    DistanceMapMetric_ResetPixelScoreKernel<<<1, 1, 0, stream>>>(dev_distance_map_score_);
    DistanceMapMetric_ResetPixelScoreKernel<<<1, 1, 0, stream>>>(dev_edge_pixels_count_);
    int left = max(bounding_box[0] - dilation, dilation);
    int bottom = max(bounding_box[1] - dilation, dilation);
    int right = min(bounding_box[2] + dilation, width - dilation - 1);
    int top = min(bounding_box[3] + dilation, height - dilation - 1);
    int crop_width = right - left + 1;
    int crop_height = top - bottom + 1;
    dim3 grid(static_cast<unsigned>(ceil(static_cast<double>(crop_width) /
                                         sqrt(static_cast<double>(threads_per_block)))),
              static_cast<unsigned>(ceil(static_cast<double>(crop_height) /
                                         sqrt(static_cast<double>(threads_per_block)))));
    DistanceMapMetric_Kernel<<<grid, threads_per_block, 0, stream>>>(
        image, distance_map->GetDeviceImagePointer(),
        dev_distance_map_score_, dev_edge_pixels_count_, width, height,
        left, bottom, crop_width);
    cudaError_t err = cudaMemcpyAsync(distance_map_score_, dev_distance_map_score_,
                                      sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(edge_pixels_count_, dev_edge_pixels_count_, sizeof(int),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    return cudaGetLastError();
}


double GPUMetrics::CompleteDistanceMapMetric(cudaStream_t stream) {
    if (!stream || !active_bank_) return 0.0;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 0.0;
    return distance_map_score_[0] / (edge_pixels_count_[0] + 0.1);
}

double GPUMetrics::DistanceMapMetric(
    GPUImage* projected_image,
    GPUFrame* distance_map,
    int dilation,
    cudaStream_t stream) {
    if (EnqueueDistanceMapMetric(
            projected_image, distance_map, dilation, stream) != cudaSuccess) {
        return 0.0;
    }
    return CompleteDistanceMapMetric(stream);
}
} /*end namespace gpu_cost_function*/
