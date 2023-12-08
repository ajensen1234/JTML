#include "gpu/gpu_metrics.cuh"

#include <cuda_runtime.h>
#include <cuda.h>

/*Grayscale Colors*/
#include "gpu/pixel_grayscale_colors.h"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"


__global__ void CurvatureHausdorfMetric_Kernel(
    unsigned char* projected_image,
    unsigned char* heatmaps,
    int* hausdorf_score,
    int width, int height,
    int left_x, int bottom_y,
    int diff_cropped_width
){
    // Have to do some fun math here to get the cropped
    // with to match up with the original image locations
    int thread_x = threadIdx.x + (blockIdx.x * blockDim.x);
    int thread_y = threadIdx.y + (blockIdx.y * blockDim.y);

    int orig_x = thread_x + left_x;
    int orig_y = thread_y + bottom_y;

    int orig_idx = orig_x + orig_y*width;

    // Grabbing keypoint locations by adding full image pixels to move
    // to the next keypoint value
    int kp_loc = orig_idx + blockDim.z * height*width;


};
__global__ void Reset_CurvatureHausdorfScore_Kernel(int* dev_curv_haus_score){
    int i = threadIdx.x;
    dev_curv_haus_score[i] = 0;
}

namespace gpu_cost_function{
    double GPUMetrics::CurvatureHeatmapMetric(GPUImage* projected_image, GPUHeatmap* gpu_heatmap){
        int height = projected_image->GetFrameHeight();
        int width = projected_image->GetFrameWidth();
        int num_kp = gpu_heatmap->GetNumKeypoints();
        int* bounding_box = projected_image->GetBoundingBox();

        Reset_CurvatureHausdorfScore_Kernel<<<1,num_kp>>>(dev_curvature_hausdorf_score_);

        int left_x = max(bounding_box[0],0);
        int bottom_y = max(bounding_box[1],0);
        int right_x = min(bounding_box[2], width-1);
        int top_y = min(bounding_box[3], height-1);
        int diff_cropped_width = right_x - left_x - 1;
        int diff_cropped_height = top_y - bottom_y + 1;

        dim3 dim_grid_bounding_box = dim3(
            ceil(static_cast<double>(diff_cropped_width) / sqrt(static_cast<double>(threads_per_block))),
            ceil(static_cast<double>(diff_cropped_height) / sqrt(static_cast<double>(threads_per_block))),
            num_kp
);
        dim3 dim_block = dim3(
            ceil(sqrt(static_cast<double>(threads_per_block))),
            ceil(sqrt(static_cast<double>(threads_per_block)))
);


        double test = 0;
        return test;
    };
};
