/*GPU Metrics Header*/
#include "gpu/gpu_metrics.cuh"

/*Cuda*/
#include "cuda_runtime.h"
#include "cuda.h"

/*Grayscale Colors*/
#include "gpu/pixel_grayscale_colors.h"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"

#include "gpu/fast_implant_dilation_metric.cuh"


__global__ void DistanceMapMetric_Kernel(
    unsigned char* projected_image,
    unsigned char* distance_map,
    int* distance_map_score,
    int* edge_pixel_count,
    int width, int height,
    int diff_kernel_left_x,
    int diff_kernel_bottom_y,
    int diff_kernel_cropped_width)
{
    // Global Thread
    int i = (blockIdx.y + gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    // Convert thread to subsize within the full image
    // i = (i/diff_kernel_cropped_width) * width + (i%diff_kernel_cropped_width)
    //     + (diff_kernel_bottom_y * width) + diff_kernel_left_x;
    int pixel;
    int distance_map_value;
    if (i<width*height){
        atomicAdd(&edge_pixel_count[0],distance_map[i]);
        pixel = projected_image[i];
        distance_map_value = distance_map[i];
        if (pixel == EDGE_PIXEL || pixel == WHITE_PIXEL){
            atomicAdd(&distance_map_score[0], 1);
            projected_image[i] = WHITE_PIXEL;

        }
    }
}

__global__ void DistanceMapMetric_ResetPixelScoreKernel(int* dev_pixel_score_){
    dev_pixel_score_[0] = 0;
}

__global__ void DistanceMapMask_Kernel(

    unsigned char* projected_image,
    unsigned char* distance_map,
    unsigned char* map_mask,
    int width, int height
){
    // Create the global thread
    int i = (blockIdx.y + gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    int dm_val;
    int proj_val;

    if (i < width*height){
        dm_val = distance_map[i];
        proj_val = projected_image[i];
        if (proj_val == WHITE_PIXEL || proj_val == EDGE_PIXEL){
            map_mask[i] = dm_val;
        } else{
            map_mask[i] = 0;
        }
    }
}


namespace gpu_cost_function{

    double GPUMetrics::DistanceMapMetric(
        GPUImage* projected_image,
        GPUFrame* distance_map,
        int dilation
){
        /*
        This is the distance map metric with the hopes to "convexify" the
        search space a little bit more.
        Instead of just using the Hamming/Dilation overlap as a metric, which has many local min.
        We employ a distance map and calculate the average distance of each projected pixel
        to the template.
            * */

        int height = projected_image->GetFrameHeight();
        int width = projected_image->GetFrameWidth();
        int* bounding_box = projected_image->GetBoundingBox();
        GPUImage* dist_mask = new GPUImage(1024,1024,0);

        /*Using previously written kernel to reset the metric*/
        DistanceMapMetric_ResetPixelScoreKernel<<<1,1>>>(dev_distance_map_score_);
        DistanceMapMetric_ResetPixelScoreKernel<<<1,1>>>(dev_edge_pixels_count_);

		int diff_kernel_left_x = max(bounding_box[0] - dilation, 0);
		int diff_kernel_bottom_y = max(bounding_box[1] - dilation, 0);
		int diff_kernel_right_x = min(bounding_box[2] + dilation, width - 1);
		int diff_kernel_top_y = min(bounding_box[3] + dilation, height - 1);
		int diff_kernel_cropped_width = diff_kernel_right_x - diff_kernel_left_x + 1;
		int diff_kernel_cropped_height = diff_kernel_top_y - diff_kernel_bottom_y + 1;

		dim_grid_image_processing_ = dim3(
			ceil(static_cast<double>(diff_kernel_cropped_width) / sqrt(static_cast<double>(threads_per_block))),
			ceil(static_cast<double>(diff_kernel_cropped_height) / sqrt(static_cast<double>(threads_per_block))));

        // dim_grid_image_processing_ = dim3(
        //     ceil(sqrt(static_cast<double>(width*height) / static_cast<double>(threads_per_block))),
        //     ceil(sqrt(static_cast<double>(height*height) / static_cast<double>(threads_per_block))));
        DistanceMapMetric_Kernel <<< dim_grid_image_processing_, threads_per_block>>>(
        projected_image->GetDeviceImagePointer(),
            distance_map->GetDeviceImagePointer(),
            dev_distance_map_score_,
            dev_edge_pixels_count_,
            width, height,
            diff_kernel_left_x,
            diff_kernel_bottom_y,
            diff_kernel_cropped_width);
        float bb_area = abs(bounding_box[0] - bounding_box[2]) * abs(bounding_box[1] - bounding_box[3]);
        cudaMemcpy(distance_map_score_, dev_distance_map_score_, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(edge_pixels_count_, dev_edge_pixels_count_, sizeof(int), cudaMemcpyDeviceToHost);
        projected_image->WriteImage("this_is_an_image_from_inside_cuda.png");

        std::cout << "Pixel score :" << distance_map_score_[0] << std::endl;
        std::cout << "Normed dm total pixels : " << edge_pixels_count_[0]/bb_area << std::endl;
        //std::cout << "Bounding Box" <<  bounding_box[0] << std::endl;
    }
} /*end namespace gpu_cost_function*/
