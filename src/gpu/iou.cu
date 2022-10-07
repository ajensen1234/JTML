/*GPU Metrics Header*/
#include "gpu/gpu_metrics.cuh"

/*Cuda*/
#include "cuda_runtime.h"
#include "cuda.h"

/*Grayscale Colors*/
#include "pixel_grayscale_colors.h"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"

/*Kernels*/
__global__ void IOU__ResetIOUScoresKernel(int* dev_intersection_score, int* dev_union_score) {
	dev_intersection_score[0] = 0;
	dev_union_score[0] = 0;
}

__global__ void IOUKernel(unsigned char* dev_A, unsigned char* dev_B, int* dev_intersection_score, int* dev_union_score,
                          int width, int height,
                          int diff_kernel_left_x, int diff_kernel_bottom_y, int diff_kernel_cropped_width) {
	/*Global Thread*/
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	/*Convert to Subsize*/
	i = (i / diff_kernel_cropped_width) * width + (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width +
		diff_kernel_left_x;


	/*If Correct Width and Height*/
	if (i < width * height) {
		int A_element = dev_A[i];
		int B_element = dev_B[i];

		if (A_element > 0 || B_element > 0) {
			atomicAdd(&dev_union_score[0], 1);
			if (A_element > 0 && B_element > 0) {
				atomicAdd(&dev_intersection_score[0], 1);
			}
		}
	}

}

namespace gpu_cost_function {
	double GPUMetrics::IOU(GPUImage* image_A, GPUImage* image_B) {

		/*Extract Bounding Boxes*/
		int* bounding_box_A = image_A->GetBoundingBox();
		int* bounding_box_B = image_B->GetBoundingBox();

		/*ASSUMES IMAGES ARE SAME DIMENSION!!!!*/
		int height = image_A->GetFrameHeight();
		int width = image_A->GetFrameWidth();

		/*Reset the IOU Score*/
		IOU__ResetIOUScoresKernel << <1, 1 >> >(dev_intersection_score_, dev_union_score_);

		/* Compute launch parameters for difference. Want same size as sub image nut with no dilation padding at edges. */
		int diff_kernel_left_x = max(min(bounding_box_A[0], bounding_box_B[0]), 0);
		int diff_kernel_bottom_y = max(min(bounding_box_A[1], bounding_box_B[1]), 0);
		int diff_kernel_right_x = min(max(bounding_box_A[2], bounding_box_B[2]), width - 1);
		int diff_kernel_top_y = min(max(bounding_box_A[3], bounding_box_B[3]), height - 1);
		int diff_kernel_cropped_width = diff_kernel_right_x - diff_kernel_left_x + 1;
		int diff_kernel_cropped_height = diff_kernel_top_y - diff_kernel_bottom_y + 1;

		dim_grid_image_processing_ = dim3::dim3(
			ceil(static_cast<double>(diff_kernel_cropped_width) / sqrt(static_cast<double>(threads_per_block))),
			ceil(static_cast<double>(diff_kernel_cropped_height) / sqrt(static_cast<double>(threads_per_block))));

		/*IOU Kernel*/
		IOUKernel << <dim_grid_image_processing_, threads_per_block >> >(
			image_A->GetDeviceImagePointer(), image_B->GetDeviceImagePointer(), dev_intersection_score_,
			dev_union_score_,
			width, height, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);

		/*Return IOU Score*/
		cudaMemcpy(intersection_score_, dev_intersection_score_, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(union_score_, dev_union_score_, sizeof(int), cudaMemcpyDeviceToHost);
		return static_cast<double>(intersection_score_[0]) / static_cast<double>(union_score_[0]);
	}
}
