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
__global__ void L_1_1_MatrixDifferenceNorm__ResetPixelScoreKernel(int* dev_pixel_score) {
	dev_pixel_score[0] = 0;
}

__global__ void L_1_1_MatrixDifferenceNorm_DifferenceKernel(unsigned char* dev_A, unsigned char* dev_B, int* result,
                                                            int width, int height,
                                                            int diff_kernel_left_x, int diff_kernel_bottom_y,
                                                            int diff_kernel_cropped_width) {
	/*Global Thread*/
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	/*Convert to Subsize*/
	i = (i / diff_kernel_cropped_width) * width + (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width +
		diff_kernel_left_x;


	/*If Correct Width and Height*/
	if (i < width * height) {
		int diff_image = dev_A[i] - dev_B[i];
		if (diff_image >= 0)
			atomicAdd(&result[0], diff_image);
		else
			atomicSub(&result[0], diff_image);
	}

}

namespace gpu_cost_function {
	double GPUMetrics::L_1_1_MatrixDifferenceNorm(GPUImage* image_A, GPUImage* image_B) {

		/*Extract Bounding Boxes*/
		int* bounding_box_A = image_A->GetBoundingBox();
		int* bounding_box_B = image_B->GetBoundingBox();

		/*ASSUMES IMAGES ARE SAME DIMENSION!!!!*/
		int height = image_A->GetFrameHeight();
		int width = image_A->GetFrameWidth();

		/*Reset the Pixel Score*/
		L_1_1_MatrixDifferenceNorm__ResetPixelScoreKernel << <1, 1 >> >(dev_pixel_score_);

		/* Compute launch parameters for difference. Want same size as sub image nut with no dilation padding at edges. */
		int diff_kernel_left_x = max(min(bounding_box_A[0], bounding_box_B[0]), 0);
		int diff_kernel_bottom_y = max(min(bounding_box_A[1], bounding_box_B[1]), 0);
		int diff_kernel_right_x = min(max(bounding_box_A[2], bounding_box_B[2]), width - 1);
		int diff_kernel_top_y = min(max(bounding_box_A[3], bounding_box_B[3]), height - 1);
		int diff_kernel_cropped_width = diff_kernel_right_x - diff_kernel_left_x + 1;
		int diff_kernel_cropped_height = diff_kernel_top_y - diff_kernel_bottom_y + 1;

		dim_grid_image_processing_ = dim3(
			ceil(static_cast<double>(diff_kernel_cropped_width) / sqrt(static_cast<double>(threads_per_block))),
			ceil(static_cast<double>(diff_kernel_cropped_height) / sqrt(static_cast<double>(threads_per_block))));

		/*L_{1,1} Matrix Norm Difference Kernel*/
		L_1_1_MatrixDifferenceNorm_DifferenceKernel << <dim_grid_image_processing_, threads_per_block >> >(
			image_A->GetDeviceImagePointer(), image_B->GetDeviceImagePointer(), dev_pixel_score_,
			width, height, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);

		/*Numerator of Pixel Score (See Mahfouz Paper: (Sum of Pixel Input * Pixel Projected)/(Sum of Pixel Projected) )*/
		cudaMemcpy(pixel_score_, dev_pixel_score_, sizeof(int), cudaMemcpyDeviceToHost);
		return pixel_score_[0];
	}
}
