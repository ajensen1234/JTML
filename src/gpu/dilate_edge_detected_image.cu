/*GPU Metrics Header*/
#include "gpu/gpu_metrics.cuh"

/*Cuda*/
#include <cuda_runtime.h>
#include <cuda.h>

/*Grayscale Colors*/
#include "pixel_grayscale_colors.h"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"

/*Kernels*/
__global__ void DilateEdgeDetectedImage_DilateKernel(unsigned char* dev_image, int width, int height,
                                                     int sub_left_x, int sub_bottom_y, int sub_cropped_width,
                                                     int dilation) {
	/*Global Thread*/
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	/*Search Direction*/
	int l = 2 * ((i % 4) / 2) - 1;
	int r = 2 * (i % 2) - 1;
	i = i / 4;
	i = (i / sub_cropped_width) * width + (i % sub_cropped_width) + sub_bottom_y * width + sub_left_x;

	/*Reused local variables*/
	int pixel;
	int location;

	/*If Correct Width and Height*/
	if (i < width * height) {
		if (dev_image[i] == WHITE_PIXEL) {
			for (int j = 1; j <= dilation; j++) {
				for (int k = 1; k <= dilation; k++) {
					location = i + l * j * width + r * k;
					pixel = dev_image[location];
					if (pixel != WHITE_PIXEL)
						dev_image[location] = DILATED_PIXEL;
				}
			}
		}
	}
}

__global__ void DilateEdgeDetectedImage_GrayDilatedEdgeToWhitePassKernel(
	unsigned char* dev_image, int width, int height,
	int diff_kernel_left_x, int diff_kernel_bottom_y, int diff_kernel_cropped_width) {
	/*Global Thread*/
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	/*Convert to Subsize*/
	i = (i / diff_kernel_cropped_width) * width + (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width +
		diff_kernel_left_x;

	/*Storage Container for Loaded Pixel*/
	int pixel;

	/*If Correct Width and Height*/
	if (i < width * height) {
		pixel = dev_image[i];
		if (pixel == DILATED_PIXEL) {
			dev_image[i] = WHITE_PIXEL;
		}
	}

}


namespace gpu_cost_function {
	bool GPUMetrics::DilateEdgeDetectedImage(GPUImage* edge_detected_image, int dilation) {
		/*Check Dilation is Sufficient*/
		if (dilation < 1)
			return true;

		/*Clear Previous Errors*/
		cudaGetLastError();

		/*Extract Bounding Box*/
		int* bounding_box = edge_detected_image->GetBoundingBox();

		/*Height and Width*/
		int height = edge_detected_image->GetFrameHeight();
		int width = edge_detected_image->GetFrameWidth();

		/* Compute launch parameters for edge detection.
		If 256, we have 16 x 16 Blocks (Read in at one less on less on all 4 sides, so 14 x 14). */
		int sub_left_x = max(bounding_box[0] - dilation, dilation);
		int sub_bottom_y = max(bounding_box[1] - dilation, dilation);
		int sub_right_x = min(bounding_box[2] + dilation, width - dilation - 1);
		int sub_top_y = min(bounding_box[3] + dilation, height - dilation - 1);
		int sub_cropped_width = sub_right_x - sub_left_x + 1;
		int sub_cropped_height = sub_top_y - sub_bottom_y + 1;
		dim_block_image_processing_ = dim3(
			ceil(sqrt(static_cast<double>(threads_per_block))),
			ceil(sqrt(static_cast<double>(threads_per_block))));

		/* Compute launch parameters for dilation. Want 4 times the size of the sub image. */
		dim_grid_image_processing_ = dim3(
			ceil(2.0 * sub_cropped_width / sqrt(static_cast<double>(threads_per_block))),
			ceil(2.0 * sub_cropped_height / sqrt(static_cast<double>(threads_per_block))));

		/*Dilation Kernel*/
		DilateEdgeDetectedImage_DilateKernel << <dim_grid_image_processing_, threads_per_block >> >(
			edge_detected_image->GetDeviceImagePointer(), width, height,
			sub_left_x, sub_bottom_y, sub_cropped_width, dilation);

		/*Change Launch Parameters For Gray Dilated Edge to White Edge Pass*/
		dim_grid_image_processing_ = dim3(
			ceil(static_cast<double>(sub_cropped_width) / sqrt(static_cast<double>(threads_per_block))),
			ceil(static_cast<double>(sub_cropped_height) / sqrt(static_cast<double>(threads_per_block))));

		/*Change Gray Dilated Edges to White, and All Others to Black*/
		DilateEdgeDetectedImage_GrayDilatedEdgeToWhitePassKernel << <dim_grid_image_processing_, threads_per_block >> >(
			edge_detected_image->GetDeviceImagePointer(),
			width, height, sub_left_x, sub_bottom_y, sub_cropped_width);

		/*CUDA Get Last Error*/
		return (cudaSuccess == cudaGetLastError());
	};
}
