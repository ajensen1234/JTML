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
__global__ void FastImplantDilationMetric_ResetPixelScoreKernel(int* dev_pixel_score) {
	dev_pixel_score[0] = 0;
}

__global__ void FastImplantDilationMetric_EdgeKernel(unsigned char* dev_image, int sub_left_x, int sub_bottom_y,
                                                     int sub_right_x, int sub_top_y, int width, int dilation) {
	/*Following notes assume 16 by 16 block size.
	/*Note: THERE MIGHT BE ARTIFACTS IN THE BUFFER PADDINGS (SIDES OF IMAGES).
	SHOULD BE HARMLESS, and fixing would decrease speed.*/

	/*This section is a little complicated. We are loading in 14 by 14 sections of the image in 16 by 16 chunks.
	Therefore we have 1 pixel of padding on each side. This is why we must have at least one dilation. Inside of the image
	the inner "core" of the loaded tiles touch each other. Ah, shared memory...*/

	/*Convert thread ID to pixel ID in original image coordinates (zero based, width by height sized)*/
	int correspondingPixelXToThread = sub_left_x - 1 + blockIdx.x * (blockDim.x - 2) + threadIdx.x;
	int correspondingPixelYToThread = sub_bottom_y - 1 + blockIdx.y * (blockDim.y - 2) + threadIdx.y;

	/*Make Sure in subCroppedImage (can only overflow above or to right since anchored at bottom left).
	Dilation is included to prevent a line on top and/or right.*/
	if (correspondingPixelXToThread <= sub_right_x + dilation && correspondingPixelYToThread <= sub_top_y + dilation) {
		int localThreadId = (threadIdx.y * blockDim.x) + threadIdx.x;
		int projectionId = correspondingPixelYToThread * width + correspondingPixelXToThread;


		/*Now load to shared 16 by 16 array the silhouette image surrounding the 14 by 14 block that is being edge detected*/
		extern __shared__ unsigned char sharedSilhouette[];
		sharedSilhouette[localThreadId] = dev_image[projectionId];
		__syncthreads();

		/* Now Only Care about inside 14 by 14 grid */
		if (0 < threadIdx.x && threadIdx.x < blockDim.x - 1 && 0 < threadIdx.y && threadIdx.y < blockDim.y - 1) {
			int left = localThreadId - 1;
			int right = localThreadId + 1;
			int top = localThreadId - blockDim.x;
			int bottom = localThreadId + blockDim.x;
			if (sharedSilhouette[localThreadId] == WHITE_PIXEL && (
				sharedSilhouette[left] == BLACK_PIXEL || sharedSilhouette[right] == BLACK_PIXEL ||
				sharedSilhouette[top] == BLACK_PIXEL || sharedSilhouette[bottom] == BLACK_PIXEL ||
				sharedSilhouette[bottom - 1] == BLACK_PIXEL || sharedSilhouette[bottom + 1] == BLACK_PIXEL ||
				sharedSilhouette[top - 1] == BLACK_PIXEL || sharedSilhouette[top + 1] == BLACK_PIXEL))
				dev_image[projectionId] = EDGE_PIXEL;
		}
	}
}

__global__ void FastImplantDilationMetric_DilateKernel(unsigned char* dev_image, int width, int height,
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
		if (dev_image[i] == EDGE_PIXEL) {
			for (int j = 1; j <= dilation; j++) {
				for (int k = 1; k <= dilation; k++) {
					location = i + l * j * width + r * k;
					pixel = dev_image[location];
					if (pixel == WHITE_PIXEL || pixel == BLACK_PIXEL)
						dev_image[location] = DILATED_PIXEL;
				}
			}
		}
	}
}

__global__ void FastImplantDilationMetric_DifferenceKernel(unsigned char* dev_image,
                                                           unsigned char* dev_comparison_image, int* result, int width,
                                                           int height,
                                                           int diff_kernel_left_x, int diff_kernel_bottom_y,
                                                           int diff_kernel_cropped_width) {
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
		if (pixel == DILATED_PIXEL || pixel == EDGE_PIXEL) {
			if (dev_comparison_image[i] == WHITE_PIXEL)
				atomicAdd(&result[0], 1);
			else
				atomicSub(&result[0], 1);
		}
	}

}


namespace gpu_cost_function {

	/*Computes DIRECT-JTA Dilation Metric Very Quickly*/
	double GPUMetrics::FastImplantDilationMetric(GPUImage* rendered_image, GPUDilatedFrame* comparison_frame,
	                                             int dilation) {

		/*Extract Bounding Box*/
		int* bounding_box = rendered_image->GetBoundingBox();

		/*Height and Width*/
		int height = rendered_image->GetFrameHeight();
		int width = rendered_image->GetFrameWidth();

		/*Reset the Pixel Score*/
		FastImplantDilationMetric_ResetPixelScoreKernel << <1, 1 >> >(dev_pixel_score_);

		/*Explanation of widths and heights
		IMAGE PROCESSING STAGE :
		-NOTES : This is the edge detection, edge dilation, and overlap with sample image computation stage.Relevant
		kernels are :
		a).edgeKernel
		b).dilateKernel
		c).differenceKernel
		- croppedWidth: The width of the cropped projection(removes black space and forms a bounding rectangle).
		- croppedHeight : The height of the cropped projection(removes black space and forms a bounding rectangle).
		- subCroppedWidth : The width of the cropped projection that is not in the dilation dead space betweeen
		the "width" and the "leanWidth" (removed on either side).
		- subCroppedHeight : The height of the cropped projection that is not in the dilation dead space betweeen
		the "height" and the "leanHeight" (removed on either side).
		- dilatedSubCroppedWidth : The subCroppedWidth with dilation padding(dilatedSubCroppedWidth = subCroppedWidth
		+ 2 * dilation).
		- dilatedSubCroppedHeight : The subCroppedHeight with dilation padding(dilatedSubCroppedHeight = subCroppedHeight
		+ 2 * dilation).

		WARNING : Pixel(x, y) coordinates are ZERO - BASED!!!
		*******************************************************************************************************************/
		int sub_left_x = max(bounding_box[0] - dilation, dilation);
		int sub_bottom_y = max(bounding_box[1] - dilation, dilation);
		int sub_right_x = min(bounding_box[2] + dilation, width - dilation - 1);
		int sub_top_y = min(bounding_box[3] + dilation, height - dilation - 1);
		int sub_cropped_width = sub_right_x - sub_left_x + 1;
		int sub_cropped_height = sub_top_y - sub_bottom_y + 1;

		/* Compute launch parameters for edge detection.
		If 256, we have 16 x 16 Blocks (Read in at one less on less on all 4 sides, so 14 x 14). */
		dim_block_image_processing_ = dim3::dim3(
			ceil(sqrt(static_cast<double>(threads_per_block))),
			ceil(sqrt(static_cast<double>(threads_per_block))));
		dim_grid_image_processing_ = dim3::dim3(
			ceil(static_cast<double>(sub_cropped_width) / static_cast<double>(dim_block_image_processing_.x - 2)),
			ceil(static_cast<double>(sub_cropped_height) / static_cast<double>(dim_block_image_processing_.y - 2)));

		/*Compute Edge Detection*/
		FastImplantDilationMetric_EdgeKernel << <dim_grid_image_processing_, dim_block_image_processing_,
			dim_block_image_processing_.x * dim_block_image_processing_.y * sizeof(unsigned char) >> >(
				rendered_image->GetDeviceImagePointer(), sub_left_x, sub_bottom_y, sub_right_x, sub_top_y, width,
				dilation);

		/* Compute launch parameters for dilation. Want 4 times the size of the sub image. */
		dim_grid_image_processing_ = dim3::dim3(
			ceil(2.0 * sub_cropped_width / sqrt(static_cast<double>(threads_per_block))),
			ceil(2.0 * sub_cropped_height / sqrt(static_cast<double>(threads_per_block))));

		/*Dilation Kernel*/
		FastImplantDilationMetric_DilateKernel << <dim_grid_image_processing_, threads_per_block >> >(
			rendered_image->GetDeviceImagePointer(), width, height,
			sub_left_x, sub_bottom_y, sub_cropped_width, dilation);


		/* Compute launch parameters for difference. Want same size as sub image nut with no dilation padding at edges. */
		int diff_kernel_left_x = max(bounding_box[0] - dilation, 0);
		int diff_kernel_bottom_y = max(bounding_box[1] - dilation, 0);
		int diff_kernel_right_x = min(bounding_box[2] + dilation, width - 1);
		int diff_kernel_top_y = min(bounding_box[3] + dilation, height - 1);
		int diff_kernel_cropped_width = diff_kernel_right_x - diff_kernel_left_x + 1;
		int diff_kernel_cropped_height = diff_kernel_top_y - diff_kernel_bottom_y + 1;

		dim_grid_image_processing_ = dim3::dim3(
			ceil(static_cast<double>(diff_kernel_cropped_width) / sqrt(static_cast<double>(threads_per_block))),
			ceil(static_cast<double>(diff_kernel_cropped_height) / sqrt(static_cast<double>(threads_per_block))));

		/*Calculate Regions of No Overlap With Comparison Image*/
		FastImplantDilationMetric_DifferenceKernel << <dim_grid_image_processing_, threads_per_block >> >(
			rendered_image->GetDeviceImagePointer(), comparison_frame->GetDeviceImagePointer(), dev_pixel_score_,
			width, height, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);

		/*Return Pixel Score (# of Pixels that are white dilated edge and  black in comparison image (which is
		a dilated version of the edge detected original x ray) minus the number of pixels that are white in the comparison
		image and white in the dilated edge)*/
		cudaMemcpy(pixel_score_, dev_pixel_score_, sizeof(int), cudaMemcpyDeviceToHost);
		return -1 * pixel_score_[0];
	};
}
