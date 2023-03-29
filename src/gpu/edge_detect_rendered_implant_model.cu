/*GPU Metrics Header*/
#include "gpu/gpu_metrics.cuh"

/*Cuda*/
#include "cuda_runtime.h"
#include "cuda.h"

/*Grayscale Colors*/
#include "pixel_grayscale_colors.h"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"


__global__ void EdgeDetectRenderedImplantModel_EdgeKernel(unsigned char* dev_image, int sub_left_x, int sub_bottom_y,
                                                          int sub_right_x, int sub_top_y, int width) {
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
	if (correspondingPixelXToThread <= sub_right_x && correspondingPixelYToThread <= sub_top_y) {
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

__global__ void EdgeDetectRenderedImplantModel_GrayEdgeToWhitePassKernel(
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
		if (pixel == EDGE_PIXEL) {
			dev_image[i] = WHITE_PIXEL;
		}
		else {
			dev_image[i] = BLACK_PIXEL;
		}
	}

}

namespace gpu_cost_function {

	/*Edge Detect Rendered Silhouette from GPU Model's GPU Image (Returns true if no error)
	Edge detected version is spit back out to the GPU Image on the Implant Model
	Can technically use on any image, just know it only marks the border between white pixels and black pixels*/
	bool GPUMetrics::EdgeDetectRenderedImplantModel(GPUImage* rendered_model_image) {

		/*Clear Previous Errors*/
		cudaGetLastError();

		/*Extract Bounding Box*/
		int* bounding_box = rendered_model_image->GetBoundingBox();

		/*Height and Width*/
		int height = rendered_model_image->GetFrameHeight();
		int width = rendered_model_image->GetFrameWidth();

		/* Compute launch parameters for edge detection.
		If 256, we have 16 x 16 Blocks (Read in at one less on less on all 4 sides, so 14 x 14). */
		int sub_cropped_width = bounding_box[2] - bounding_box[0] + 1;
		int sub_cropped_height = bounding_box[3] - bounding_box[1] + 1;
		dim_block_image_processing_ = dim3(
			ceil(sqrt(static_cast<double>(threads_per_block))),
			ceil(sqrt(static_cast<double>(threads_per_block))));
		dim_grid_image_processing_ = dim3(
			ceil(static_cast<double>(sub_cropped_width) / static_cast<double>(dim_block_image_processing_.x - 2)),
			ceil(static_cast<double>(sub_cropped_height) / static_cast<double>(dim_block_image_processing_.y - 2)));

		/*Compute Edge Detection*/
		EdgeDetectRenderedImplantModel_EdgeKernel << <dim_grid_image_processing_, dim_block_image_processing_,
			dim_block_image_processing_.x * dim_block_image_processing_.y * sizeof(unsigned char) >> >(
				rendered_model_image->GetDeviceImagePointer(), bounding_box[0], bounding_box[1], bounding_box[2],
				bounding_box[3], width);

		/*Change Launch Parameters For Gray Edge to White Edge Pass*/
		dim_grid_image_processing_ = dim3(
			ceil(static_cast<double>(sub_cropped_width) / sqrt(static_cast<double>(threads_per_block))),
			ceil(static_cast<double>(sub_cropped_height) / sqrt(static_cast<double>(threads_per_block))));

		/*Change Gray Edges to White, and All Others to Black*/
		EdgeDetectRenderedImplantModel_GrayEdgeToWhitePassKernel << <dim_grid_image_processing_, threads_per_block >> >(
			rendered_model_image->GetDeviceImagePointer(),
			width, height, bounding_box[0], bounding_box[1], sub_cropped_width);

		/*CUDA Get Last Error*/
		return (cudaSuccess == cudaGetLastError());
	}
}
