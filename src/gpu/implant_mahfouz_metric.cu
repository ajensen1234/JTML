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
__global__ void ImplantMahfouzMetric_ResetPixelScoreKernel(int* dev_pixel_score) {
	dev_pixel_score[0] = 0;
}

/*Some NOISE in this method, shouldn't change results really*/
__global__ void ImplantMahfouzMetric_DilateKernelInverseMahfouzScale(unsigned char* dev_image, int width, int height,
                                                                     int sub_left_x, int sub_bottom_y,
                                                                     int sub_cropped_width) {
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

	/* Measures The Reduction from the Dilated Picel Value, based on distance away(L2) from edge
	Needs to Be Multiplied by L2 distance form line and substracted from dilated pixel to get weight.
	Then in difference kernel, should be scaled by 255/Dilated_Pixel (which should be 99)*/
	float inverse_reduction = static_cast<float>(DILATED_PIXEL) / static_cast<float>(sqrt(32.0));
	float distance_L2;

	/*If Correct Width and Height*/
	if (i < width * height) {
		if (dev_image[i] == EDGE_PIXEL) {
			for (int j = 1; j <= 3; j++) {
				for (int k = 1; k <= 3; k++) {
					location = i + l * j * width + r * k;
					pixel = dev_image[location];
					distance_L2 = sqrt(static_cast<float>(k) * k + j * j);
					unsigned char scaled_dilation_value = (DILATED_PIXEL - distance_L2 * inverse_reduction);
					if (pixel == WHITE_PIXEL || pixel < scaled_dilation_value)
						dev_image[location] = scaled_dilation_value;
				}
			}
		}
	}
}

__global__ void ImplantMahfouzMetric_EdgeMahfouzNumeratorKernel(unsigned char* dev_image,
                                                                unsigned char* dev_comparison_image, int* result,
                                                                int width, int height,
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
		if (pixel > BLACK_PIXEL && pixel < WHITE_PIXEL) {
			if (dev_comparison_image[i] != BLACK_PIXEL)
				atomicAdd(&result[0], 2.55 * pixel); /*Input Image is just Sobel edge detected image*/
		}
	}

}

__global__ void ImplantMahfouzMetric_EdgeMahfouzDenominatorKernel(unsigned char* dev_image, int* result, int width,
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
		if (pixel > BLACK_PIXEL && pixel < WHITE_PIXEL) {
			atomicAdd(&result[0], 2.55 * pixel);
			//Only Multiplies Edge and DIlated Edge Pixels (Normalizes them on 0 - 255 scale)
		}
	}

}

__global__ void ImplantMahfouzMetric_SilhouetteMahfouzNumeratorKernel(unsigned char* dev_image,
                                                                      unsigned char* dev_intensity_comparison_image,
                                                                      int* result, int width, int height,
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
		if (pixel == WHITE_PIXEL) {
			atomicAdd(&result[0], dev_intensity_comparison_image[i]);
		}

	}

}

__global__ void ImplantMahfouzMetric_IntensityMahfouzDenominatorKernel(unsigned char* dev_image, int* result, int width,
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
		if (pixel == WHITE_PIXEL) {
			atomicAdd(&result[0], 1);
		}
	}

}

__global__ void ImplantMahfouzMetric_EdgeKernel(unsigned char* dev_image, int sub_left_x, int sub_bottom_y,
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

namespace gpu_cost_function {

	/*Computes Version of Mahfouz Metric Very Quickly (Subtle Changes and also Not Using Simulated Annealing Obviously...)
	The score returned is:
	This function is for implants.*/
	double GPUMetrics::ImplantMahfouzMetric(GPUImage* rendered_image, GPUDilatedFrame* comparison_dilated_frame,
	                                        GPUIntensityFrame* comparison_intensity_frame) {
		/*Extract Bounding Box*/
		int* bounding_box = rendered_image->GetBoundingBox();

		/*Height and Width*/
		int height = rendered_image->GetFrameHeight();
		int width = rendered_image->GetFrameWidth();

		/*Reset the Pixel Score*/
		ImplantMahfouzMetric_ResetPixelScoreKernel << <1, 1 >> >(dev_pixel_score_);

		/* Compute launch parameters for difference. Want same size as sub image nut with no dilation padding at edges. */
		int diff_kernel_left_x = max(bounding_box[0], 0);
		int diff_kernel_bottom_y = max(bounding_box[1], 0);
		int diff_kernel_right_x = min(bounding_box[2], width - 1);
		int diff_kernel_top_y = min(bounding_box[3], height - 1);
		int diff_kernel_cropped_width = diff_kernel_right_x - diff_kernel_left_x + 1;
		int diff_kernel_cropped_height = diff_kernel_top_y - diff_kernel_bottom_y + 1;

		dim_grid_image_processing_ = dim3::dim3(
			ceil(static_cast<double>(diff_kernel_cropped_width) / sqrt(static_cast<double>(threads_per_block))),
			ceil(static_cast<double>(diff_kernel_cropped_height) / sqrt(static_cast<double>(threads_per_block))));

		/*Calculate Mahfouz Edge Kernel (Numerator) (Note Rendered Image is ALWAYS White)*/
		ImplantMahfouzMetric_SilhouetteMahfouzNumeratorKernel << <dim_grid_image_processing_, threads_per_block >> >(
			rendered_image->GetDeviceImagePointer(), comparison_intensity_frame->GetWhiteSilhouetteDeviceImagePointer(),
			dev_pixel_score_,
			width, height, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);

		/*Numerator of Pixel Score (See Mahfouz Paper: (Sum of Pixel Input * Pixel Projected)/(Sum of Pixel Projected) )*/
		cudaMemcpy(pixel_score_, dev_pixel_score_, sizeof(int), cudaMemcpyDeviceToHost);
		double intensity_score = pixel_score_[0];
		double num = intensity_score;
		/*Reset and Calculate Denominator*/
		ImplantMahfouzMetric_ResetPixelScoreKernel << <1, 1 >> >(dev_pixel_score_);

		/*Calculate Mahfouz  Kernel (Denominator)*/
		ImplantMahfouzMetric_IntensityMahfouzDenominatorKernel << <dim_grid_image_processing_, threads_per_block >> >(
			rendered_image->GetDeviceImagePointer(), dev_pixel_score_,
			width, height, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);

		/*Denominator of Pixel Score (See Mahfouz Paper: (Sum of Pixel Input * Pixel Projected)/(Sum of Pixel Projected) )*/
		cudaMemcpy(pixel_score_, dev_pixel_score_, sizeof(int), cudaMemcpyDeviceToHost);
		if (pixel_score_ != 0)
			intensity_score = intensity_score / static_cast<double>(pixel_score_[0]);
		else
			intensity_score = 0;

		/*Contour Section*/
		/*Reset the Pixel Score*/
		ImplantMahfouzMetric_ResetPixelScoreKernel << <1, 1 >> >(dev_pixel_score_);

		int sub_left_x = max(bounding_box[0] - 3, 3);
		int sub_bottom_y = max(bounding_box[1] - 3, 3);
		int sub_right_x = min(bounding_box[2] + 3, width - 3 - 1);
		int sub_top_y = min(bounding_box[3] + 3, height - 3 - 1);
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
		ImplantMahfouzMetric_EdgeKernel << <dim_grid_image_processing_, dim_block_image_processing_,
			dim_block_image_processing_.x * dim_block_image_processing_.y * sizeof(unsigned char) >> >(
				rendered_image->GetDeviceImagePointer(), sub_left_x, sub_bottom_y, sub_right_x, sub_top_y, width, 3);

		/* Compute launch parameters for dilation. Want 4 times the size of the sub image. */
		dim_grid_image_processing_ = dim3::dim3(
			ceil(2.0 * sub_cropped_width / sqrt(static_cast<double>(threads_per_block))),
			ceil(2.0 * sub_cropped_height / sqrt(static_cast<double>(threads_per_block))));

		/*Dilation Inverse Kernel for Dilation Level of 3 With Linear Inverse Weighting*/
		ImplantMahfouzMetric_DilateKernelInverseMahfouzScale << <dim_grid_image_processing_, threads_per_block >> >(
			rendered_image->GetDeviceImagePointer(), width, height,
			sub_left_x, sub_bottom_y, sub_cropped_width);

		/* Compute launch parameters for difference. Want same size as sub image nut with no dilation padding at edges. */
		diff_kernel_left_x = max(bounding_box[0] - 3, 0);
		diff_kernel_bottom_y = max(bounding_box[1] - 3, 0);
		diff_kernel_right_x = min(bounding_box[2] + 3, width - 1);
		diff_kernel_top_y = min(bounding_box[3] + 3, height - 1);
		diff_kernel_cropped_width = diff_kernel_right_x - diff_kernel_left_x + 1;
		diff_kernel_cropped_height = diff_kernel_top_y - diff_kernel_bottom_y + 1;

		dim_grid_image_processing_ = dim3::dim3(
			ceil(static_cast<double>(diff_kernel_cropped_width) / sqrt(static_cast<double>(threads_per_block))),
			ceil(static_cast<double>(diff_kernel_cropped_height) / sqrt(static_cast<double>(threads_per_block))));

		/*Calculate Mahfouz Edge Kernel (Numerator)*/
		ImplantMahfouzMetric_EdgeMahfouzNumeratorKernel << <dim_grid_image_processing_, threads_per_block >> >(
			rendered_image->GetDeviceImagePointer(), comparison_dilated_frame->GetDeviceImagePointer(),
			dev_pixel_score_,
			width, height, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);

		/*Numerator of Pixel Score (See Mahfouz Paper: (Sum of Pixel Input * Pixel Projected)/(Sum of Pixel Projected) )*/
		cudaMemcpy(pixel_score_, dev_pixel_score_, sizeof(int), cudaMemcpyDeviceToHost);
		double contour_score = 255.0 * pixel_score_[0];

		/*Reset and Calculate Denominator*/
		ImplantMahfouzMetric_ResetPixelScoreKernel << <1, 1 >> >(dev_pixel_score_);

		/*Calculate Mahfouz Edge Kernel (Denominator)*/
		ImplantMahfouzMetric_EdgeMahfouzDenominatorKernel << <dim_grid_image_processing_, threads_per_block >> >(
			rendered_image->GetDeviceImagePointer(), dev_pixel_score_,
			width, height, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);

		/*Denominator of Pixel Score (See Mahfouz Paper: (Sum of Pixel Input * Pixel Projected)/(Sum of Pixel Projected) )*/
		cudaMemcpy(pixel_score_, dev_pixel_score_, sizeof(int), cudaMemcpyDeviceToHost);
		if (pixel_score_ != 0)
			contour_score = contour_score / static_cast<double>(pixel_score_[0]);
		else
			contour_score = 0;
		return contour_score * (-2.67) + intensity_score * (-1);
	};
}
