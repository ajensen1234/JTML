///*Dilation Metric Header*/
//#include "metric_toolbox.cuh"
//
///*Standard Library*/
//#include <algorithm>
//#include <iostream>
//
///*Uchar Colors*/
//#define WHITE_PIXEL 255
//#define BLACK_PIXEL 0
//#define EDGE_PIXEL 100
//#define DILATED_PIXEL 99
//
///*CUDA Custom Registration Namespace (Compiling as DLL)*/
//namespace gpu_cost_function {
//	MetricToolbox::MetricToolbox(int width, int height) : RegistrationMetric(width, height) {
//		/*Initialize Dilation to 1 and black silhouette to true*/
//		dilation_ = 1;
//		black_comp_silhouette_ = true;
//	}
//
//	MetricToolbox::MetricToolbox() {}
//
//	__global__ void ResetPixelScoreKernel(int *dev_pixel_score) {
//		dev_pixel_score[0] = 0;
//	}
//
//	__global__ void EdgeKernel(unsigned char *dev_image, int sub_left_x, int sub_bottom_y,
//		int sub_right_x, int sub_top_y, int width, int dilation)
//	{
//		/*Following notes assume 16 by 16 block size.
//		/*Note: THERE MIGHT BE ARTIFACTS IN THE BUFFER PADDINGS (SIDES OF IMAGES).
//		SHOULD BE HARMLESS, and fixing would decrease speed.*/
//
//		/*This section is a little complicated. We are loading in 14 by 14 sections of the image in 16 by 16 chunks.
//		Therefore we have 1 pixel of padding on each side. This is why we must have at least one dilation. Inside of the image
//		the inner "core" of the loaded tiles touch each other. Ah, shared memory...*/
//
//		/*Convert thread ID to pixel ID in original image coordinates (zero based, width by height sized)*/
//		int correspondingPixelXToThread = sub_left_x - 1 + blockIdx.x*(blockDim.x - 2) + threadIdx.x;
//		int correspondingPixelYToThread = sub_bottom_y - 1 + blockIdx.y*(blockDim.y - 2) + threadIdx.y;
//
//		/*Make Sure in subCroppedImage (can only overflow above or to right since anchored at bottom left).
//		Dilation is included to prevent a line on top and/or right.*/
//		if (correspondingPixelXToThread <= sub_right_x + dilation && correspondingPixelYToThread <= sub_top_y + dilation)
//		{
//			int localThreadId = (threadIdx.y * blockDim.x) + threadIdx.x;
//			int projectionId = correspondingPixelYToThread * width + correspondingPixelXToThread;
//
//
//			/*Now load to shared 16 by 16 array the silhouette image surrounding the 14 by 14 block that is being edge detected*/
//			extern __shared__ unsigned char sharedSilhouette[];
//			sharedSilhouette[localThreadId] = dev_image[projectionId];
//			__syncthreads();
//
//			/* Now Only Care about inside 14 by 14 grid */
//			if (0 < threadIdx.x && threadIdx.x < blockDim.x - 1 && 0 < threadIdx.y && threadIdx.y < blockDim.y - 1)
//			{
//				int left = localThreadId - 1;
//				int right = localThreadId + 1;
//				int top = localThreadId - blockDim.x;
//				int bottom = localThreadId + blockDim.x;
//				if (sharedSilhouette[localThreadId] == WHITE_PIXEL && (
//					sharedSilhouette[left] == BLACK_PIXEL || sharedSilhouette[right] == BLACK_PIXEL ||
//					sharedSilhouette[top] == BLACK_PIXEL || sharedSilhouette[bottom] == BLACK_PIXEL ||
//					sharedSilhouette[bottom - 1] == BLACK_PIXEL || sharedSilhouette[bottom + 1] == BLACK_PIXEL ||
//					sharedSilhouette[top - 1] == BLACK_PIXEL || sharedSilhouette[top + 1] == BLACK_PIXEL))
//					dev_image[projectionId] = EDGE_PIXEL;
//			}
//		}
//	}
//
//	__global__ void DilateKernel(unsigned char *dev_image, int width, int height,
//		int sub_left_x, int sub_bottom_y, int sub_cropped_width, int dilation)
//	{
//		/*Global Thread*/
//		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//
//		/*Search Direction*/
//		int l = 2 * ((i % 4) / 2) - 1;
//		int r = 2 * (i % 2) - 1;
//		i = i / 4;
//		i = (i / sub_cropped_width)*width + (i % sub_cropped_width) + sub_bottom_y * width + sub_left_x;
//
//		/*Reused local variables*/
//		int pixel;
//		int location;
//
//		/*If Correct Width and Height*/
//		if (i < width*height)
//		{
//			if (dev_image[i] == EDGE_PIXEL)
//			{
//				for (int j = 1; j <= dilation; j++)
//				{
//					for (int k = 1; k <= dilation; k++)
//					{
//						location = i + l * j * width + r * k;
//						pixel = dev_image[location];
//						if (pixel == WHITE_PIXEL || pixel == BLACK_PIXEL)
//							dev_image[location] = DILATED_PIXEL;
//					}
//				}
//			}
//		}
//	}
//
//	__global__ void DifferenceKernel(unsigned char* dev_image, unsigned char *dev_comparison_image, int *result, int width, int height,
//		int diff_kernel_left_x, int diff_kernel_bottom_y, int diff_kernel_cropped_width)
//	{
//		/*Global Thread*/
//		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//
//		/*Convert to Subsize*/
//		i = (i / diff_kernel_cropped_width)*width + (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width + diff_kernel_left_x;
//
//		/*Storage Container for Loaded Pixel*/
//		int pixel;
//
//		/*If Correct Width and Height*/
//		if (i < width*height)
//		{
//			pixel = dev_image[i];
//			if (pixel == DILATED_PIXEL || pixel == EDGE_PIXEL)
//			{
//				if (dev_comparison_image[i] == WHITE_PIXEL)
//					atomicAdd((int *)&result[0], 1);
//				else atomicSub((int *)&result[0], 1);
//			}
//		}
//
//	}
//
//	double MetricToolbox::ComputeDilationMetric(int *bounding_box) {
//		/*Reset the Pixel Score*/
//		ResetPixelScoreKernel << <1, 1 >> >(dev_pixel_score_);
//
//		/*Explanation of widths and heights
//		IMAGE PROCESSING STAGE :
//		-NOTES : This is the edge detection, edge dilation, and overlap with sample image computation stage.Relevant
//		kernels are :
//		a).edgeKernel
//		b).dilateKernel
//		c).differenceKernel
//		- croppedWidth: The width of the cropped projection(removes black space and forms a bounding rectangle).
//		- croppedHeight : The height of the cropped projection(removes black space and forms a bounding rectangle).
//		- subCroppedWidth : The width of the cropped projection that is not in the dilation dead space betweeen
//		the "width" and the "leanWidth" (removed on either side).
//		- subCroppedHeight : The height of the cropped projection that is not in the dilation dead space betweeen
//		the "height" and the "leanHeight" (removed on either side).
//		- dilatedSubCroppedWidth : The subCroppedWidth with dilation padding(dilatedSubCroppedWidth = subCroppedWidth
//		+ 2 * dilation).
//		- dilatedSubCroppedHeight : The subCroppedHeight with dilation padding(dilatedSubCroppedHeight = subCroppedHeight
//		+ 2 * dilation).
//
//		WARNING : Pixel(x, y) coordinates are ZERO - BASED!!!
//		*******************************************************************************************************************/
//		int sub_left_x = max(bounding_box[0] - dilation_, dilation_);
//		int sub_bottom_y = max(bounding_box[1] - dilation_, dilation_);
//		int sub_right_x = min(bounding_box[2] + dilation_, width_ - dilation_ - 1);
//		int sub_top_y = min(bounding_box[3] + dilation_, height_ - dilation_ - 1);
//		int sub_cropped_width = sub_right_x - sub_left_x + 1;
//		int sub_cropped_height = sub_top_y - sub_bottom_y + 1;
//
//		/* Compute launch parameters for edge detection.
//		If 256, we have 16 x 16 Blocks (Read in at one less on less on all 4 sides, so 14 x 14). */
//		dim_block_image_processing_ = dim3(
//			ceil(sqrt((double)(threads_per_block))),
//			ceil(sqrt((double)(threads_per_block))));
//		dim_grid_image_processing_ = dim3(
//			ceil((double)sub_cropped_width / (double)(dim_block_image_processing_.x - 2)),
//			ceil((double)sub_cropped_height / (double)(dim_block_image_processing_.y - 2)));
//
//		/*Compute Edge Detection*/
//		EdgeKernel << <dim_grid_image_processing_, dim_block_image_processing_,
//			dim_block_image_processing_.x*dim_block_image_processing_.y * sizeof(unsigned char) >> >(
//				dev_image_, sub_left_x, sub_bottom_y, sub_right_x, sub_top_y, width_, dilation_);
//
//		/* Compute launch parameters for dilation. Want 4 times the size of the sub image. */
//		dim_grid_image_processing_ = dim3(
//			ceil((double)(2.0 * sub_cropped_width) / sqrt((double)threads_per_block)),
//			ceil((double)(2.0 * sub_cropped_height) / sqrt((double)threads_per_block)));
//
//		/*Dilation Kernel*/
//		DilateKernel << <dim_grid_image_processing_, threads_per_block >> >(dev_image_, width_, height_,
//			sub_left_x, sub_bottom_y, sub_cropped_width, dilation_);
//
//
//		/* Compute launch parameters for difference. Want same size as sub image nut with no dilation padding at edges. */
//		int diff_kernel_left_x = max(bounding_box[0] - dilation_, 0);
//		int diff_kernel_bottom_y = max(bounding_box[1] - dilation_, 0);
//		int diff_kernel_right_x = min(bounding_box[2] + dilation_, width_ - 1);
//		int diff_kernel_top_y = min(bounding_box[3] + dilation_, height_ - 1);
//		int diff_kernel_cropped_width = diff_kernel_right_x - diff_kernel_left_x + 1;
//		int diff_kernel_cropped_height = diff_kernel_top_y - diff_kernel_bottom_y + 1;
//
//		dim_grid_image_processing_ = dim3(
//			ceil((double)(diff_kernel_cropped_width) / sqrt((double)threads_per_block)),
//			ceil((double)(diff_kernel_cropped_height) / sqrt((double)threads_per_block)));
//
//		/*Calculate Regions of No Overlap With Comparison Image*/
//		DifferenceKernel << <dim_grid_image_processing_, threads_per_block >> >(dev_image_, dev_dilation_comparison_image_, dev_pixel_score_,
//			width_, height_, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);
//
//		/*Return Pixel Score (# of Pixels that are white dilated edge and  black in comparison image (which is
//		a dilated version of the edge detected original x ray) minus the number of pixels that are white in the comparison
//		image and white in the dilated edge)*/
//		cudaMemcpy(pixel_score_, dev_pixel_score_, sizeof(int), cudaMemcpyDeviceToHost);
//
//
//		/*Calculate Tweaked Value of Metric Score:*/
//		/*(Sum of Original Image white pixels - pixel score) scaled by a similar normalizing constant used in old JTA*/
//		metric_score_ = 0.001 * (dilation_comparison_white_pix_count_ - pixel_score_[0]); // See what effect normalizing constant has on 
//
//
//		return metric_score_;
//	}
//
//	/*Some NOISE in this method, shouldn't change results really*/
//	__global__ void DilateKernelInverseMahfouzScale(unsigned char *dev_image, int width, int height,
//		int sub_left_x, int sub_bottom_y, int sub_cropped_width)
//	{
//		/*Global Thread*/
//		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//
//		/*Search Direction*/
//		int l = 2 * ((i % 4) / 2) - 1;
//		int r = 2 * (i % 2) - 1;
//		i = i / 4;
//		i = (i / sub_cropped_width)*width + (i % sub_cropped_width) + sub_bottom_y * width + sub_left_x;
//
//		/*Reused local variables*/
//		int pixel;
//		int location;
//
//		/* Measures The Reduction from the Dilated Picel Value, based on distance away(L2) from edge
//		Needs to Be Multiplied by L2 distance form line and substracted from dilated pixel to get weight.
//		Then in difference kernel, should be scaled by 255/Dilated_Pixel (which should be 99)*/
//		float inverse_reduction = (float)DILATED_PIXEL / (float)(sqrt(32.0));
//		float distance_L2;
//
//		/*If Correct Width and Height*/
//		if (i < width*height)
//		{
//			if (dev_image[i] == EDGE_PIXEL)
//			{
//				for (int j = 1; j <= 3; j++)
//				{
//					for (int k = 1; k <= 3; k++)
//					{
//						location = i + l * j * width + r * k;
//						pixel = dev_image[location];
//						distance_L2 = sqrt((float)k*k + j * j);
//						unsigned char scaled_dilation_value = (DILATED_PIXEL - distance_L2 * inverse_reduction);
//						if (pixel == WHITE_PIXEL || pixel < scaled_dilation_value)
//							dev_image[location] = scaled_dilation_value;
//					}
//				}
//			}
//		}
//	}
//
//	__global__ void EdgeMahfouzNumeratorKernel(unsigned char* dev_image, unsigned char *dev_comparison_image, int *result, int width, int height,
//		int diff_kernel_left_x, int diff_kernel_bottom_y, int diff_kernel_cropped_width)
//	{
//		/*Global Thread*/
//		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//
//		/*Convert to Subsize*/
//		i = (i / diff_kernel_cropped_width)*width + (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width + diff_kernel_left_x;
//
//		/*Storage Container for Loaded Pixel*/
//		int pixel;
//
//		/*If Correct Width and Height*/
//		if (i < width*height)
//		{
//			pixel = dev_image[i];
//			if (pixel > BLACK_PIXEL && pixel < WHITE_PIXEL)
//			{
//				if (dev_comparison_image[i] != BLACK_PIXEL)
//					atomicAdd((int *)&result[0], 2.55*pixel); /*Input Image is just Sobel edge detected image*/
//			}
//		}
//
//	}
//
//	__global__ void EdgeMahfouzDenominatorKernel(unsigned char* dev_image, unsigned char *dev_comparison_image, int *result, int width, int height,
//		int diff_kernel_left_x, int diff_kernel_bottom_y, int diff_kernel_cropped_width)
//	{
//		/*Global Thread*/
//		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//
//		/*Convert to Subsize*/
//		i = (i / diff_kernel_cropped_width)*width + (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width + diff_kernel_left_x;
//
//		/*Storage Container for Loaded Pixel*/
//		int pixel;
//
//		/*If Correct Width and Height*/
//		if (i < width*height)
//		{
//			pixel = dev_image[i];
//			if (pixel > BLACK_PIXEL && pixel < WHITE_PIXEL)
//			{
//				atomicAdd((int *)&result[0], 2.55*pixel); //Only Multiplies Edge and DIlated Edge Pixels (Normalizes them on 0 - 255 scale)
//			}
//		}
//
//	}
//
//	__global__ void BlackSilhouetteMahfouzNumeratorKernel(unsigned char* dev_image, unsigned char *dev_intensity_comparison_image, int *result, int width, int height,
//		int diff_kernel_left_x, int diff_kernel_bottom_y, int diff_kernel_cropped_width)
//	{
//		/*Global Thread*/
//		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//
//		/*Convert to Subsize*/
//		i = (i / diff_kernel_cropped_width)*width + (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width + diff_kernel_left_x;
//
//		/*Storage Container for Loaded Pixel*/
//		int pixel;
//
//		/*If Correct Width and Height*/
//		if (i < width*height)
//		{
//			pixel = dev_image[i];
//			if (pixel == WHITE_PIXEL)
//			{
//				atomicAdd((int *)&result[0], (255 - dev_intensity_comparison_image[i]));
//			}
//		}
//
//	}
//
//	__global__ void WhiteSilhouetteMahfouzNumeratorKernel(unsigned char* dev_image, unsigned char *dev_intensity_comparison_image, int *result, int width, int height,
//		int diff_kernel_left_x, int diff_kernel_bottom_y, int diff_kernel_cropped_width)
//	{
//		/*Global Thread*/
//		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//
//		/*Convert to Subsize*/
//		i = (i / diff_kernel_cropped_width)*width + (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width + diff_kernel_left_x;
//
//		/*Storage Container for Loaded Pixel*/
//		int pixel;
//
//		/*If Correct Width and Height*/
//		if (i < width*height)
//		{
//			pixel = dev_image[i];
//			if (pixel == WHITE_PIXEL)
//			{
//				atomicAdd((int *)&result[0], dev_intensity_comparison_image[i]);
//			}
//
//		}
//
//	}
//
//	__global__ void IntensityMahfouzDenominatorKernel(unsigned char* dev_image, unsigned char *dev_intensity_comparison_image, int *result, int width, int height,
//		int diff_kernel_left_x, int diff_kernel_bottom_y, int diff_kernel_cropped_width)
//	{
//		/*Global Thread*/
//		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//
//		/*Convert to Subsize*/
//		i = (i / diff_kernel_cropped_width)*width + (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width + diff_kernel_left_x;
//
//		/*Storage Container for Loaded Pixel*/
//		int pixel;
//
//		/*If Correct Width and Height*/
//		if (i < width*height)
//		{
//			pixel = dev_image[i];
//			if (pixel == WHITE_PIXEL)
//			{
//				atomicAdd((int *)&result[0], 1);
//			}
//		}
//
//	}
//
//	double MetricToolbox::ComputeMahfouzMetric(int *launch_packet) {
//
//		/*Reset the Pixel Score*/
//		ResetPixelScoreKernel << <1, 1 >> >(dev_pixel_score_);
//
//		/* Compute launch parameters for difference. Want same size as sub image nut with no dilation padding at edges. */
//		int diff_kernel_left_x = max(launch_packet[0], 0);
//		int diff_kernel_bottom_y = max(launch_packet[1], 0);
//		int diff_kernel_right_x = min(launch_packet[2], width_ - 1);
//		int diff_kernel_top_y = min(launch_packet[3], height_ - 1);
//		int diff_kernel_cropped_width = diff_kernel_right_x - diff_kernel_left_x + 1;
//		int diff_kernel_cropped_height = diff_kernel_top_y - diff_kernel_bottom_y + 1;
//
//		dim_grid_image_processing_ = dim3(
//			ceil((double)(diff_kernel_cropped_width) / sqrt((double)threads_per_block)),
//			ceil((double)(diff_kernel_cropped_height) / sqrt((double)threads_per_block)));
//
//		/*Calculate Mahfouz Edge Kernel (Numerator)*/
//		if (black_comp_silhouette_) {
//			BlackSilhouetteMahfouzNumeratorKernel << <dim_grid_image_processing_, threads_per_block >> >(dev_image_, dev_intensity_comparison_image_, dev_pixel_score_,
//				width_, height_, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);
//		}
//		else {
//			WhiteSilhouetteMahfouzNumeratorKernel << <dim_grid_image_processing_, threads_per_block >> >(dev_image_, dev_intensity_comparison_image_, dev_pixel_score_,
//				width_, height_, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);
//		}
//
//		/*Numerator of Pixel Score (See Mahfouz Paper: (Sum of Pixel Input * Pixel Projected)/(Sum of Pixel Projected) )*/
//		cudaMemcpy(pixel_score_, dev_pixel_score_, sizeof(int), cudaMemcpyDeviceToHost);
//		double intensity_score = pixel_score_[0];
//		double num = intensity_score;
//		/*Reset and Calculate Denominator*/
//		ResetPixelScoreKernel << <1, 1 >> >(dev_pixel_score_);
//
//		/*Calculate Mahfouz  Kernel (Denominator)*/
//		IntensityMahfouzDenominatorKernel << <dim_grid_image_processing_, threads_per_block >> >(dev_image_, dev_intensity_comparison_image_, dev_pixel_score_,
//			width_, height_, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);
//
//		/*Denominator of Pixel Score (See Mahfouz Paper: (Sum of Pixel Input * Pixel Projected)/(Sum of Pixel Projected) )*/
//		cudaMemcpy(pixel_score_, dev_pixel_score_, sizeof(int), cudaMemcpyDeviceToHost);
//		if (pixel_score_ != 0)
//			intensity_score = intensity_score / (double)pixel_score_[0];
//		else
//			intensity_score = 0;
//
//		/*Contour Section*/
//		/*Reset the Pixel Score*/
//		ResetPixelScoreKernel << <1, 1 >> >(dev_pixel_score_);
//
//		int sub_left_x = max(launch_packet[0] - 3, 3);
//		int sub_bottom_y = max(launch_packet[1] - 3, 3);
//		int sub_right_x = min(launch_packet[2] + 3, width_ - 3 - 1);
//		int sub_top_y = min(launch_packet[3] + 3, height_ - 3 - 1);
//		int sub_cropped_width = sub_right_x - sub_left_x + 1;
//		int sub_cropped_height = sub_top_y - sub_bottom_y + 1;
//
//		/* Compute launch parameters for edge detection.
//		If 256, we have 16 x 16 Blocks (Read in at one less on less on all 4 sides, so 14 x 14). */
//		dim_block_image_processing_ = dim3(
//			ceil(sqrt((double)(threads_per_block))),
//			ceil(sqrt((double)(threads_per_block))));
//		dim_grid_image_processing_ = dim3(
//			ceil((double)sub_cropped_width / (double)(dim_block_image_processing_.x - 2)),
//			ceil((double)sub_cropped_height / (double)(dim_block_image_processing_.y - 2)));
//
//		/*Compute Edge Detection*/
//		EdgeKernel << <dim_grid_image_processing_, dim_block_image_processing_,
//			dim_block_image_processing_.x*dim_block_image_processing_.y * sizeof(unsigned char) >> >(
//				dev_image_, sub_left_x, sub_bottom_y, sub_right_x, sub_top_y, width_, 3);
//
//		/* Compute launch parameters for dilation. Want 4 times the size of the sub image. */
//		dim_grid_image_processing_ = dim3(
//			ceil((double)(2.0 * sub_cropped_width) / sqrt((double)threads_per_block)),
//			ceil((double)(2.0 * sub_cropped_height) / sqrt((double)threads_per_block)));
//
//		/*Dilation Inverse Kernel for Dilation Level of 3 With Linear Inverse Weighting*/
//		DilateKernelInverseMahfouzScale << <dim_grid_image_processing_, threads_per_block >> >(dev_image_, width_, height_,
//			sub_left_x, sub_bottom_y, sub_cropped_width);
//
//		/* Compute launch parameters for difference. Want same size as sub image nut with no dilation padding at edges. */
//		diff_kernel_left_x = max(launch_packet[0] - 3, 0);
//		diff_kernel_bottom_y = max(launch_packet[1] - 3, 0);
//		diff_kernel_right_x = min(launch_packet[2] + 3, width_ - 1);
//		diff_kernel_top_y = min(launch_packet[3] + 3, height_ - 1);
//		diff_kernel_cropped_width = diff_kernel_right_x - diff_kernel_left_x + 1;
//		diff_kernel_cropped_height = diff_kernel_top_y - diff_kernel_bottom_y + 1;
//
//		dim_grid_image_processing_ = dim3(
//			ceil((double)(diff_kernel_cropped_width) / sqrt((double)threads_per_block)),
//			ceil((double)(diff_kernel_cropped_height) / sqrt((double)threads_per_block)));
//
//		/*Calculate Mahfouz Edge Kernel (Numerator)*/
//		EdgeMahfouzNumeratorKernel << <dim_grid_image_processing_, threads_per_block >> >(dev_image_, dev_dilation_comparison_image_, dev_pixel_score_,
//			width_, height_, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);
//
//		/*Numerator of Pixel Score (See Mahfouz Paper: (Sum of Pixel Input * Pixel Projected)/(Sum of Pixel Projected) )*/
//		cudaMemcpy(pixel_score_, dev_pixel_score_, sizeof(int), cudaMemcpyDeviceToHost);
//		double contour_score = 255.0 * pixel_score_[0];
//
//		/*Reset and Calculate Denominator*/
//		ResetPixelScoreKernel << <1, 1 >> >(dev_pixel_score_);
//
//		/*Calculate Mahfouz Edge Kernel (Denominator)*/
//		EdgeMahfouzDenominatorKernel << <dim_grid_image_processing_, threads_per_block >> >(dev_image_, dev_dilation_comparison_image_, dev_pixel_score_,
//			width_, height_, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width);
//
//		/*Denominator of Pixel Score (See Mahfouz Paper: (Sum of Pixel Input * Pixel Projected)/(Sum of Pixel Projected) )*/
//		cudaMemcpy(pixel_score_, dev_pixel_score_, sizeof(int), cudaMemcpyDeviceToHost);
//		if (pixel_score_ != 0)
//			contour_score = contour_score / (double)pixel_score_[0];
//		else
//			contour_score = 0;
//		return contour_score * (-2.67) + intensity_score * (-1);
//	}
//
//	void MetricToolbox::SetDilation(int dilation) {
//		if (dilation < 1) dilation = 1;
//		dilation_ = dilation;
//	}
//
//	int MetricToolbox::GetDilation() {
//		return dilation_;
//	}
//
//	void MetricToolbox::SetBlackSilhouette(bool black_silhouette) {
//		black_comp_silhouette_ = black_silhouette;
//	};
//
//	bool MetricToolbox::GetBlackSilhouette() {
//		return black_comp_silhouette_;
//	};
//}
