/*GPU Image Functions Header*/
#include "gpu_image_functions.cuh"

/*Cuda*/
#include "cuda_runtime.h"
#include "cuda.h"
/*Cuda Random*/
#include <curand.h>
#include <curand_kernel.h>

/*Grayscale Colors*/
#include "pixel_grayscale_colors.h"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"

/*Random C++*/
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

/*Kernels*/
/*Blending Two Images Kernel*/
__global__ void BlendGrayscaleKernel(unsigned char* dev_dest, unsigned char *dev_second, float alpha, int width, int height) {

	/*Global Thread*/
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	/*If Correct Width and Height*/
	if (i < width*height)
	{
		dev_dest[i] = alpha* dev_dest[i] + (1 - alpha)*dev_second[i];
	}
}

/*Pasting Non Black Pixel Kernel*/
__global__ void PasteNonBlackPixelsKernel(unsigned char* dev_dest, unsigned char *dev_second, int width, int height) {

	/*Global Thread*/
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	/*If Correct Width and Height*/
	if (i < width*height)
	{
		if (dev_second[i] > 0)
			dev_dest[i] = dev_second[i];
	}
}

/*Normalize Image to Range*/
__global__ void InitializeMaxMinKernel(int *dev_max, int* dev_min)
{
	dev_max[0] = 0;
	dev_min[0] = 255;
}
__global__ void GetMaxMinPixelsKernel(unsigned char* dev_image, int *dev_max, int* dev_min, int width, int height) {

	/*Global Thread*/
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	/*If Correct Width and Height*/
	if (i < width*height)
	{
		atomicMax((int *)&dev_max[0], dev_image[i]);
		atomicMin((int *)&dev_min[0], dev_image[i]);
	}
}
__global__ void ScaleImageToRangeKernel(unsigned char* dev_image, int *dev_max, int* dev_min, unsigned int lower_bound, unsigned int upper_bound, int width, int height) {

	/*Global Thread*/
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	/*If Correct Width and Height*/
	if (i < width*height)
	{
		dev_image[i] = ((float)(dev_image[i] - dev_min[0]) / (float)(dev_max[0] - dev_min[0])) * ((float) (upper_bound - lower_bound)) + lower_bound;
	}
}

/*Convolution Kernel*/
__global__ void ConvolutionKernel(unsigned char* dev_dest_image, unsigned char* dev_input_image, float* dev_kernel, int kernel_size, int width, int height) {

	/*Global Thread*/
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	/*If Correct Width and Height*/
	if (i < width*height)
	{
		/*Convert i into coordinates for grid*/
		int i_row = i / (width);
		int i_col = i - i_row* (width);
		
		/*Only proceed if not out of bounds, otherwise paste original pixel (this ignores the boundaries essentially)*/
		if ((i_row < (height - (kernel_size - 1) / 2)) && (i_row >= ((kernel_size - 1) / 2)) && (i_col <  (width - (kernel_size - 1) / 2)) && (i_col >= ((kernel_size - 1) / 2))) {
			float value = 0;
			/*For Each Row*/
			for (int row = -1 * ((kernel_size - 1) / 2); row <= ((kernel_size - 1) / 2); row++) {
				/*For Each col*/
				for (int col = -1 * ((kernel_size - 1) / 2); col <= ((kernel_size - 1) / 2); col++) {
					/*Get Kernel Index*/
					int ker_row = row + ((kernel_size - 1) / 2);
					int ker_col = -1*col + ((kernel_size - 1) / 2); //The -1 flips the kernel so its in the order described in the header

					/*Update Pixel*/
					value += (float)dev_input_image[row*width + i + col] * dev_kernel[ker_row*kernel_size + ker_col];
				}
			}
			dev_dest_image[i] = value;
		}
		else {
			dev_dest_image[i] = dev_input_image[i];
		}

		
	}

}

/*Random Image Noise*/
__global__ void AddUniformRandomNoiseKernel(unsigned char* dev_image, float* dev_random_container, int lower_bound, int upper_bound, int width, int height) {

	/*Global Thread*/
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	/*If Correct Width and Height*/
	if (i < width*height)
	{
		int noisy_pixel = dev_image[i] + dev_random_container[i]*(upper_bound - lower_bound) + lower_bound;
		if (noisy_pixel > 255)
			noisy_pixel = 255;
		else if (noisy_pixel < 0)
			noisy_pixel = 0;
		dev_image[i] = noisy_pixel;
	}
}

/*Compile Grid*/
__global__ void CompileGridKernel(unsigned char** dev_images, unsigned char* dev_grid, int image_width, int image_height, int grid_width, int grid_height) {
	/*Global Thread*/
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	/*If Correct Width and Height*/
	if (i < image_width*grid_width*image_height*grid_height)
	{
		/*Convert i into coordinates for grid*/
		int i_row = i / (image_width*grid_width);
		int i_col = i - i_row* (image_width*grid_width); // Essentially modulo. This is faster?
		/*Grid Row and Column Indices (Grids are matrices of images)*/
		int grid_row = i_row / image_height;
		int grid_col = i_col / image_width;
		/*Image Row and Column Indices (Image ae matrices of pixels)*/
		int img_row = i_row - grid_row*image_height;
		int img_col = i_col - grid_col*image_width;

		/*Convert Grid Indices to Selection of GPU Image on device*/
		int dev_imag_ind = grid_row*grid_width + grid_col;

		/*Convert Image Indices to single "i" style index to access smaller image pixels*/
		int dev_imag_i = img_row*image_width + img_col;

		// char* x = dev_images[dev_imag_ind];
		//dev_images[dev_imag_ind][i/4];// x[dev_imag_i];/*255.0*((float)dev_imag_i / (float)(250 * 250));*/// 
		dev_grid[i] = dev_images[dev_imag_ind][dev_imag_i];
	}
}

namespace gpu_cost_function {
	/*Blend two grayscale images. The result is stored in the first image which is referred to as the "destination image".
	The parameter alpha ranges from 0 to 1 and blends as follows: ALPHA*destination_image_pixel + (1 - ALPHA)*secondary_image_pixel.
	Bool return value indicates success.*/
	bool BlendGrayscaleImages(GPUImage* destination_image, GPUImage* secondary_image, float alpha) {
		/*Make Sure Alpha is between 0 and 1*/
		if (alpha < 0)
			alpha = 0;
		else if (alpha > 1)
			alpha = 1;

		/*Clear Previous Errors*/
		cudaGetLastError();

		/*Make sure images are same height and width*/
		int height = destination_image->GetFrameHeight();
		int width = destination_image->GetFrameWidth();
		if (height != secondary_image->GetFrameHeight() || width != secondary_image->GetFrameWidth())
			return false;

		/*Image Processing Dimension Grid*/
		dim3 dim_grid_image_processing_ = dim3::dim3(
			ceil((double)(width) / sqrt((double)threads_per_block)),
			ceil((double)(height) / sqrt((double)threads_per_block)));

		/*Kernel*/
		BlendGrayscaleKernel << <dim_grid_image_processing_, threads_per_block >> >(destination_image->GetDeviceImagePointer(), secondary_image->GetDeviceImagePointer(), alpha,
			width, height);

		/*CUDA Get Last Error*/
		return (cudaSuccess == cudaGetLastError());
	}

	/*Paste Non Zero (Black) Pixels of secondary_image on top of destination_image
	Bool return value indicates success.*/
	bool PasteNonBlackPixels(GPUImage* destination_image, GPUImage* secondary_image) {
		/*Clear Previous Errors*/
		cudaGetLastError();

		/*Make sure images are same height and width*/
		int height = destination_image->GetFrameHeight();
		int width = destination_image->GetFrameWidth();
		if (height != secondary_image->GetFrameHeight() || width != secondary_image->GetFrameWidth())
			return false;

		/*Image Processing Dimension Grid*/
		dim3 dim_grid_image_processing_ = dim3::dim3(
			ceil((double)(width) / sqrt((double)threads_per_block)),
			ceil((double)(height) / sqrt((double)threads_per_block)));

		/*Kernel*/
		PasteNonBlackPixelsKernel << <dim_grid_image_processing_, threads_per_block >> >(destination_image->GetDeviceImagePointer(), secondary_image->GetDeviceImagePointer(),
			width, height);


		/*CUDA Get Last Error*/
		return (cudaSuccess == cudaGetLastError());
	}
	bool PasteNonBlackPixels(unsigned char* dev_destination_image, unsigned char* dev_secondary_image, int height, int width) {
		/*Clear Previous Errors*/
		cudaGetLastError();

		/*Image Processing Dimension Grid*/
		dim3 dim_grid_image_processing_ = dim3::dim3(
			ceil((double)(width) / sqrt((double)threads_per_block)),
			ceil((double)(height) / sqrt((double)threads_per_block)));

		/*Kernel*/
		PasteNonBlackPixelsKernel << <dim_grid_image_processing_, threads_per_block >> >(dev_destination_image, dev_secondary_image,
			width, height);


		/*CUDA Get Last Error*/
		return (cudaSuccess == cudaGetLastError());
	}

	/*Scale grayscale image so that the min and max pixels in the original image are transformed linearly to the bounds specified in the
	arguments. This new image is returned in the original image.
	Bool return value indicates success.*/
	bool ScaleGrayscaleToRange(GPUImage* grayscale_image, unsigned int lower_bound, unsigned int upper_bound) {
		/*Make Sure Bounds are within 0 - 255 range*/
		if (lower_bound < 0)
			lower_bound = 0;
		else if (lower_bound > 255)
			lower_bound = 255;
		if (upper_bound < 0)
			upper_bound = 0;
		else if (upper_bound > 255)
			upper_bound = 255;

		/*Clear Previous Errors*/
		cudaGetLastError();

		/*Make sure images are same height and width*/
		int height = grayscale_image->GetFrameHeight();
		int width = grayscale_image->GetFrameWidth();

		/*Image Processing Dimension Grid*/
		dim3 dim_grid_image_processing_ = dim3::dim3(
			ceil((double)(width) / sqrt((double)threads_per_block)),
			ceil((double)(height) / sqrt((double)threads_per_block)));

		/*Create Device Pointer Container for Max and Min*/
		int* dev_max = 0;
		int* dev_min = 0;
		cudaMalloc((void**)&dev_max, 1 * sizeof(int));
		cudaMalloc((void**)&dev_min, 1 * sizeof(int));
		InitializeMaxMinKernel << <1, 1 >> > (dev_max, dev_min); // Sets dev_max = 0, dev_min = 255

		/*Kernels*/
		GetMaxMinPixelsKernel << <dim_grid_image_processing_, threads_per_block >> > (grayscale_image->GetDeviceImagePointer(), dev_max, dev_min, width, height);
		ScaleImageToRangeKernel << <dim_grid_image_processing_, threads_per_block >> > (grayscale_image->GetDeviceImagePointer(), dev_max, dev_min, lower_bound, upper_bound, width, height);
		
		/*Free Memory for dev-max/min*/
		cudaFree(dev_max);
		cudaFree(dev_min);

		/*CUDA Get Last Error*/
		return (cudaSuccess == cudaGetLastError());

	}

	/*Convolution Operation. Does not use zero padding, but rather convolves on the subimage that is inside the regular image but would allow the
	kernel to fully execute. Kernel size (assumes square) is the number of pixels (this must be odd). It is assumed that the kernel array is
	given in top left to right/ top to bottom order (for instance in a 3 by 3 kernel:
	0 1 2
	3 4 5
	6 7 8
	specifies the order of the kernel).
	Results from convolving the kernel with the input image are spit out into the destination image.
	Bool return value indicates success.*/
	bool  Convolve(GPUImage* dest_image, GPUImage* input_image, float* dev_kernel, int kernel_size) {
		/*Make Sure Kernel is Odd*/
		if (kernel_size % 2 == 0)
			return false;

		/*Clear Previous Errors*/
		cudaGetLastError();

		/*Make sure images are same height and width*/
		int height = dest_image->GetFrameHeight();
		int width = dest_image->GetFrameWidth();
		if (height != input_image->GetFrameHeight() || width != input_image->GetFrameWidth())
			return false;

		/*Image Processing Dimension Grid*/
		dim3 dim_grid_image_processing_ = dim3::dim3(
			ceil((double)(width) / sqrt((double)threads_per_block)),
			ceil((double)(height) / sqrt((double)threads_per_block)));

		/*Kernel*/
		ConvolutionKernel << <dim_grid_image_processing_, threads_per_block >> > (dest_image->GetDeviceImagePointer(),input_image->GetDeviceImagePointer(),dev_kernel,kernel_size, width, height);

		/*CUDA Get Last Error*/
		return (cudaSuccess == cudaGetLastError());
	}


	/*Add Uniform random noise to each pixel (individually generated for each pixel) which is added to the existing pixel. If Value is out of bounds [0,255]
	snap to the lower or upper bound. Uniform range runs from lower bound to upper bound.
	MUST CALL InitializeCUDARandom FIRST!!!
	Bool return value indicates success.*/
	bool AddUniformNoise(GPUImage* grayscale_image, curandGenerator_t* prng, float* dev_random_container, int lower_bound, int upper_bound){
		
		/*Clear Previous Errors*/
		cudaGetLastError();

		/*Get Height and Width*/
		int height = grayscale_image->GetFrameHeight();
		int width = grayscale_image->GetFrameWidth();

		/*Image Processing Dimension Grid*/
		dim3 dim_grid_image_processing_ = dim3::dim3(
			ceil((double)(width) / sqrt((double)threads_per_block)),
			ceil((double)(height) / sqrt((double)threads_per_block)));

		/*Generate Random Uniform*/
		curandGenerateUniform(*prng, dev_random_container, width * height);

		/*Kernels*/
		AddUniformRandomNoiseKernel << <dim_grid_image_processing_, threads_per_block >> > (grayscale_image->GetDeviceImagePointer(), dev_random_container, lower_bound, upper_bound, width, height);


		/*CUDA Get Last Error*/
		return (cudaSuccess == cudaGetLastError());

	}

	/*Copies An Array of GPUImages to one Grid. Image width and height are for the smaller images.
	Grid with and height are expressed in number of small images.
	Bool return value indiates success.*/
	bool CompileGrid(unsigned char** dev_images, unsigned char* dev_grid, int image_width, int image_height, int grid_width, int grid_height) {
		/*Clear Previous Errors*/
		cudaGetLastError();

		/*Grid Pixel Dimensions*/
		int grid_pixel_height = image_height*grid_height;
		int grid_pixel_width = image_width*grid_width;

		/*Image Processing Dimension Grid*/
		dim3 dim_grid_image_processing_ = dim3::dim3(
			ceil((double)(grid_pixel_width) / sqrt((double)threads_per_block)),
			ceil((double)(grid_pixel_height) / sqrt((double)threads_per_block)));

		/*Kernels*/
		CompileGridKernel << <dim_grid_image_processing_, threads_per_block >> > (dev_images, dev_grid, image_width, image_height, grid_width, grid_height);

		/*CUDA Get Last Error*/
		return (cudaSuccess == cudaGetLastError());

	}
}