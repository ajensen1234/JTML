#ifndef GPU_IMAGE_FUNCTIONS_H
#define GPU_IMAGE_FUNCTIONS_H

/*Cuda*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*Cuda Random*/
#include <curand.h>
#include <curand_kernel.h>

/*GPU Frame/Model*/
#include "gpu/gpu_model.cuh"
#include "gpu/gpu_frame.cuh"
#include "gpu/gpu_edge_frame.cuh"
#include "gpu/gpu_intensity_frame.cuh"
#include "gpu/gpu_dilated_frame.cuh"

/*Pose Matrix Class*/
#include "pose_matrix.h"
#include "core/preprocessor-defs.h"

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {

	/*Blend two grayscale images. The result is stored in the first image which is referred to as the "destination image".
	 The parameter alpha ranges from 0 to 1 and blends as follows: ALPHA*destination_image_pixel + (1 - ALPHA)*secondary_image_pixel.
	 Bool return value indicates success.*/
	JTML_DLL bool BlendGrayscaleImages(GPUImage* destination_image, GPUImage* secondary_image, float alpha);
	/*Paste Non Zero (Black) Pixels of secondary_image on top of destination_image
	Bool return value indicates success. Can pass as GPU Image* or 
	with height/width.*/
	JTML_DLL bool PasteNonBlackPixels(GPUImage* destination_image, GPUImage* secondary_image);
	JTML_DLL bool PasteNonBlackPixels(unsigned char* dev_destination_image, unsigned char* dev_secondary_image, int height, int width);

	/*Scale grayscale image so that the min and max pixels in the original image are  to the bounds specified in the
	arguments and in between pixels are scaled linearly. This new image is returned in the original image.
	Bool return value indicates success.*/
	JTML_DLL bool ScaleGrayscaleToRange(GPUImage* grayscale_image, unsigned int lower_bound, unsigned int upper_bound);

	/*Convolution Operation. Does not use zero padding, but rather convolves on the subimage that is inside the regular image but would allow the
	kernel to fully execute. Kernel size (assumes square) is the number of pixels (this must be odd). It is assumed that the kernel array is 
	given in top left to right/ top to bottom order (for instance in a 3 by 3 kernel: 
	0 1 2
	3 4 5
	6 7 8
	specifies the order of the kernel).
	Results from convolving the kernel with the input image are spit out into the destination image.
	Bool return value indicates success.*/
	JTML_DLL bool Convolve(GPUImage* dest_image, GPUImage* input_image, float* dev_kernel, int kernel_size);

	/*Add Uniform random noise to each pixel (individually generated for each pixel) which is added to the existing pixel. If Value is out of bounds [0,255]
	snap to the lower or upper bound. Uniform range runs from lower bound to upper bound.
	MUST CALL InitializeCUDARandom FIRST!!!
	Bool return value indicates success.*/
	JTML_DLL bool AddUniformNoise(GPUImage* grayscale_image, curandGenerator_t* prng, float* dev_random_container,int lower_bound, int upper_bound);

	/*Copies An Array of GPUImages's device array to one grid. Image width and height are for the smaller images.
	Grid with and height are expressed in number of small images.
	Bool return value indiates success.*/
	JTML_DLL bool CompileGrid(unsigned char** dev_images, unsigned char* dev_grid, int image_width, int image_height, int grid_width, int grid_height);
}

#endif // !GPU_IMAGE_FUNCTIONS_H
