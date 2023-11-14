/*GPU Metrics Header*/
#include "gpu/gpu_metrics.cuh"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"

/*Grayscale Colors*/
#include "pixel_grayscale_colors.h"

namespace gpu_cost_function {

	/*Constructor and Destructor for GPU Metrics Class*/
	GPUMetrics::GPUMetrics() {

		/*Initialized Correctly?*/
		initialized_correctly_ = true;

		/*Initialize Pinned Memory for Slightly Faster Transfer if Using Mismatched Pixel Count*/
		cudaHostAlloc((void**)&pixel_score_, sizeof(int), cudaHostAllocDefault);
		if (cudaGetLastError() != cudaSuccess)
			initialized_correctly_ = false;

		/*Initialize Pinned Memory for IOU Intermediary */
		cudaHostAlloc((void**)&intersection_score_, sizeof(int), cudaHostAllocDefault);
		if (cudaGetLastError() != cudaSuccess)
			initialized_correctly_ = false;

		/*Initialize Pinned Memory for IOU Intermediary */
		cudaHostAlloc((void**)&union_score_, sizeof(int), cudaHostAllocDefault);
		if (cudaGetLastError() != cudaSuccess)
			initialized_correctly_ = false;

		/*Allocate GPU buffers for pixel score.*/
		cudaMalloc((void**)&dev_pixel_score_, sizeof(int));
		if (cudaGetLastError() != cudaSuccess)
			initialized_correctly_ = false;

		/*Allocate GPU buffers for IOU Intermediary*/
		cudaMalloc((void**)&dev_intersection_score_, sizeof(int));
		if (cudaGetLastError() != cudaSuccess)
			initialized_correctly_ = false;

		/*Allocate GPU buffers for IOU Intermediary*/
		cudaMalloc((void**)&dev_union_score_, sizeof(int));
		if (cudaGetLastError() != cudaSuccess)
			initialized_correctly_ = false;

		/*Allocate GPU buffers for comparison white pixel count.*/
		cudaMalloc((void**)&dev_white_pix_count_, sizeof(int));
		if (cudaGetLastError() != cudaSuccess)
			initialized_correctly_ = false;


		/*Upload (Reset) white pixel count for comparison image from Host to Device.*/
		white_pix_count_ = 0;
		cudaMemcpy(dev_white_pix_count_, &white_pix_count_, sizeof(int), cudaMemcpyHostToDevice);
		if (cudaGetLastError() != cudaSuccess)
			initialized_correctly_ = false;


        cudaHostAlloc((void**)&distance_map_score_, sizeof(int), cudaHostAllocDefault);
		if (cudaGetLastError() != cudaSuccess)
			initialized_correctly_ = false;
        // Allocate some memory for the dev dm score
        cudaMalloc((void**)&dev_distance_map_score_, sizeof(int));
		if (cudaGetLastError() != cudaSuccess){
			initialized_correctly_ = false;
        }


        // Allocating memory for the edge pixels count (GPU and CPU)
        cudaMalloc((void**)&dev_edge_pixels_count_, sizeof(int));
		if (cudaGetLastError() != cudaSuccess){
			initialized_correctly_ = false;
        }
        cudaHostAlloc((void**)&edge_pixels_count_, sizeof(int), cudaHostAllocDefault);
		if (cudaGetLastError() != cudaSuccess){
			initialized_correctly_ = false;
        }




	};

	GPUMetrics::~GPUMetrics() {
		/*Free CUDA*/
		cudaFree(dev_pixel_score_);
		cudaFree(dev_intersection_score_);
		cudaFree(dev_union_score_);
        cudaFree(dev_edge_pixels_count_);
        cudaFree(dev_distance_map_score_);
        cudaFree(dev_curvature_hausdorf_score_);

		/*Free Host*/
		cudaFreeHost(pixel_score_);
		cudaFreeHost(intersection_score_);
		cudaFreeHost(union_score_);
        cudaFreeHost(distance_map_score_);
        cudaFreeHost(edge_pixels_count_);
        cudaFreeHost(curvature_hausdorf_score_);
	};

	/*Reset White Pixel Count*/
	__global__ void ComputeSumWhitePixels__ResetWhitePixelScoreKernel(int* dev_white_pix_count_) {
		dev_white_pix_count_[0] = 0;
	}

	/*CUDA White Pixel Sum Function*/
	__global__ void WhitePixelSum(unsigned char* dev_dilation_comparison_image, int* dev_comparison_white_pix_count,
	                              int width, int height) {
		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

		if (i < width * height) {
			if (dev_dilation_comparison_image[i] == WHITE_PIXEL)
				atomicAdd(&dev_comparison_white_pix_count[0], 1);
		}
	};

	/*Computes Sum of White Pixels in Image*/
	int GPUMetrics::ComputeSumWhitePixels(GPUImage* image, cudaError* error) {

		/*Reset Errors*/
		cudaGetLastError();

		/*Reset White Pixel Count*/
		ComputeSumWhitePixels__ResetWhitePixelScoreKernel << <1, 1 >> >(dev_white_pix_count_);

		/*Get Sum of White Pixels in Dilation Comparison Image and Total Pixel Sum*/
		auto dim_grid_comparison_white_pix = dim3(
			ceil(sqrt(
				static_cast<double>(image->GetFrameWidth() * image->GetFrameHeight()) / static_cast<double>(256))),
			ceil(sqrt(
				static_cast<double>(image->GetFrameWidth() * image->GetFrameHeight()) / static_cast<double>(256))));
		WhitePixelSum << <dim_grid_comparison_white_pix, 256 >> >(image->GetDeviceImagePointer(), dev_white_pix_count_,
		                                                          image->GetFrameWidth(), image->GetFrameHeight());
		cudaMemcpy(&white_pix_count_, dev_white_pix_count_, sizeof(int), cudaMemcpyDeviceToHost);
		/*Get Errors*/
		*error = cudaGetLastError();
		return white_pix_count_;

	};


	/*Get Initialized Correctly*/
	bool GPUMetrics::IsInitializedCorrectly() {
		return initialized_correctly_;
	}
}
