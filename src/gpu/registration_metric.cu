///*Registration Metric Header*/
//#include "registration_metric.cuh"
//
///*OpenCV 3.1 Library*/
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//
///*Standard Library*/
//#include <iostream>
//
///*Uchar Colors*/
//#define WHITE_PIXEL 255
//#define BLACK_PIXEL 0
//
///*CUDA Custom Registration Namespace (Compiling as DLL)*/
//namespace gpu_cost_function {
//
//	/*Constructor*/
//	RegistrationMetric::RegistrationMetric() {}
//	RegistrationMetric::RegistrationMetric(int width, int height) {
//		/*Initialize Basic Variables*/
//		width_ = width;
//		if (width_ < 1) width_ = 1;
//		height_ = height;
//		if (height_ < 1) height_ = 1;
//		intialized_cuda_ = false;
//		metric_score_ = -1; /*Could never really be negative so this is a good error check*/
//		dilation_comparison_white_pix_count_ = 0;
//		dilation_comparison_pixel_sum_ = 0;
//
//		/*Initialize Device Variables to NULL*/
//		dev_edge_comparison_image_ = 0;
//		dev_dilation_comparison_image_ = 0;
//		dev_intensity_comparison_image_ = 0;
//		dev_image_ = 0;
//		dev_pixel_score_ = 0;
//		dev_dilation_comparison_white_pix_count_ = 0;
//		dev_dilation_comparison_pixel_sum_ = 0;
//	}
//
//	/*Destructor*/
//	RegistrationMetric::~RegistrationMetric() {
//		/*Free CUDA*/
//		FreeCuda();
//	}
//
//	void RegistrationMetric::FreeCuda() {
//		/*Free CUDA*/
//		cudaFree(dev_dilation_comparison_image_);
//		cudaFree(dev_intensity_comparison_image_);
//		cudaFree(dev_pixel_score_);
//		cudaFree(dev_dilation_comparison_white_pix_count_);
//		cudaFree(dev_dilation_comparison_pixel_sum_);
//
//		/*Free Host*/
//		cudaFreeHost(pixel_score_);
//	}
//
//	__global__ void ComparisonWhitePixel(unsigned char *dev_dilation_comparison_image, int *dev_comparison_white_pix_count, int width, int height) {
//		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//
//		if (i < width*height) {
//			if (dev_dilation_comparison_image[i] == WHITE_PIXEL)
//				atomicAdd((int *)&dev_comparison_white_pix_count[0], 1);
//		}
//	}
//
//	__global__ void ComparisonPixelSum(unsigned char *dev_dilation_comparison_image, int *dev_comparison_pixel_sum, int width, int height) {
//		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//
//		if (i < width*height) {
//			if (dev_dilation_comparison_image[i] != BLACK_PIXEL)
//				atomicAdd((int *)&dev_comparison_pixel_sum[0], dev_dilation_comparison_image[i]);
//		}
//	}
//
//	cudaError_t RegistrationMetric::InitializeCUDA(unsigned char* dev_image, unsigned char* edge_comparison_image, unsigned char* dilation_comparison_image, unsigned char* intensity_comparison_image, int device) {
//		/*CUDA Error Status*/
//		cudaGetLastError();  //Resets Errors
//		cudaError_t cudaStatus;
//
//		/*Choose which GPU to run on, change this on a multi-GPU system.*/
//		cudaSetDevice(device);
//
//		/*Check for Errors*/
//		cudaStatus = cudaGetLastError();
//		if (cudaStatus != cudaSuccess)
//		{
//			intialized_cuda_ = false;
//			FreeCuda();
//			return cudaStatus;
//		}
//
//		/*Initialize device image pointer which was already allocated by render engine*/
//		dev_image_ = dev_image;
//
//		/*Initialize Pinned Memory for Slightly Faster Transfer if Using Mismatched Pixel Count*/
//		cudaHostAlloc((void**)&pixel_score_, sizeof(int), cudaHostAllocDefault);
//
//		/*Allocate GPU buffers for pixel score.*/
//		cudaMalloc((void**)&dev_pixel_score_, sizeof(int));
//
//		/*Allocate GPU buffers for comparison white pixel count.*/
//		cudaMalloc((void**)&dev_dilation_comparison_white_pix_count_, sizeof(int));
//
//		/*Allocate GPU buffers for comparison pixel sum.*/
//		cudaMalloc((void**)&dev_dilation_comparison_pixel_sum_, sizeof(int));
//
//		/*Allocate GPU buffers for edge comparison image.*/
//		cudaMalloc((void**)&dev_edge_comparison_image_, width_ * height_ * sizeof(unsigned char));
//
//		/*Allocate GPU buffers for dilation comparison image.*/
//		cudaMalloc((void**)&dev_dilation_comparison_image_, width_ * height_ * sizeof(unsigned char));
//
//		/*Allocate GPU buffers for intensity comparison image.*/
//		cudaMalloc((void**)&dev_intensity_comparison_image_, width_ * height_ * sizeof(unsigned char));
//
//		/*Check for Errors*/
//		cudaStatus = cudaGetLastError();
//		if (cudaStatus != cudaSuccess)
//		{
//			intialized_cuda_ = false;
//			FreeCuda();
//			return cudaStatus;
//		}
//
//		/*Upload Edge Comparison Image from Host to Device*/
//		cudaMemcpy(dev_edge_comparison_image_, dilation_comparison_image, width_ * height_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
//
//		/*Upload Dilation Comparison Image from Host to Device*/
//		cudaMemcpy(dev_dilation_comparison_image_, dilation_comparison_image, width_ * height_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
//
//		/*Upload Intensity Comparison Image from Host to Device*/
//		cudaMemcpy(dev_intensity_comparison_image_, intensity_comparison_image, width_ * height_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
//
//		/*Upload (Reset) white pixel count for comparison image from Host to Device.*/
//		dilation_comparison_white_pix_count_ = 0;
//		cudaMemcpy(dev_dilation_comparison_white_pix_count_, &dilation_comparison_white_pix_count_, sizeof(int), cudaMemcpyHostToDevice);
//
//		/*Upload (Reset) pixel sum for comparison image from Host to Device.*/
//		dilation_comparison_pixel_sum_ = 0;
//		cudaMemcpy(dev_dilation_comparison_pixel_sum_, &dilation_comparison_pixel_sum_, sizeof(int), cudaMemcpyHostToDevice);
//
//		/*Check for Errors*/
//		cudaStatus = cudaGetLastError();
//		if (cudaStatus != cudaSuccess)
//		{
//			intialized_cuda_ = false;
//			FreeCuda();
//			return cudaStatus;
//		}
//
//		/*Get Sum of White Pixels in Dilation Comparison Image and Total Pixel Sum*/
//		dim3 dim_grid_comparison_white_pix = dim3(
//			ceil(sqrt(
//			(double)(width_*height_) / (double)256)),
//			ceil(sqrt(
//			(double)(width_*height_) / (double)256)));
//		ComparisonWhitePixel << <dim_grid_comparison_white_pix, 256 >> >(dev_dilation_comparison_image_, dev_dilation_comparison_white_pix_count_, width_, height_);
//		cudaMemcpy(&dilation_comparison_white_pix_count_, dev_dilation_comparison_white_pix_count_, sizeof(int), cudaMemcpyDeviceToHost);
//		ComparisonPixelSum << <dim_grid_comparison_white_pix, 256 >> >(dev_dilation_comparison_image_, dev_dilation_comparison_pixel_sum_, width_, height_);
//		cudaMemcpy(&dilation_comparison_pixel_sum_, dev_dilation_comparison_pixel_sum_, sizeof(int), cudaMemcpyDeviceToHost);
//
//		/*Check for Errors*/
//		cudaStatus = cudaGetLastError();
//		if (cudaStatus == cudaSuccess) intialized_cuda_ = true;
//		else {
//			intialized_cuda_ = false;
//			FreeCuda();
//		}
//
//		return cudaStatus;
//	}
//
//	/*Reset Comparison Image Pointer*/
//	cudaError_t RegistrationMetric::SetEdgeComparisonImage(unsigned char* edge_comparison_image) {
//		/*Check Initialized First*/
//		if (!intialized_cuda_) {
//			std::cout << "\nCUDA not Initialized for Registration Metric - Cannot Set Another Comparison Image!";
//			return cudaErrorMemoryAllocation;
//		}
//
//		/*CUDA Error Status*/
//		cudaGetLastError();  //Resets Errors
//		cudaError_t cudaStatus;
//
//		/*Assuming that Comparison Image is same size as previous*/
//		/*Upload Edge Comparison Image from Host to Device*/
//		cudaMemcpy(dev_edge_comparison_image_, edge_comparison_image, width_ * height_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
//
//
//		/*Check for Errors*/
//		cudaStatus = cudaGetLastError();
//		if (cudaStatus != cudaSuccess)
//			return cudaStatus;
//
//		/*Check for Errors*/
//		cudaStatus = cudaGetLastError();
//		return cudaStatus;
//	}
//
//
//	/*Reset Comparison Image Pointer*/
//	cudaError_t RegistrationMetric::SetDilationComparisonImage(unsigned char* dilation_comparison_image) {
//		/*Check Initialized First*/
//		if (!intialized_cuda_) {
//			std::cout << "\nCUDA not Initialized for Registration Metric - Cannot Set Another Comparison Image!";
//			return cudaErrorMemoryAllocation;
//		}
//
//		/*CUDA Error Status*/
//		cudaGetLastError();  //Resets Errors
//		cudaError_t cudaStatus;
//
//		/*Assuming that Comparison Image is same size as previous*/
//		/*Upload Dilation Comparison Image from Host to Device*/
//		cudaMemcpy(dev_dilation_comparison_image_, dilation_comparison_image, width_ * height_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
//
//		/*Upload (Reset) white pixel count for comparison image from Host to Device.*/
//		dilation_comparison_white_pix_count_ = 0;
//		cudaMemcpy(dev_dilation_comparison_white_pix_count_, &dilation_comparison_white_pix_count_, sizeof(int), cudaMemcpyHostToDevice);
//
//		/*Upload (Reset) pixel sum for comparison image from Host to Device.*/
//		dilation_comparison_pixel_sum_ = 0;
//		cudaMemcpy(dev_dilation_comparison_pixel_sum_, &dilation_comparison_pixel_sum_, sizeof(int), cudaMemcpyHostToDevice);
//
//		/*Check for Errors*/
//		cudaStatus = cudaGetLastError();
//		if (cudaStatus != cudaSuccess)
//			return cudaStatus;
//
//		/*Get Sum of White Pixels in Comparison Image and Total Pixel Sum*/
//		dim3 dim_grid_comparison_white_pix = dim3(
//			ceil(sqrt(
//			(double)(width_*height_) / (double)256)),
//			ceil(sqrt(
//			(double)(width_*height_) / (double)256)));
//		ComparisonWhitePixel << <dim_grid_comparison_white_pix, 256 >> >(dev_dilation_comparison_image_, dev_dilation_comparison_white_pix_count_, width_, height_);
//		cudaMemcpy(&dilation_comparison_white_pix_count_, dev_dilation_comparison_white_pix_count_, sizeof(int), cudaMemcpyDeviceToHost);
//		ComparisonPixelSum << <dim_grid_comparison_white_pix, 256 >> >(dev_dilation_comparison_image_, dev_dilation_comparison_pixel_sum_, width_, height_);
//		cudaMemcpy(&dilation_comparison_pixel_sum_, dev_dilation_comparison_pixel_sum_, sizeof(int), cudaMemcpyDeviceToHost);
//
//		/*Check for Errors*/
//		cudaStatus = cudaGetLastError();
//		return cudaStatus;
//	}
//
//	/*Reset Comparison Image Pointer*/
//	cudaError_t RegistrationMetric::SetIntensityComparisonImage(unsigned char* intensity_comparison_image) {
//		/*Check Initialized First*/
//		if (!intialized_cuda_) {
//			std::cout << "\nCUDA not Initialized for Registration Metric - Cannot Set Another Comparison Image!";
//			return cudaErrorMemoryAllocation;
//		}
//
//		/*CUDA Error Status*/
//		cudaGetLastError();  //Resets Errors
//		cudaError_t cudaStatus;
//
//		/*Assuming that Comparison Image is same size as previous*/
//		/*Upload Comparison Image from Host to Device*/
//		cudaMemcpy(dev_intensity_comparison_image_, intensity_comparison_image, width_ * height_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
//
//
//		/*Check for Errors*/
//		cudaStatus = cudaGetLastError();
//		if (cudaStatus != cudaSuccess)
//			return cudaStatus;
//
//		/*Check for Errors*/
//		cudaStatus = cudaGetLastError();
//		return cudaStatus;
//	}
//
//	void RegistrationMetric::WriteImage(std::string file_location) {
//		/*Check Initialized First*/
//		if (!intialized_cuda_) {
//			std::cout << "\nCUDA not Initialized for Registration Metric - Cannot Write!";
//			return;
//		}
//
//		/*Array for Storing Device Image on Host*/
//		unsigned char* host_image = (unsigned char*)malloc(width_*height_ * sizeof(unsigned char));
//		cudaMemcpy(host_image, dev_image_, width_*height_ * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//
//		/*OpenCV Image Container/Write Function*/
//		cv::Mat projection_mat = cv::Mat(height_, width_, CV_8UC1, host_image); /*Reverse before flip*/
//		cv::Mat output_mat = cv::Mat(width_, height_, CV_8UC1);
//		cv::flip(projection_mat, output_mat, 0);
//		cv::imwrite(file_location, output_mat);
//
//		/*Free Array*/
//		free(host_image);
//	}
//
//}
