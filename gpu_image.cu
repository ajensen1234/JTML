/*GPU Image Header*/
#include "gpu_image.cuh"

/*Cuda*/
#include "cuda_runtime.h"

/*OpenCV 3.1 Library*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace gpu_cost_function {
	GPUImage::GPUImage(int width, int height, int gpu_device) {
		/*Start out Assuming Initialized Incorrectly*/
		initialized_correctly_ = false;
		
		/*CUDA Error Status*/
		cudaGetLastError(); //Resets Errors
		cudaError_t cudaStatus;

		/*Initialize Pinned Memory for Slightly Faster Transfer*/
		cudaHostAlloc((void**)&bounding_box_, 4 * sizeof(int), cudaHostAllocDefault);
		bounding_box_[0] = 0; bounding_box_[1] = 0; bounding_box_[2] = width - 1; bounding_box_[3] = height - 1;

		/*Choose which GPU to run on, change this on a multi-GPU system.*/
		cudaSetDevice(gpu_device);

		/*Check for Errors*/
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			image_on_gpu_ = false;
		}
		else {
			/*Initialize Private Host Variables*/
			width_ = width;
			height_ = height;
			device_ = gpu_device;
			dev_image_ = 0;

			/*Allocate GPU buffers for image, triangles.*/
			cudaMalloc((void**)&dev_image_, width_ * height_ * sizeof(unsigned char));

			/*Check for Errors*/
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				image_on_gpu_ = false;
				/*Free CUDA*/
				cudaFree(dev_image_);
			}
			else {
				image_on_gpu_ = true;
			}
		}
		/*Correctly Initialized*/
		initialized_correctly_ = true;
	};

	GPUImage::GPUImage(int width, int height, int gpu_device, unsigned char* host_image) {
		/*Start out Assuming Initialized Incorrectly*/
		initialized_correctly_ = false;

		/*CUDA Error Status*/
		cudaGetLastError();  //Resets Errors
		cudaError_t cudaStatus;

		/*Initialize Pinned Memory for Slightly Faster Transfer*/
		cudaHostAlloc((void**)&bounding_box_, 4 * sizeof(int), cudaHostAllocDefault);
		bounding_box_[0] = 0; bounding_box_[1] = 0; bounding_box_[2] = width - 1; bounding_box_[3] = height - 1;

		/*Choose which GPU to run on, change this on a multi-GPU system.*/
		cudaSetDevice(gpu_device);

		/*Check for Errors*/
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cudaFreeHost(bounding_box_);
			initialized_correctly_ = false;
			image_on_gpu_ = false;
			return;
		}
		else {
			/*Initialize Private Host Variables*/
			width_ = width;
			height_ = height;
			device_ = gpu_device;
			dev_image_ = 0;

			/*Allocate GPU buffers for image, triangles.*/
			cudaMalloc((void**)&dev_image_, width_ * height_ * sizeof(unsigned char));

			/*Check for Errors*/
			cudaStatus = cudaGetLastError();

			if (cudaStatus != cudaSuccess || width_*height_ == 0) {
				initialized_correctly_ = false;
				image_on_gpu_ = false;
				/*Free CUDA*/
				cudaFree(dev_image_);
				cudaFreeHost(bounding_box_);
				return;
			}
			else {
				/*Upload Image from Host to Device*/
				cudaMemcpy(dev_image_, host_image, width_ * height_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
				
				/*Check for Errors*/
				cudaStatus = cudaGetLastError();

				if (cudaStatus != cudaSuccess) {
					initialized_correctly_ = false;
					image_on_gpu_ = false;
					/*Free CUDA*/
					cudaFree(dev_image_);
					cudaFreeHost(bounding_box_);
					return;
				}
				else {
					/*Correctly Initialized*/
					initialized_correctly_ = true;
					image_on_gpu_ = true;
					return;
				}
			}
		}
		
	};

	GPUImage::GPUImage() {
		image_on_gpu_ = false;
		initialized_correctly_ = false;
		width_ = 0;
		height_ = 0;
		device_ = -1;
		dev_image_ = 0;
		bounding_box_ = 0;

	};

	GPUImage::~GPUImage() {
		/*Free CUDA*/
		cudaFree(dev_image_);
		cudaFreeHost(bounding_box_);
	};

	bool GPUImage::UploadBlankImageToGPU(int width, int height) {
		/*CUDA Error Status*/
		cudaGetLastError();  //Resets Errors
		cudaError_t cudaStatus;

		/*Check if Image Already on GPU*/
		if (image_on_gpu_) {

			/*Choose which GPU to run on, change this on a multi-GPU system.*/
			cudaSetDevice(device_);

			/*Check for Errors*/
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				return false;
			}
			else {
				/*Free CUDA*/
				cudaFree(dev_image_);
				cudaFreeHost(bounding_box_);
				if (cudaStatus != cudaSuccess) {
					return false;
				}
				else {
					image_on_gpu_ = false;
				}
			}
		}

		/*Initialize Pinned Memory for Slightly Faster Transfer*/
		cudaHostAlloc((void**)&bounding_box_, 4 * sizeof(int), cudaHostAllocDefault);
		bounding_box_[0] = 0; bounding_box_[1] = 0; bounding_box_[2] = width - 1; bounding_box_[3] = height - 1;

		/*Choose which GPU to run on, change this on a multi-GPU system.*/
		cudaSetDevice(device_);

		/*Check for Errors*/
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cudaFreeHost(bounding_box_);
			image_on_gpu_ = false;
		}
		else {
			/*Initialize Private Host Variables*/
			width_ = width;
			height_ = height;
			dev_image_ = 0;

			/*Allocate GPU buffers for image, triangles.*/
			cudaMalloc((void**)&dev_image_, width_ * height_ * sizeof(unsigned char));

			/*Check for Errors*/
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				image_on_gpu_ = false;
				/*Free CUDA*/
				cudaFree(dev_image_);
				cudaFreeHost(bounding_box_);
			}
			else {
				image_on_gpu_ = true;
			}
		}
		return image_on_gpu_;
	};

	bool GPUImage::UploadImageToGPU(int width, int height, unsigned char* host_image) {
		/*CUDA Error Status*/
		cudaGetLastError();  //Resets Errors
		cudaError_t cudaStatus;

		/*Check if Image Already on GPU*/
		if (image_on_gpu_) {

			/*Choose which GPU to run on, change this on a multi-GPU system.*/
			cudaSetDevice(device_);

			/*Check for Errors*/
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				return false;
			}
			else {
				/*Free CUDA*/
				cudaFree(dev_image_);
				cudaFreeHost(bounding_box_);
				if (cudaStatus != cudaSuccess) {
					return false;
				}
				else {
					image_on_gpu_ = false;
				}
			}
		}

		/*Initialize Pinned Memory for Slightly Faster Transfer*/
		cudaHostAlloc((void**)&bounding_box_, 4 * sizeof(int), cudaHostAllocDefault);
		bounding_box_[0] = 0; bounding_box_[1] = 0; bounding_box_[2] = width - 1; bounding_box_[3] = height - 1;

		/*Choose which GPU to run on, change this on a multi-GPU system.*/
		cudaSetDevice(device_);

		/*Check for Errors*/
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cudaFreeHost(bounding_box_);
			image_on_gpu_ = false;
		}
		else {
			/*Initialize Private Host Variables*/
			width_ = width;
			height_ = height;
			dev_image_ = 0;

			/*Allocate GPU buffers for image, triangles.*/
			cudaMalloc((void**)&dev_image_, width_ * height_ * sizeof(unsigned char));

			/*Check for Errors*/
			cudaStatus = cudaGetLastError();

			if (cudaStatus != cudaSuccess) {
				image_on_gpu_ = false;
				/*Free CUDA*/
				cudaFree(dev_image_);
				cudaFreeHost(bounding_box_);
			}
			else {
				/*Upload Image from Host to Device*/
				cudaMemcpy(dev_image_, host_image, width_ * height_ * sizeof(unsigned char), cudaMemcpyHostToDevice);

				/*Check for Errors*/
				cudaStatus = cudaGetLastError();

				if (cudaStatus != cudaSuccess) {
					image_on_gpu_ = false;
					/*Free CUDA*/
					cudaFree(dev_image_);
					cudaFreeHost(bounding_box_);
				}
				else {
					image_on_gpu_ = true;
				}
			}
		}
		return image_on_gpu_;
	};

	bool GPUImage::RemoveImageFromGPU() {
		/*CUDA Error Status*/
		cudaGetLastError();  //Resets Errors
		cudaError_t cudaStatus;

		/*Check if Image Already on GPU*/
		if (image_on_gpu_) {

			/*Choose which GPU to run on, change this on a multi-GPU system.*/
			cudaSetDevice(device_);

			/*Check for Errors*/
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				return false;
			}
			else {
				/*Free CUDA*/
				cudaFree(dev_image_);
				cudaFreeHost(bounding_box_);
				if (cudaStatus != cudaSuccess) {
					return false;
				}
				else {
					image_on_gpu_ = false;
				}
			}
		}
		return true;
	};

	bool GPUImage::CheckImageOnGPU() {
		return image_on_gpu_;
	};

	unsigned char* GPUImage::GetDeviceImagePointer() {
		if (image_on_gpu_) {
			return dev_image_;
		}
		else {
			cudaFree(bounding_box_);
			bounding_box_ = 0;
			cudaFree(dev_image_);
			dev_image_ = 0;
			return dev_image_;
		}
	};

	int* GPUImage::GetBoundingBox() {
		if (image_on_gpu_) {
			return bounding_box_;
		}
		else {
			cudaFree(bounding_box_);
			bounding_box_ = 0;
			return bounding_box_;
		}
	}

	bool GPUImage::IsInitializedCorrectly() {
		return initialized_correctly_;
	}

	bool GPUImage::WriteImage(std::string file_name) {
		/*Check Initialized First*/
		if (!initialized_correctly_) {
			std::cout << "\nCUDA not Initialized for GPU Image - Cannot Write!";
			return false;
		}

		/*Array for Storing Device Image on Host*/
		unsigned char* host_image = (unsigned char*)malloc(width_*height_ * sizeof(unsigned char));
		cudaMemcpy(host_image, this->GetDeviceImagePointer(), width_*height_ * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		/*OpenCV Image Container/Write Function*/
		cv::Mat projection_mat = cv::Mat(height_, width_, CV_8UC1, host_image);/*Reverse before flip*/
		cv::Mat output_mat = cv::Mat(width_, height_, CV_8UC1);
		cv::flip(projection_mat, output_mat, 0);
		bool result = cv::imwrite(file_name, output_mat);

		/*Free Array*/
		free(host_image);
		return result;
	}

	/*Get Image Size Parameters*/
	int GPUImage::GetFrameHeight() {
		return height_;
	};
	int GPUImage::GetFrameWidth() {
		return width_;
	};
}