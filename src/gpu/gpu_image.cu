/*GPU Image Header*/
#include "gpu/gpu_image.cuh"

#include "gpu/cuda_check.cuh"

/*Cuda*/
#include "cuda_runtime.h"

/*OpenCV 3.1 Library*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace gpu_cost_function {
namespace {
[[nodiscard]] unique_host_ptr<int>
make_default_bounding_box(int width, int height) {
    auto bounding_box = make_cuda_host_unique<int>(4);
    int* raw_bounding_box = bounding_box.get();
    raw_bounding_box[0] = 0;
    raw_bounding_box[1] = 0;
    raw_bounding_box[2] = width - 1;
    raw_bounding_box[3] = height - 1;
    return bounding_box;
}
} // namespace

GPUImage::GPUImage(int width, int height, int gpu_device) {
    /*Start out Assuming Initialized Incorrectly*/
    initialized_correctly_ = false;

    /*CUDA Error Status*/
    cudaGetLastError(); // Resets Errors
    cudaError_t cudaStatus;

    /*Initialize Pinned Memory for Slightly Faster Transfer*/
    bounding_box_ = make_default_bounding_box(width, height);

    /*Choose which GPU to run on, change this on a multi-GPU system.*/
    CUDA_CHECK(cudaSetDevice(gpu_device));

    /*Check for Errors*/
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        image_on_gpu_ = false;
    } else {
        /*Initialize Private Host Variables*/
        width_ = width;
        height_ = height;
        device_ = gpu_device;
        dev_image_.reset();

        /*Allocate GPU buffers for image, triangles.*/
        dev_image_ = make_cuda_unique<unsigned char>(width_ * height_);

        /*Check for Errors*/
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            image_on_gpu_ = false;
            /*Free CUDA*/
            dev_image_.reset();
        } else {
            image_on_gpu_ = true;
        }
    }
    /*Correctly Initialized*/
    initialized_correctly_ = true;
};

GPUImage::GPUImage(
    int width, int height, int gpu_device, unsigned char* host_image) {
    /*Start out Assuming Initialized Incorrectly*/
    initialized_correctly_ = false;

    /*CUDA Error Status*/
    cudaGetLastError(); // Resets Errors
    cudaError_t cudaStatus;

    /*Initialize Pinned Memory for Slightly Faster Transfer*/
    bounding_box_ = make_default_bounding_box(width, height);

    /*Choose which GPU to run on, change this on a multi-GPU system.*/
    CUDA_CHECK(cudaSetDevice(gpu_device));

    /*Check for Errors*/
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        bounding_box_.reset();
        initialized_correctly_ = false;
        image_on_gpu_ = false;
        return;
    }
    /*Initialize Private Host Variables*/
    width_ = width;
    height_ = height;
    device_ = gpu_device;
    dev_image_.reset();

    /*Allocate GPU buffers for image, triangles.*/
    dev_image_ = make_cuda_unique<unsigned char>(width_ * height_);

    /*Check for Errors*/
    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess || width_ * height_ == 0) {
        initialized_correctly_ = false;
        image_on_gpu_ = false;
        /*Free CUDA*/
        dev_image_.reset();
        bounding_box_.reset();
        return;
    }
    /*Upload Image from Host to Device*/
    CUDA_CHECK(cudaMemcpy(
        dev_image_.get(),
        host_image,
        width_ * height_ * sizeof(unsigned char),
        cudaMemcpyHostToDevice));

    /*Check for Errors*/
    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess) {
        initialized_correctly_ = false;
        image_on_gpu_ = false;
        /*Free CUDA*/
        dev_image_.reset();
        bounding_box_.reset();
        return;
    }
    /*Correctly Initialized*/
    initialized_correctly_ = true;
    image_on_gpu_ = true;
    return;
};

GPUImage::GPUImage() {
    image_on_gpu_ = false;
    initialized_correctly_ = false;
    width_ = 0;
    height_ = 0;
    device_ = -1;
    dev_image_.reset();
    bounding_box_.reset();
};

GPUImage::~GPUImage() = default;

bool GPUImage::UploadBlankImageToGPU(int width, int height) {
    /*CUDA Error Status*/
    cudaGetLastError(); // Resets Errors
    cudaError_t cudaStatus;

    /*Check if Image Already on GPU*/
    if (image_on_gpu_) {

        /*Choose which GPU to run on, change this on a multi-GPU system.*/
    CUDA_CHECK(cudaSetDevice(device_));

        /*Check for Errors*/
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            return false;
        }
        /*Free CUDA*/
        dev_image_.reset();
        bounding_box_.reset();
        if (cudaStatus != cudaSuccess) {
            return false;
        }
        image_on_gpu_ = false;
    }

    /*Initialize Pinned Memory for Slightly Faster Transfer*/
    bounding_box_ = make_default_bounding_box(width, height);

    /*Choose which GPU to run on, change this on a multi-GPU system.*/
    CUDA_CHECK(cudaSetDevice(device_));

    /*Check for Errors*/
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        bounding_box_.reset();
        image_on_gpu_ = false;
    } else {
        /*Initialize Private Host Variables*/
        width_ = width;
        height_ = height;
        dev_image_.reset();

        /*Allocate GPU buffers for image, triangles.*/
        dev_image_ = make_cuda_unique<unsigned char>(width_ * height_);

        /*Check for Errors*/
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            image_on_gpu_ = false;
            /*Free CUDA*/
            dev_image_.reset();
            bounding_box_.reset();
        } else {
            image_on_gpu_ = true;
        }
    }
    return image_on_gpu_;
};

bool GPUImage::UploadImageToGPU(
    int width, int height, unsigned char* host_image) {
    /*CUDA Error Status*/
    cudaGetLastError(); // Resets Errors
    cudaError_t cudaStatus;

    /*Check if Image Already on GPU*/
    if (image_on_gpu_) {

        /*Choose which GPU to run on, change this on a multi-GPU system.*/
        CUDA_CHECK(cudaSetDevice(device_));

        /*Check for Errors*/
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            return false;
        }
        /*Free CUDA*/
        dev_image_.reset();
        bounding_box_.reset();
        if (cudaStatus != cudaSuccess) {
            return false;
        }
        image_on_gpu_ = false;
    }

    /*Initialize Pinned Memory for Slightly Faster Transfer*/
    bounding_box_ = make_default_bounding_box(width, height);

    /*Choose which GPU to run on, change this on a multi-GPU system.*/
        CUDA_CHECK(cudaSetDevice(device_));

    /*Check for Errors*/
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        bounding_box_.reset();
        image_on_gpu_ = false;
    } else {
        /*Initialize Private Host Variables*/
        width_ = width;
        height_ = height;
        dev_image_.reset();

        /*Allocate GPU buffers for image, triangles.*/
        dev_image_ = make_cuda_unique<unsigned char>(width_ * height_);

        /*Check for Errors*/
        cudaStatus = cudaGetLastError();

        if (cudaStatus != cudaSuccess) {
            image_on_gpu_ = false;
            /*Free CUDA*/
            dev_image_.reset();
            bounding_box_.reset();
        } else {
            /*Upload Image from Host to Device*/
            CUDA_CHECK(cudaMemcpy(
                dev_image_.get(),
                host_image,
                width_ * height_ * sizeof(unsigned char),
                cudaMemcpyHostToDevice));

            /*Check for Errors*/
            cudaStatus = cudaGetLastError();

            if (cudaStatus != cudaSuccess) {
                image_on_gpu_ = false;
                /*Free CUDA*/
                dev_image_.reset();
                bounding_box_.reset();
            } else {
                image_on_gpu_ = true;
            }
        }
    }
    return image_on_gpu_;
};

bool GPUImage::RemoveImageFromGPU() {
    /*CUDA Error Status*/
    cudaGetLastError(); // Resets Errors
    cudaError_t cudaStatus;

    /*Check if Image Already on GPU*/
    if (image_on_gpu_) {

        /*Choose which GPU to run on, change this on a multi-GPU system.*/
        CUDA_CHECK(cudaSetDevice(device_));

        /*Check for Errors*/
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            return false;
        }
        /*Free CUDA*/
        dev_image_.reset();
        bounding_box_.reset();
        if (cudaStatus != cudaSuccess) {
            return false;
        }
        image_on_gpu_ = false;
    }
    return true;
};

bool GPUImage::CheckImageOnGPU() {
    return image_on_gpu_;
};

unsigned char* GPUImage::GetDeviceImagePointer() {
    return image_on_gpu_ ? dev_image_.get() : nullptr;
};

int* GPUImage::GetBoundingBox() {
    return image_on_gpu_ ? bounding_box_.get() : nullptr;
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
    auto host_image = static_cast<unsigned char*>(
        malloc(width_ * height_ * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(
        host_image,
        this->GetDeviceImagePointer(),
        width_ * height_ * sizeof(unsigned char),
        cudaMemcpyDeviceToHost));

    /*OpenCV Image Container/Write Function*/
    auto projection_mat =
        cv::Mat(height_, width_, CV_8UC1, host_image); /*Reverse before flip*/
    auto output_mat = cv::Mat(width_, height_, CV_8UC1);
    flip(projection_mat, output_mat, 0);
    bool result = imwrite(file_name, output_mat);

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
} // namespace gpu_cost_function
