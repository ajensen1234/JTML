#include "gpu/gpu_heatmaps.cuh"

namespace gpu_cost_function{
    GPUHeatmap::GPUHeatmap(int width, int height,
                           int gpu_device, int num_keypoints,
                           unsigned char* host_heatmaps){
        // Start out assuming initialized incorrectly
        initialized_correctly_ = false;

        /*Cuda Error Status*/
        cudaGetLastError();
        cudaError_t cudaStatus;

        cudaSetDevice(gpu_device);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess){
            heatmap_on_gpu_ = false;
            return;
        }
        else {
            width_ = width;
            height_ = height;
            device_ = gpu_device;
            num_keypoints_ = num_keypoints;
            dev_heatmap_ = 0;

            cudaMalloc((void**)&dev_heatmap_, width_ * height_* num_keypoints_ * sizeof(unsigned char));

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess){
                heatmap_on_gpu_ = false;
                cudaFree(dev_heatmap_);
            } else {
                heatmap_on_gpu_ = true;
            }

            cudaMemcpy(dev_heatmap_, host_heatmaps, width_ * height_ * num_keypoints_ * sizeof(unsigned char),
                       cudaMemcpyHostToDevice);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess){
                initialized_correctly_ = false;
                heatmap_on_gpu_ = false;
                cudaFree(dev_heatmap_);
                return;
            }
            initialized_correctly_ = true;
            heatmap_on_gpu_ = true;

            // Initialize the pointer to the heatmap metric values

            return;
        }

    };
    GPUHeatmap::~GPUHeatmap(){
        cudaFree(dev_heatmap_);
    };
    unsigned char* GPUHeatmap::GetDeviceHeatmapPointer(){
        return dev_heatmap_;
    };
    int GPUHeatmap::GetFrameWidth(){
        return width_;
    };
    int GPUHeatmap::GetFrameHeight(){
        return height_;
    };
    int GPUHeatmap::GetNumKeypoints(){
        return num_keypoints_;
    };
    bool GPUHeatmap::IsInitializedCorrectly(){
        return initialized_correctly_;
    };
}
