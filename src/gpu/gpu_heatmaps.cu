#include "gpu/gpu_heatmaps.cuh"

/*CUDA Error Checking*/
#include "gpu/cuda_check.cuh"

namespace gpu_cost_function {
GPUHeatmap::GPUHeatmap(
    int width,
    int height,
    int gpu_device,
    int num_keypoints,
    unsigned char* host_heatmaps) {
    // Start out assuming initialized incorrectly
    initialized_correctly_ = false;

    /*Cuda Error Status*/
    cudaGetLastError();
    cudaError_t cudaStatus;

    CUDA_CHECK(cudaSetDevice(gpu_device));

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        heatmap_on_gpu_ = false;
        return;
    } else {
        width_ = width;
        height_ = height;
        device_ = gpu_device;
        num_keypoints_ = num_keypoints;

        dev_heatmap_ = make_cuda_unique<unsigned char>(
            width_ * height_ * num_keypoints_);

        heatmap_on_gpu_ = true;

        CUDA_CHECK(cudaMemcpy(
            dev_heatmap_.get(),
            host_heatmaps,
            width_ * height_ * num_keypoints_ * sizeof(unsigned char),
            cudaMemcpyHostToDevice));
        initialized_correctly_ = true;
        heatmap_on_gpu_ = true;

        // Initialize the pointer to the heatmap metric values

        return;
    }
};
GPUHeatmap::~GPUHeatmap() = default;
unsigned char* GPUHeatmap::GetDeviceHeatmapPointer() {
    return dev_heatmap_.get();
};
int GPUHeatmap::GetFrameWidth() {
    return width_;
};
int GPUHeatmap::GetFrameHeight() {
    return height_;
};
int GPUHeatmap::GetNumKeypoints() {
    return num_keypoints_;
};
bool GPUHeatmap::IsInitializedCorrectly() {
    return initialized_correctly_;
};
} // namespace gpu_cost_function
