#include "compute/gpu_heatmaps.cuh"

namespace gpu_cost_function {
GPUHeatmap::GPUHeatmap(
    int width,
    int height,
    int gpu_device,
    int num_keypoints,
    unsigned char* host_heatmaps) {
    // Start out assuming initialized incorrectly
    initialized_correctly_ = false;
    dev_heatmap_ = 0;
    width_ = width;
    height_ = height;
    device_ = gpu_device;
    num_keypoints_ = num_keypoints;

    /*Owner fix (2026-08-12): curvature heatmaps exist only after an ML
     * segmentation (Frame::setCurvatureHeatmaps); a study loaded without
     * segmentation has zero keypoints and a null host buffer — a 0-size
     * cudaMemcpy from nullptr fails on some drivers and aborted the
     * optimizer run. Zero keypoints is a legitimate state: no upload, no
     * device buffer, initialized (the cost functions query
     * GetNumKeypoints() == 0 and no-op).*/
    if (num_keypoints <= 0) {
        heatmap_on_gpu_ = false;
        initialized_correctly_ = true;
        return;
    }

    /*Cuda Error Status*/
    cudaGetLastError();
    cudaError_t cudaStatus;

    cudaSetDevice(gpu_device);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        heatmap_on_gpu_ = false;
        return;
    } else {
        cudaMalloc(
            (void**)&dev_heatmap_,
            width_ * height_ * num_keypoints_ * sizeof(unsigned char));

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            heatmap_on_gpu_ = false;
            cudaFree(dev_heatmap_);
        } else {
            heatmap_on_gpu_ = true;
        }

        cudaMemcpy(
            dev_heatmap_,
            host_heatmaps,
            width_ * height_ * num_keypoints_ * sizeof(unsigned char),
            cudaMemcpyHostToDevice);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
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
GPUHeatmap::~GPUHeatmap() {
    cudaFree(dev_heatmap_);
};
unsigned char* GPUHeatmap::GetDeviceHeatmapPointer() {
    return dev_heatmap_;
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
}  // namespace gpu_cost_function
