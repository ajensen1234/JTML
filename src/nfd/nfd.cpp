// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "nfd/nfd.h"

JTML_NFD::JTML_NFD() {}

JTML_NFD::~JTML_NFD() {}

bool JTML_NFD::Initialize(Calibration cal_file, std::vector<Model> model_list,
                          std::vector<Frame> frames_list,
                          QModelIndexList selected_models,
                          unsigned int primary_model_index,
                          QString error_message) {
    calibration_ = cal_file;

    /*Cuda Initialization*/
    int cuda_device_id = 0, gpu_device_count = 0, device_count;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&device_count);
    if (cudaResultCode != cudaSuccess) device_count = 0;
    /* Machines with no GPUs can still report one emulation device */
    for (int device = 0; device < device_count; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999 &&
            properties.major >= 5) /* 9999 means emulation only */
            ++gpu_device_count;
    }
    /*If no Cuda Compatitble Devices with Compute Capability Greater Than 5,
     * Exit*/
    if (gpu_device_count == 0) {
        error_message =
            "No Cuda Compatitble Devices with Compute Capability Greater Than "
            "5!";
        successful_initialization_ = false;
        return successful_initialization_;
    }

    all_models_ = model_list;
    frames_ = frames_list;
    primary_model_ = all_models_[primary_model_index];

    int width = frames_[0].GetEdgeImage().cols;
    int height = frames_[0].GetEdgeImage().rows;

    /*Upload GPU Models*/
    gpu_principal_model_ = new GPUModel(
        primary_model_.model_name_, true, width, height, cuda_device_id, true,
        &primary_model_.triangle_vertices_[0],
        &primary_model_.triangle_normals_[0],
        primary_model_.triangle_vertices_.size() / 9,
        calibration_.camera_A_principal_);
    if (!gpu_principal_model_->IsInitializedCorrectly()) {
        delete gpu_principal_model_;
        gpu_principal_model_ = 0;
        error_message = "Error uploading principal model to GPU!";
        successful_initialization_ = false;
        return successful_initialization_;
    }
}

void JTML_NFD::Run() {
    nfd_library lib(calibration_, *gpu_principal_model_, 30, 30, 3, 3);
}
