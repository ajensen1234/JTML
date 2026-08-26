/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#include "services/segmentation_controller.h"

/*CUDA cache clear (moved verbatim with the torch code)*/
#include <c10/cuda/CUDACachingAllocator.h>

/*compute::segment_image (compute layer)*/
#include "compute/machine_learning_tools.h"

namespace jta {

cv::Mat SegmentationController::SegmentFrame(
    const cv::Mat& original_image,
    bool black_sil_used,
    torch::jit::Module* model,
    unsigned int input_width,
    unsigned int input_height) {
    cv::Mat unpadded = segment_image(
        original_image, black_sil_used, model, input_width, input_height);
    // Explicitly clear CUDA cache to free up GPU memory after processing
    // each image. This is particularly helpful for GPUs with limited VRAM,
    // like the RTX 4070, to prevent out-of-memory errors during sequential
    // image processing.
    c10::cuda::CUDACachingAllocator::emptyCache();
    return unpadded;
}

Point6D SegmentationController::EstimateImplantPose(
    ImplantEstimateContext& ctx,
    const cv::Mat& orig_inverted,
    bool clamp_z_to_principal) {
    return ImplantEstimator::EstimateImplantPose(
        ctx, orig_inverted, clamp_z_to_principal);
}

}  // namespace jta
