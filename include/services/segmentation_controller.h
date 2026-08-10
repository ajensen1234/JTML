/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*SegmentationController (plan 004 U8 / R12): the per-frame segment +
 * implant-estimate operations of the MainScreen segment/estimate block,
 * previously inline in segmentHelperFunction and the two estimate slots.
 *
 * Per-frame controller API (the load-bearing U8 decision): the controller
 * NEVER owns a loop. The view owns the per-frame loops, the progress
 * (ui.pose_progress / ui.pose_label), qApp->processEvents() and the qvtk
 * update/render calls -- the ~44 progress/render interleavings stay
 * byte-identical in the slots. The controller exposes one segment op
 * (SegmentFrame) and one estimate op (EstimateImplantPose, delegating to
 * the ImplantEstimator) per frame. The view keeps the file dialogs (the
 * nested on_actionSegment_FemHR_triggered() call and the pose .pt dialog)
 * and the torch model loading (torch::jit::load stays in the slots).
 *
 * GPU/black_sil_used/CUDACachingAllocator calls moved with the torch code
 * verbatim: SegmentFrame wraps compute::segment_image + the CUDA cache
 * clear exactly as segmentHelperFunction did per frame (mono and biplane
 * alike). Link consequence: jtml_services is GPU-linked as of U8 (torch +
 * jtml_compute on its link line -- see src/services/CMakeLists.txt);
 * headless consumers are unaffected (tests compile layer .cpps directly).*/

#ifndef SEGMENTATION_CONTROLLER_H
#define SEGMENTATION_CONTROLLER_H

/*OpenCV*/
#include <opencv2/core/mat.hpp>

/*Torch model pointer*/
#include <torch/script.h>

/*Pose type + the per-frame estimate context*/
#include "domain/data_structures_6D.h"
#include "services/implant_estimator.h"

namespace jta {

class SegmentationController {
public:
    /*Per-frame segment: wraps compute::segment_image + the CUDA cache clear
     * verbatim (the two calls segmentHelperFunction made per frame, mono and
     * biplane alike). Returns the segmented image; the view copies it into
     * the frame's inverted image and runs the Frame post-processing
     * (edge/dilated/distance-map/curvature heatmaps).*/
    cv::Mat SegmentFrame(
        const cv::Mat& original_image,
        bool black_sil_used,
        torch::jit::Module* model,
        unsigned int input_width,
        unsigned int input_height);

    /*Per-frame estimate: the estimate math for one inverted image
     * (delegates to the compute-side math in jta::ImplantEstimator).
     * clamp_z_to_principal preserves the femoral slot's principal-distance
     * z clamp (true) vs the tibial slot's direct z (false) byte-identically.*/
    Point6D EstimateImplantPose(
        ImplantEstimateContext& ctx,
        const cv::Mat& orig_inverted,
        bool clamp_z_to_principal);
};

}  // namespace jta

#endif /* SEGMENTATION_CONTROLLER_H */
