// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*MlOrchestrator implementation (plan 006 U8): the per-frame
 * segment -> estimate -> SavePose -> seed chain, relocated verbatim from
 * MainScreen::segmentHelperFunction + the two estimate slots and
 * MlBridge::runSegmentOnCurrentFrame / estimateCurrentFrame. The torch/CUDA
 * calls stay behind the injected SegmentOp/EstimateOp seams (the views own
 * the .pt loading + the GPUModel/context construction); this TU is plain
 * C++/OpenCV (no torch include — the header stays torch-free).*/

#include "services/ml_orchestrator.h"

#include <exception>

namespace jta {

MlSegmentStatus MlOrchestrator::SegmentFrame(
    Frame& frame,
    int aperture,
    int low_threshold,
    int high_threshold,
    int dilation,
    bool full_postprocessing,
    const SegmentOp& segment_op) {
    /*The injected op (torch load + SegmentFrame — the per-frame segment
     * call both views made); a throw is the failure contract (the
     * production CUDA path surfaces c10::Error the same way).*/
    cv::Mat segmented;
    try {
        segmented = segment_op(frame.GetOriginalImage());
    } catch (const std::exception&) {
        return MlSegmentStatus::SegmentFailed;
    }

    /*Empty-result guard (owner feedback 2026-08-11): a segmentation that
     * produced NO contours must not run the post-processing tail — the
     * GPU curvature heatmaps only exist when the ML contours exist, and
     * edge/dilation/distance on an empty Mat is garbage-in-garbage-out.
     * The Frame methods are additionally self-guarded (setCurvatureHeatmaps
     * no-ops on an empty inverted image), so this is the semantic gate,
     * not the only crash fence.*/
    if (segmented.empty()) {
        return MlSegmentStatus::SegmentFailed;
    }

    /*The per-frame tail (segmentHelperFunction / runSegmentOnCurrentFrame
     * verbatim): the segmented result replaces the inverted image, then
     * the Frame post-processing. The widgets mono path + the QML bridge run
     * the full tail (distance map + curvature heatmaps); the widgets
     * biplane branch keeps edge + dilation only
     * (full_postprocessing=false).*/
    segmented.copyTo(frame.GetInvertedImage());
    frame.SetEdgeImage(aperture, low_threshold, high_threshold,
                       /*use_reverse=*/true);
    frame.SetDilatedImage(dilation);
    if (full_postprocessing) {
        frame.SetDistanceMap();
        frame.setCurvatureHeatmaps();
    }
    return MlSegmentStatus::Ok;
}

MlEstimateOutcome MlOrchestrator::EstimateFrame(
    int frame_index,
    int model_index,
    const cv::Mat& inverted_image,
    const EstimateOp& estimate_op,
    const SavePoseFn& save_pose) {
    MlEstimateOutcome outcome;
    try {
        outcome.pose = estimate_op(inverted_image);
    } catch (const std::exception&) {
        /*No pose, no save, no seed — the view surfaces the status and the
         * plain-optimize path is unaffected.*/
        return outcome;
    }
    outcome.status = MlEstimateStatus::Ok;
    if (save_pose) {
        save_pose(frame_index, model_index, outcome.pose);
    }
    /*Seed handoff (R8): returned to the view, which wires it to its run
     * controller (widgets: the storage write above IS the seed; QML:
     * OptimizerBridge::setSeedPose). Stale-frame/model guards live in the
     * run controller (U5 takeSeedForRun, M10a restore applies).*/
    outcome.seed.frame = frame_index;
    outcome.seed.model = model_index;
    outcome.seed.pose = outcome.pose;
    return outcome;
}

}  // namespace jta
