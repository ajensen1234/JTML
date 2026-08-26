// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*MlOrchestrator (plan 006 U8 / R12 part, R13, F3; AE4): ONE shared
 * segment/estimate orchestration over SegmentationController /
 * ImplantEstimator for BOTH front-ends. The widgets segment/estimate slots
 * (mainscreen.cpp segmentHelperFunction + the two estimate paths) and
 * MlBridge thin onto it: each view keeps its per-frame loops, .pt loading
 * (torch::jit::load), the GPUModel/ImplantEstimateContext construction, the
 * progress/render interleave and the status surface; the orchestrator owns
 * the per-frame chain — segment op -> copy into the frame's inverted image
 * + the Frame post-processing tail, estimate op -> SavePose -> the seed
 * handoff.
 *
 * The torch/CUDA seams are the INJECTED CALLABLES (SegmentOp / EstimateOp):
 * SegmentFrame / EstimateImplantPose are not stubbable as-is (raw
 * torch::jit::Module* + CUDA compute), so the callable seam is what makes
 * the happy path and the segment/estimate-failure scenarios headless-
 * testable; the .pt-load and real CUDA estimate paths stay manual-visual.
 *
 * Seed handoff: EstimateFrame RETURNS the seed (frame, model, pose). The
 * VIEW wires it to its run controller (U5 setSeedPose) — no
 * services->coordinator references (layering). Widgets wiring note: the
 * estimate slots' SavePose into model_locations_ IS the widgets seed
 * (LaunchOptimizer copies the storage by value); QML calls
 * OptimizerBridge::setSeedPose. Stale-frame/model seed guards live in the
 * run controller (U5 takeSeedForRun — the seed applies only when the run's
 * frame is still the seeded frame and the seeded model is still the primary
 * selection; M10a restore applies).
 *
 * Degradation (AE4): the missing-.pt guards stay in the views (they own the
 * availability flags); the orchestrator's failure statuses cover the
 * runnable path (an op throws -> clear failure status, no frame mutation /
 * no save / no seed, the view surfaces it and the plain-optimize path is
 * untouched).
 *
 * Plain services class: Qt/GPU/torch-free (header-only OpenCV Mat + domain
 * Point6D + std::function); the torch include-after-Qt rule never applies
 * to this header. The Frame post-processing calls (SetEdgeImage /
 * SetDilatedImage / SetDistanceMap / setCurvatureHeatmaps) are the same
 * per-frame tail both views ran (the mono path; the widgets biplane branch
 * skips distance/curvature via full_postprocessing=false).*/

#ifndef ML_ORCHESTRATOR_H
#define ML_ORCHESTRATOR_H

/*Standard*/
#include <functional>

/*OpenCV*/
#include <opencv2/core/mat.hpp>

/*Frame (the caller-owned per-frame data the orchestrator mutates) + pose*/
#include "compute/frame.h"
#include "domain/data_structures_6D.h"

namespace jta {

/*Segment step status. The view maps Ok/not-Ok onto its own status surface.*/
enum class MlSegmentStatus {
    Ok,
    /*The injected segment op threw: no frame mutation, nothing saved.*/
    SegmentFailed,
};

/*Estimate step status.*/
enum class MlEstimateStatus {
    Ok,
    /*The injected estimate op threw: no pose, no SavePose, no seed.*/
    EstimateFailed,
};

/*The seed handoff (R8 — the ML estimate seeds the optimizer). The
 * orchestrator returns it; the VIEW wires it to its run controller
 * (widgets: the storage write IS the seed — LaunchOptimizer copies the
 * storage by value; QML: OptimizerBridge::setSeedPose). Stale guards live
 * in the run controller (U5 takeSeedForRun).*/
struct MlSeed {
    int frame = -1;
    int model = -1;
    Point6D pose;
};

/*Outcome of one estimate step: status + the pose + the seed handoff
 * (seed valid only when status == Ok).*/
struct MlEstimateOutcome {
    MlEstimateStatus status = MlEstimateStatus::EstimateFailed;
    Point6D pose;
    MlSeed seed;
};

class MlOrchestrator {
public:
    /*Injected per-frame segment op. Production wiring: the view's
     * torch::jit::load + SegmentationController::SegmentFrame (the lambda
     * closes over the torch module pointer, black_sil_used and the input
     * dims). Returns the segmented (unpadded) image; throws std::exception
     * to signal failure (the production CUDA path surfaces c10::Error the
     * same way — the orchestrator converts it into a clear status).*/
    using SegmentOp = std::function<cv::Mat(const cv::Mat& original_image)>;

    /*Injected per-frame estimate op. Production wiring: the view's
     * built ImplantEstimateContext + SegmentationController::
     * EstimateImplantPose (clamp_z_to_principal per implant kind). Returns
     * the estimated pose; throws std::exception to signal failure. A zero
     * pose is NOT a failure (it is a valid estimate).*/
    using EstimateOp = std::function<Point6D(const cv::Mat& inverted_image)>;

    /*Per-frame pose save: both views write their own LocationStorage
     * (model_locations_.SavePose(frame, model, pose)). Injected so the
     * headless tests can record calls against a fake storage.*/
    using SavePoseFn =
        std::function<void(int frame, int model, const Point6D& pose)>;

    /*Segment one frame (the segmentHelperFunction / runSegmentOnCurrentFrame
     * per-frame body): run the op, copy the segmented result into the
     * frame's inverted image, then the Frame post-processing tail —
     * SetEdgeImage(use_reverse=true) + SetDilatedImage, and (mono path)
     * SetDistanceMap + setCurvatureHeatmaps. The view keeps the .pt load,
     * the dilation/edge parameter sourcing (widgets: UI spin box/sliders,
     * trunk-manager Dilation; QML: the frame's stored values, its trunk
     * manager), the per-frame loops and the progress/render interleave.*/
    MlSegmentStatus SegmentFrame(
        Frame& frame,
        int aperture,
        int low_threshold,
        int high_threshold,
        int dilation,
        bool full_postprocessing,
        const SegmentOp& segment_op);

    /*Estimate one frame (the estimate-slot per-frame body): run the op,
     * then SavePose(frame_index, model_index, pose) through the injected
     * save seam. Returns the outcome carrying the pose + the seed handoff
     * (seed = {frame_index, model_index, pose} on Ok). On failure nothing
     * is saved and the seed stays invalid — the view surfaces the status
     * and the run is unaffected.*/
    MlEstimateOutcome EstimateFrame(
        int frame_index,
        int model_index,
        const cv::Mat& inverted_image,
        const EstimateOp& estimate_op,
        const SavePoseFn& save_pose);
};

}  // namespace jta

#endif /* ML_ORCHESTRATOR_H */
