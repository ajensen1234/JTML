// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U8: MlOrchestrator pins (R12 part, R13, F3; AE4). Deterministic
// Catch2 headless tests that direct-compile the orchestrator (plain
// C++/OpenCV — no Qt, no torch, no GPU) against the headless Frame twin:
// the injected SegmentOp / EstimateOp / SavePoseFn seams are the torch/CUDA
// boundaries, so the happy path and the failure scenarios are headless-
// testable (the .pt-load and real CUDA estimate paths stay manual-visual).
//
// Pins (plan 006 U8 test scenarios a-d, headless subset):
//  - (a) happy path: segment -> estimate -> save -> seed, driven through
//    stubbed callables — the segment op receives the frame's ORIGINAL
//    image, the segmented result lands in the inverted image + the
//    post-processing tail runs with the passed parameters, the estimate op
//    receives the inverted image, the pose is saved exactly once at
//    (frame, model), and the returned seed carries (frame, model, pose)
//    for the view's run-controller wiring;
//  - (d) segment failure: the op throws -> SegmentFailed, the frame is
//    untouched (no copy, no post-processing, no threshold writes); the
//    estimate failure path: the op throws -> EstimateFailed, no save, the
//    seed stays invalid — the view surfaces the status and the
//    plain-optimize path is unaffected;
//  - (b) degradation: the missing-.pt guards live in the VIEWS (the
//    orchestrator is never invoked without models — pinned by the existing
//    jtml.experimental_ml_bridge degradation suite, kept green);
//  - (c) stale seed: the frame/model-change drop between estimate and run
//    is the run controller's takeSeedForRun guard (U5) — pinned in
//    optimizer_run_controller_core_test.cpp ("seed lifecycle (one-shot,
//    stale guards, clear)") and experimental_ml_bridge_test.cpp ("a
//    stale-frame seed never overrides another frame"). The orchestrator's
//    contract here: the seed returns the estimated (frame, model) so the
//    wiring cannot re-target it.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>
#include <tuple>
#include <vector>

#include "compute/frame.h"
#include "services/ml_orchestrator.h"

using Catch::Approx;

namespace {

/*Kneel_1 fixture frame (repo-root WORKING_DIRECTORY, like the other
 * headless suites): the headless Frame twin reads the .tif via OpenCV.*/
const char* kFrameImage = "example_studies/Kneel_1/AT_K1_V1_0160.tif";

Frame MakeFrame() {
    Frame frame(kFrameImage, /*aperture=*/3, /*low=*/30, /*high=*/100,
                /*dilation=*/4);
    REQUIRE(!frame.GetOriginalImage().empty());
    return frame;
}

/*A stub segment op: a uniform 200-valued Mat the size of the input — a
 * recognizable "segmented" result (its Canny is empty, which also proves
 * the post-processing tail re-ran on the NEW inverted image).*/
cv::Mat StubSegment(const cv::Mat& original) {
    return cv::Mat(original.size(), CV_8UC1, cv::Scalar(200));
}

/*The injected save seam: records (frame, model, pose) calls.*/
struct PoseRecorder {
    std::vector<std::tuple<int, int, Point6D>> calls;

    void operator()(int frame, int model, const Point6D& pose) {
        calls.emplace_back(frame, model, pose);
    }
};

bool MatsEqual(const cv::Mat& a, const cv::Mat& b) {
    return a.size() == b.size() && cv::countNonZero(a != b) == 0;
}

}  // namespace

TEST_CASE("ml_orchestrator: segment happy path — op input, inverted copy, "
          "post-processing tail",
          "[ml_orchestrator]") {
    jta::MlOrchestrator orch;
    Frame frame = MakeFrame();
    /*Deep copy: GetDistanceMap returns a shared-buffer Mat — the "before"
     * snapshot must not alias the live buffer SetDistanceMap writes.*/
    const cv::Mat distance_before = frame.GetDistanceMap().clone();

    bool op_called = false;
    cv::Mat original_seen;
    const jta::MlSegmentStatus status = orch.SegmentFrame(
        frame,
        /*aperture=*/5, /*low=*/40, /*high=*/120, /*dilation=*/2,
        /*full_postprocessing=*/true,
        [&](const cv::Mat& original) {
            op_called = true;
            original_seen = original;
            return StubSegment(original);
        });

    REQUIRE(status == jta::MlSegmentStatus::Ok);
    REQUIRE(op_called);
    /*The op received the frame's ORIGINAL image (the GPU segment input).*/
    REQUIRE(MatsEqual(original_seen, frame.GetOriginalImage()));
    /*The segmented result replaced the inverted image (copyTo tail).*/
    REQUIRE(MatsEqual(frame.GetInvertedImage(), StubSegment(frame.GetOriginalImage())));
    /*The post-processing tail ran with the PASSED parameters: SetEdgeImage
     * stored the new thresholds and re-ran Canny on the new inverted image
     * (a uniform 200 fill has no edges — the pre-segment edge was Canny of
     * the x-ray, non-empty); SetDilatedImage/SetDistanceMap/
     * setCurvatureHeatmaps completed without error.*/
    REQUIRE(frame.GetAperture() == 5);
    REQUIRE(frame.GetLowThreshold() == 40);
    REQUIRE(frame.GetHighThreshold() == 120);
    REQUIRE(cv::sum(frame.GetEdgeImage())[0] == 0);
    REQUIRE(!MatsEqual(frame.GetDistanceMap(), distance_before));
}

TEST_CASE("ml_orchestrator: biplane tail — edge + dilation only, distance "
          "map kept",
          "[ml_orchestrator]") {
    /*The widgets camera-B branch (full_postprocessing=false): the shared
     * op + inverted copy + edge/dilated run, but the mono distance-map +
     * curvature tail is skipped — the pre-segment distance map survives.*/
    jta::MlOrchestrator orch;
    Frame frame = MakeFrame();
    /*Deep copy (the snapshot must not alias the live buffer).*/
    const cv::Mat distance_before = frame.GetDistanceMap().clone();

    const jta::MlSegmentStatus status = orch.SegmentFrame(
        frame,
        /*aperture=*/5, /*low=*/40, /*high=*/120, /*dilation=*/2,
        /*full_postprocessing=*/false,
        [&](const cv::Mat& original) { return StubSegment(original); });

    REQUIRE(status == jta::MlSegmentStatus::Ok);
    REQUIRE(MatsEqual(frame.GetInvertedImage(), StubSegment(frame.GetOriginalImage())));
    REQUIRE(frame.GetAperture() == 5);
    REQUIRE(MatsEqual(frame.GetDistanceMap(), distance_before));
}

TEST_CASE("ml_orchestrator: segment failure — status surfaced, frame "
          "untouched",
          "[ml_orchestrator]") {
    /*Scenario (d): the injected op throws (the production CUDA path
     * surfaces c10::Error the same way) -> SegmentFailed, no frame
     * mutation of any kind: no inverted copy, no edge re-run, no threshold
     * writes. The view surfaces the status and the plain-optimize path is
     * unaffected.*/
    jta::MlOrchestrator orch;
    Frame frame = MakeFrame();
    const cv::Mat inverted_before = frame.GetInvertedImage();
    const cv::Mat edge_before = frame.GetEdgeImage();
    const int aperture_before = frame.GetAperture();

    const jta::MlSegmentStatus status = orch.SegmentFrame(
        frame,
        /*aperture=*/5, /*low=*/40, /*high=*/120, /*dilation=*/2,
        /*full_postprocessing=*/true,
        [](const cv::Mat&) -> cv::Mat {
            throw std::runtime_error("segment failed");
        });

    REQUIRE(status == jta::MlSegmentStatus::SegmentFailed);
    REQUIRE(MatsEqual(frame.GetInvertedImage(), inverted_before));
    REQUIRE(MatsEqual(frame.GetEdgeImage(), edge_before));
    REQUIRE(frame.GetAperture() == aperture_before);
}

TEST_CASE("ml_orchestrator: segment empty result — no tail, status "
          "surfaced",
          "[ml_orchestrator]") {
    /*Owner feedback 2026-08-11: a segmentation that produced NO contours
     * (empty Mat) must not run the post-processing tail — the GPU
     * curvature heatmaps only exist when the ML contours exist. Status
     * surfaced, frame untouched (no inverted copy, no edge re-run).*/
    jta::MlOrchestrator orch;
    Frame frame = MakeFrame();
    const cv::Mat inverted_before = frame.GetInvertedImage();
    const cv::Mat edge_before = frame.GetEdgeImage();

    const jta::MlSegmentStatus status = orch.SegmentFrame(
        frame,
        /*aperture=*/5, /*low=*/40, /*high=*/120, /*dilation=*/2,
        /*full_postprocessing=*/true,
        [](const cv::Mat&) -> cv::Mat { return cv::Mat(); });

    REQUIRE(status == jta::MlSegmentStatus::SegmentFailed);
    REQUIRE(MatsEqual(frame.GetInvertedImage(), inverted_before));
    REQUIRE(MatsEqual(frame.GetEdgeImage(), edge_before));
}

TEST_CASE("ml_orchestrator: estimate happy path — op input, save once, seed "
          "returned",
          "[ml_orchestrator]") {
    /*Scenario (a): the estimate op runs on the frame's inverted image, the
     * pose is saved exactly once at (frame, model) through the injected
     * seam (both views' LocationStorage::SavePose), and the RETURNED seed
     * carries (frame, model, pose) — the view wires it to its run
     * controller (widgets: the storage write IS the seed; QML:
     * setSeedPose).*/
    jta::MlOrchestrator orch;
    PoseRecorder recorder;
    const Point6D expected_pose(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

    const cv::Mat inverted = MakeFrame().GetInvertedImage();
    bool op_called = false;
    cv::Mat inverted_seen;
    const jta::MlEstimateOutcome outcome = orch.EstimateFrame(
        /*frame_index=*/2, /*model_index=*/1, inverted,
        [&](const cv::Mat& inv) {
            op_called = true;
            inverted_seen = inv;
            return expected_pose;
        },
        std::ref(recorder));

    REQUIRE(outcome.status == jta::MlEstimateStatus::Ok);
    REQUIRE(op_called);
    REQUIRE(MatsEqual(inverted_seen, inverted));

    REQUIRE(recorder.calls.size() == 1);
    REQUIRE(std::get<0>(recorder.calls[0]) == 2);
    REQUIRE(std::get<1>(recorder.calls[0]) == 1);
    const Point6D saved = std::get<2>(recorder.calls[0]);
    REQUIRE(saved.x == Approx(1.0));
    REQUIRE(saved.y == Approx(2.0));
    REQUIRE(saved.z == Approx(3.0));
    REQUIRE(saved.xa == Approx(4.0));
    REQUIRE(saved.ya == Approx(5.0));
    REQUIRE(saved.za == Approx(6.0));

    /*The seed handoff carries the estimated frame/model — the stale drop
     * on a later frame/model change is the run controller's guard (U5).*/
    REQUIRE(outcome.seed.frame == 2);
    REQUIRE(outcome.seed.model == 1);
    REQUIRE(outcome.seed.pose.x == Approx(1.0));
    REQUIRE(outcome.seed.pose.za == Approx(6.0));
}

TEST_CASE("ml_orchestrator: estimate failure — no save, no seed",
          "[ml_orchestrator]") {
    /*Scenario (d): the estimate op throws -> EstimateFailed; the pose is
     * NOT saved and the seed stays invalid — the view surfaces the status,
     * the estimate display stays empty and the plain-optimize path is
     * unaffected (no partial state).*/
    jta::MlOrchestrator orch;
    PoseRecorder recorder;

    const jta::MlEstimateOutcome outcome = orch.EstimateFrame(
        /*frame_index=*/0, /*model_index=*/0, MakeFrame().GetInvertedImage(),
        [](const cv::Mat&) -> Point6D {
            throw std::runtime_error("estimate failed");
        },
        std::ref(recorder));

    REQUIRE(outcome.status == jta::MlEstimateStatus::EstimateFailed);
    REQUIRE(recorder.calls.empty());
    REQUIRE(outcome.seed.frame == -1);
    REQUIRE(outcome.seed.model == -1);
}

TEST_CASE("ml_orchestrator: a zero pose is a valid estimate, not a failure",
          "[ml_orchestrator]") {
    /*Contract pin: the estimate op returns a Point6D by value — a zero
     * pose is a legitimate estimate (failure is signaled by throwing, not
     * by the pose values).*/
    jta::MlOrchestrator orch;
    PoseRecorder recorder;

    const jta::MlEstimateOutcome outcome = orch.EstimateFrame(
        /*frame_index=*/0, /*model_index=*/0, MakeFrame().GetInvertedImage(),
        [](const cv::Mat&) { return Point6D(); },
        std::ref(recorder));

    REQUIRE(outcome.status == jta::MlEstimateStatus::Ok);
    REQUIRE(recorder.calls.size() == 1);
    REQUIRE(outcome.seed.frame == 0);
    REQUIRE(outcome.seed.model == 0);
}
