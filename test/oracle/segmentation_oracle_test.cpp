// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Tier-2 GPU oracle (plan 004 U8, APPEARANCE-based). NOT in the headless
// default suite — must be run explicitly on a GPU machine under the `oracle`
// label:
//   pixi run cmake --build .build --target jtml_test_segmentation_oracle
//   ctest --test-dir .build -L oracle   (or run .build/bin/jtml_test_segmentation_oracle)
//
// Gates the U8 extraction (SegmentationController + ImplantEstimator): the
// per-frame segmentation + implant-estimate path must still produce poses
// whose rendered silhouettes match the known-good labels via IoU — mirroring
// the jtml.oracle appearance contract (silhouette, never raw pose; the
// two-tier spec + tolerance are documented in golden_oracle.org).
//
// The segmentation + pose-regression .pt torch models are USER-PROVIDED —
// they are NOT in test/golden/ (the repo never ships trained weights; only
// calibration.txt / fem_golden.jts / fem_oracle_captured.jtak / baseline.json
// live there). Set these env vars on the GPU machine to run the real gate:
//   JTML_SEG_PT           segmentation model (torch script, .pt)
//   JTML_FEM_ESTIMATE_PT  femoral pose regression model (torch script, .pt)
// When either is absent the test SKIPS cleanly with a message — it stays
// registered and runnable so the GPU machine can run it as soon as models are
// provided. Structure mirrors jtml.oracle: per-frame pipeline, per-frame
// empirical label correspondence, per-frame IoU gate.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <catch2/catch_test_macros.hpp>

#include "compute/frame.h"
#include "compute/gpu_metrics.cuh"
#include "compute/gpu_model.cuh"
#include "domain/data_structures_6D.h"
#include "services/calibration.h"
#include "services/implant_estimator.h"
#include "services/model.h"
#include "services/segmentation_controller.h"

using gpu_cost_function::GPUImage;
using gpu_cost_function::GPUModel;
using gpu_cost_function::GPUMetrics;
using gpu_cost_function::Pose;

namespace {

const std::string kStudyDir = "example_studies/Kneel_1/";
const std::vector<std::string> kBaseImages = {
    kStudyDir + "1024/2806.tif",
    kStudyDir + "1024/2807.tif",
    kStudyDir + "1024/2808.tif"};
const std::vector<std::string> kLabels = {
    kStudyDir + "Labels/fem/AT_K1_V1_0160_label_fem.tif",
    kStudyDir + "Labels/fem/AT_K1_V1_0170_label_fem.tif",
    kStudyDir + "Labels/fem/AT_K1_V1_0180_label_fem.tif"};
const std::string kFemStl = kStudyDir + "KR_right_7_fem.stl";

const int kWidth = 1024;
const int kHeight = 1024;
const int kDevice = 0;
const unsigned int kInputWidth = 1024;
const unsigned int kInputHeight = 1024;

/*User-provided .pt fixture env vars (absent -> clean skip).*/
const char* kSegPtEnv = "JTML_SEG_PT";
const char* kFemEstimatePtEnv = "JTML_FEM_ESTIMATE_PT";

bool FixtureAvailable(const char* env_var) {
    const char* path = std::getenv(env_var);
    if (path == nullptr || path[0] == '\0') return false;
    std::ifstream f(path);
    return f.good();
}

std::vector<unsigned char> GrayscaleUchar(const std::string& path,
                                          bool flip_vertical = false) {
    cv::Mat rgb = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (rgb.empty()) throw std::runtime_error("could not read image: " + path);
    if (rgb.cols != kWidth || rgb.rows != kHeight)
        throw std::runtime_error("unexpected image size for " + path);
    // The binary label TIFFs are stored with a bottom-left y-origin while the
    // GPU renderer outputs top-left origin; flip the label vertically so both
    // are in the same image frame (same convention as jtml.oracle).
    if (flip_vertical) cv::flip(rgb, rgb, 0);
    std::vector<unsigned char> buf((size_t)kWidth * kHeight);
    for (int y = 0; y < kHeight; ++y) {
        const unsigned char* row = rgb.ptr<unsigned char>(y);
        std::copy(row, row + kWidth, buf.begin() + (size_t)y * kWidth);
    }
    return buf;
}

}  // namespace

TEST_CASE("Tier-2 GPU oracle: segmentation + implant estimate path",
          "[oracle][gpu]") {
    /*Clean skip when the user-provided .pt fixtures are absent (nothing in
     * test/golden/ is a torch model): the test stays registered and passes
     * with a message until the GPU machine provides JTML_SEG_PT +
     * JTML_FEM_ESTIMATE_PT.*/
    if (!FixtureAvailable(kSegPtEnv) || !FixtureAvailable(kFemEstimatePtEnv)) {
        std::cout
            << "[oracle] SKIPPED: segmentation/estimate .pt fixtures absent -- "
               "set JTML_SEG_PT and JTML_FEM_ESTIMATE_PT (torch script "
               "models) to run the U8 segmentation+estimate gate on the GPU "
               "machine."
            << std::endl;
        return;
    }

    /*Load the user-provided torch models onto CUDA (torch::jit::load, as the
     * slots do -- the view keeps the loading, the test mirrors the view).*/
    torch::jit::Module seg_module(
        torch::jit::load(std::getenv(kSegPtEnv), torch::kCUDA));
    torch::jit::Module* seg_model = &seg_module;
    torch::jit::Module est_module(
        torch::jit::load(std::getenv(kFemEstimatePtEnv), torch::kCUDA));
    torch::jit::Module* est_model = &est_module;

    /*Calibration for the estimate math (matches the golden capture:
     * principal distance 1198, pixel pitch 0.373 -- same constants as
     * jtml.oracle's BuildFramePipeline).*/
    CameraCalibration cam(1198.0f, -1.0f * 0.0f, -1.0f * 0.0f, 0.373f);
    Calibration calib(cam);

    /*Segmentation: per-frame SegmentFrame + the Frame post-processing the
     * segmentHelperFunction loop applied (edge/dilated/distance/heatmaps).
     * The controller API is per-frame: the test owns the loop, mirroring the
     * view's ownership of the loop + progress/render interleave (U8 per-frame
     * API -- the controller never owns a loop).*/
    std::vector<Frame> frames;
    frames.reserve(kBaseImages.size());
    for (const auto& path : kBaseImages) {
        frames.emplace_back(path, 3, 0, 150, /*dilation=*/6);
    }
    jta::SegmentationController controller;
    for (auto& frame : frames) {
        cv::Mat segmented = controller.SegmentFrame(
            frame.GetOriginalImage(), /*black_sil_used=*/false, seg_model,
            kInputWidth, kInputHeight);
        REQUIRE(!segmented.empty());
        REQUIRE(segmented.rows == kHeight);
        REQUIRE(segmented.cols == kWidth);
        segmented.copyTo(frame.GetInvertedImage());
        frame.SetEdgeImage(3, 0, 150, /*use_reverse=*/true);
        frame.SetDilatedImage(6);
        frame.SetDistanceMap();
        frame.setCurvatureHeatmaps();
        /*A no-op model would leave the inverted image untouched: the
         * segmentation must actually change the frame.*/
        cv::Mat changed;
        cv::absdiff(frame.GetInvertedImage(), frame.GetOriginalImage(),
                    changed);
        REQUIRE(cv::sum(changed)[0] > 0);
    }

    /*Estimate: build the per-frame context exactly like the estimate slots
     * (GPU model from the STL, torch pose model, scratch buffers,
     * calibration), then run the per-frame estimate op per frame. Femoral
     * path: clamp_z_to_principal = true (the slot's z clamp).*/
    Model femur(kFemStl, "femur", "femur");
    REQUIRE(femur.initialized_correctly_);
    int triangle_count = static_cast<int>(femur.triangle_vertices_.size() / 9);
    REQUIRE(triangle_count > 0);
    GPUModel gpu_mod("femur", /*principal=*/true, kWidth, kHeight, kDevice,
                     /*use_backface_culling=*/false,
                     &femur.triangle_vertices_[0], &femur.triangle_normals_[0],
                     triangle_count, calib.camera_A_principal_);
    REQUIRE(gpu_mod.IsInitializedCorrectly());

    jta::ImplantEstimateContext ctx;
    ctx.gpu_mod = &gpu_mod;
    ctx.model = est_model;
    ctx.host_image = static_cast<unsigned char*>(
        malloc(kInputWidth * kInputHeight * sizeof(unsigned char)));
    REQUIRE(ctx.host_image != nullptr);
    ctx.orientation = new float[3];
    ctx.gpu_byte_placeholder = torch::zeros(
        {1, 1, kInputHeight, kInputWidth},
        device(torch::kCUDA).dtype(torch::kByte));
    ctx.calibration = calib;
    ctx.input_width = kInputWidth;
    ctx.input_height = kInputHeight;
    ctx.orig_width = kWidth;
    ctx.orig_height = kHeight;

    std::vector<Point6D> estimated;
    estimated.reserve(frames.size());
    for (int i = 0; i < (int)frames.size(); ++i) {
        Point6D pose = controller.EstimateImplantPose(
            ctx, frames[i].GetInvertedImage(), /*clamp_z_to_principal=*/true);
        estimated.push_back(pose);
        std::cout << "[oracle] frame " << i << " estimated pose: (" << pose.x
                  << ", " << pose.y << ", " << pose.z << ", " << pose.xa
                  << ", " << pose.ya << ", " << pose.za << ")" << std::endl;
    }
    free(ctx.host_image);
    delete[] ctx.orientation;

    /*Appearance gate (mirrors jtml.oracle): render the implant at each
     * estimated pose and compare the silhouette to the known-good labels via
     * IoU. Label correspondence is resolved empirically (the label names are
     * not frame-aligned -- see oracle_test.cpp).*/
    std::vector<std::vector<unsigned char>> label_bufs;
    std::vector<GPUImage*> label_gpus;
    for (const auto& path : kLabels) {
        label_bufs.push_back(GrayscaleUchar(path, /*flip_vertical=*/true));
        label_gpus.push_back(
            new GPUImage(kWidth, kHeight, kDevice, label_bufs.back().data()));
        REQUIRE(label_gpus.back()->IsInitializedCorrectly());
    }
    GPUMetrics metrics;
    REQUIRE(metrics.IsInitializedCorrectly());

    // Same appearance tolerance as jtml.oracle (documented in
    // golden_oracle.org's two-tier spec). Re-confirm the exact threshold on
    // the first fixture run: the estimate path's accuracy depends on the
    // user-provided .pt models, which the repo cannot validate in advance.
    const double kIouGate = 0.85;
    for (int f = 0; f < (int)estimated.size(); ++f) {
        Pose render_pose(estimated[f].x, estimated[f].y, estimated[f].z,
                         estimated[f].xa, estimated[f].ya, estimated[f].za);
        gpu_mod.SetCurrentPrimaryCameraPose(render_pose);
        REQUIRE(gpu_mod.RenderPrimaryCamera(render_pose));
        GPUImage* render = gpu_mod.GetPrimaryCameraRenderedImage();
        int best_label = 0;
        double best_iou = -1.0;
        for (int i = 0; i < (int)label_gpus.size(); ++i) {
            double v = metrics.IOU(render, label_gpus[i]);
            std::cout << "[oracle] frame " << f << " IoU vs label[" << i
                      << "] = " << v << std::endl;
            if (v > best_iou) {
                best_iou = v;
                best_label = i;
            }
        }
        std::cout << "[oracle] frame " << f << " best label index = "
                  << best_label << " (" << kLabels[best_label] << ")"
                  << std::endl;
        CAPTURE(f, best_iou);
        std::cout << "[oracle] frame " << f << " appearance gate: best IoU "
                  << best_iou << " vs threshold " << kIouGate << std::endl;
        REQUIRE(best_iou > kIouGate);
    }
    for (auto* g : label_gpus) delete g;

    /*Edge case (R12 test scenario): reset-segmentation restores the
     * pre-segmentation frame state. Frame's reset contract: the inverted
     * image returns to 255 - original (see Frame::ResetFromOriginal).*/
    cv::Mat restored = 255 - frames[0].GetOriginalImage();
    frames[0].ResetFromOriginal();
    cv::Mat reset_diff;
    cv::absdiff(frames[0].GetInvertedImage(), restored, reset_diff);
    REQUIRE(cv::sum(reset_diff)[0] == 0);
}
