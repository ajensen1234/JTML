// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Tier-2 GPU oracle (plan U6, APPEARANCE-based). NOT in the headless default
// suite — must be run explicitly on a GPU machine under the `oracle` label:
//   pixi run cmake --build .build --target jtml_test_oracle
//   ctest --test-dir .build -L oracle   (or run .build/bin/jtml_test_oracle)
//
// The load-bearing gate is a SILHOUETTE comparison, NOT raw pose values:
// DIRECT numeric convergence is noisy, so we render the femur at the optimized
// pose and compare the rendered silhouette to the known-good binary label via
// IOU. This is a behavior-preservation gate: it catches drift in the rewire
// that bound the extracted DirectOptimizer to the real GPU DIRECT_DILATION
// cost.
//
// Fixture mapping (verified empirically on the RTX 3090): fem.jts poses were
// captured against the 1024/2806-2808.tif base frames (baseline.json:
// base_images = 1024/*.tif). The labels live in Labels/fem/ with the names
// AT_K1_V1_0160/0170/0180_label_fem.tif — the label that visually corresponds
// to a given 1024 frame is NOT name-aligned, so we resolve it by rendering the
// femur at the fem.jts start pose and picking the label with the highest IOU
// (diagnostic), then use that same label for the optimized-pose gate.

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/data_structures_6D.h"
#include "core/direct_optimizer.h"
#include "core/frame.h"
#include "core/model.h"
#include "core/calibration.h"

#include "cost_functions/CostFunctionManager.h"

#include "gpu/camera_calibration.h"
#include "gpu/gpu_metrics.cuh"
#include "gpu/gpu_model.cuh"
#include "gpu/pose_matrix.h"

using gpu_cost_function::Pose;
using gpu_cost_function::GPUEdgeFrame;
using gpu_cost_function::GPUDilatedFrame;
using gpu_cost_function::GPUIntensityFrame;
using gpu_cost_function::GPUFrame;
using gpu_cost_function::GPUHeatmap;
using gpu_cost_function::GPUImage;
using gpu_cost_function::GPUModel;
using gpu_cost_function::GPUMetrics;

namespace {

const std::string kStudyDir = "example_studies/Kneel_1/";
// fem.jts poses were captured against these frames.
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

// fem.jts per-frame start poses. Point6D order is (x,y,z, x_rot, y_rot,
// z_rot) = (x_tran, y_tran, z_tran, x_rot, y_rot, z_rot) from baseline.json
// expected_pose_per_frame (Table 1 of the golden spec).
std::vector<Point6D> StartPoses() {
    return {
        Point6D(18.52191, 19.69514, -1027.713, -7.419319, -0.2587041,
                -26.69708),  // frame 0 (2806.tif)
        Point6D(19.01747, 20.15555, -1026.732, -7.56846, -0.2358893,
                -27.37827),  // frame 1 (2807.tif)
        Point6D(16.4709, 16.4248, -1028.69, -7.678545, 0.3264978,
                -24.12223),  // frame 2 (2808.tif)
    };
}

Point6D SearchRange() {
    return Point6D(12.0, 12.0, 15.0, 15.0, 15.0, 15.0);
}

std::vector<unsigned char> GrayscaleUchar(const std::string& path,
                                          bool flip_vertical = false) {
    cv::Mat rgb = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (rgb.empty()) throw std::runtime_error("could not read image: " + path);
    if (rgb.cols != kWidth || rgb.rows != kHeight)
        throw std::runtime_error("unexpected image size for " + path);
    // The binary label TIFFs are stored with a bottom-left y-origin while the
    // GPU renderer outputs top-left origin; flip the label vertically so both
    // are in the same image frame (verified: fem.jts render vs flipped label[0]
    // has IoU == 1.0).
    if (flip_vertical) cv::flip(rgb, rgb, 0);
    std::vector<unsigned char> buf((size_t)kWidth * kHeight);
    for (int y = 0; y < kHeight; ++y) {
        const unsigned char* row = rgb.ptr<unsigned char>(y);
        std::copy(row, row + kWidth, buf.begin() + (size_t)y * kWidth);
    }
    return buf;
}

// Flatten a single-channel grayscale cv::Mat into a contiguous uchar buffer
// (used for GPU frame uploads from the Frame's processed images).
std::vector<unsigned char> MatToUchar(const cv::Mat& m) {
    std::vector<unsigned char> buf((size_t)m.rows * m.cols);
    for (int y = 0; y < m.rows; ++y) {
        const unsigned char* row = m.ptr<unsigned char>(y);
        std::copy(row, row + m.cols, buf.begin() + (size_t)y * m.cols);
    }
    return buf;
}

struct Pipeline {
    GPUModel* model = nullptr;
    GPUMetrics* metrics = nullptr;
    PoseMatrix* pose_storage = nullptr;
    std::vector<GPUEdgeFrame*> edge_a;
    std::vector<GPUDilatedFrame*> dilated_a;
    std::vector<GPUIntensityFrame*> intensity_a;
    std::vector<GPUFrame*> distance_maps;
    std::vector<GPUHeatmap*> heatmaps;
    std::vector<GPUModel*> non_principal;
    jta_cost_function::CostFunctionManager* trunk = nullptr;

    ~Pipeline() {
        delete trunk;
        for (auto* p : edge_a) delete p;
        for (auto* p : dilated_a) delete p;
        for (auto* p : intensity_a) delete p;
        for (auto* p : distance_maps) delete p;
        for (auto* p : heatmaps) delete p;
        delete pose_storage;
        delete metrics;
        delete model;
    }
};

Pose ToPose(const Point6D& p) {
    return Pose(p.x, p.y, p.z, p.xa, p.ya, p.za);
}

// Builds a monoplane GPU pipeline for one base frame (mirrors
// OptimizerManager::Initialize): uploads the processed Frame outputs (edge /
// dilation / intensity / distance-map / curvature heatmaps) and wires a trunk
// DIRECT_DILATION cost manager. The caller owns the returned Pipeline (its
// destructor frees the GPU objects).
Pipeline BuildFramePipeline(const std::string& base_image) {
    Frame frame(base_image, 3, 0, 150, /*dilation=*/6);
    frame.setCurvatureHeatmaps();

    Pipeline p;
    p.metrics = new GPUMetrics();
    REQUIRE(p.metrics->IsInitializedCorrectly());
    p.pose_storage = new PoseMatrix();

    auto edge_upload = MatToUchar(frame.GetEdgeImage());
    auto edge = new GPUEdgeFrame(
        kWidth, kHeight, kDevice, edge_upload.data(),
        frame.GetHighThreshold(), frame.GetLowThreshold(), frame.GetAperture());
    REQUIRE(edge->IsInitializedCorrectly());
    p.edge_a.push_back(edge);

    auto dil_upload = MatToUchar(frame.GetDilationImage());
    auto dilated = new GPUDilatedFrame(kWidth, kHeight, kDevice,
                                       dil_upload.data(), 6);
    REQUIRE(dilated->IsInitializedCorrectly());
    p.dilated_a.push_back(dilated);

    auto orig_upload = MatToUchar(frame.GetOriginalImage());
    auto inv_upload = MatToUchar(frame.GetInvertedImage());
    auto intensity = new GPUIntensityFrame(kWidth, kHeight, kDevice,
                                           orig_upload.data(), false,
                                           inv_upload.data());
    REQUIRE(intensity->IsInitializedCorrectly());
    p.intensity_a.push_back(intensity);

    auto dist_upload = MatToUchar(frame.GetDistanceMap());
    auto dm = new GPUFrame(kWidth, kHeight, kDevice, dist_upload.data());
    REQUIRE(dm->IsInitializedCorrectly());
    p.distance_maps.push_back(dm);

    auto hm = new GPUHeatmap(kWidth, kHeight, kDevice,
                             frame.GetNumCurvatureKeypoints(),
                             frame.getCurvatureHeatmaps().data());
    REQUIRE(hm->IsInitializedCorrectly());
    p.heatmaps.push_back(hm);

    Model femur(kFemStl, "femur", "femur");
    REQUIRE(femur.initialized_correctly_);
    int triangle_count =
        static_cast<int>(femur.triangle_vertices_.size() / 9);
    REQUIRE(triangle_count > 0);

    CameraCalibration cam(1198.0f, -1.0f * 0.0f, -1.0f * 0.0f, 0.373f);
    Calibration calib(cam);
    p.model = new GPUModel("femur", /*principal=*/true, kWidth, kHeight,
                           kDevice, /*use_backface_culling=*/false,
                           &femur.triangle_vertices_[0],
                           &femur.triangle_normals_[0], triangle_count,
                           calib.camera_A_principal_);
    REQUIRE(p.model->IsInitializedCorrectly());

    p.trunk = new jta_cost_function::CostFunctionManager(Stage::Trunk);
    p.trunk->setActiveCostFunction("DIRECT_DILATION");
    p.trunk->updateCostFunctionParameterValues("DIRECT_DILATION",
                                                "Dilation", 6);
    p.trunk->UploadData(&p.edge_a, &p.dilated_a, &p.intensity_a, &p.edge_a,
                        &p.dilated_a, &p.intensity_a, p.model,
                        &p.non_principal, p.metrics, p.pose_storage,
                        /*biplane=*/false);
    p.trunk->UploadDistanceMap(&p.distance_maps, &p.heatmaps);
    p.trunk->setCurrentFrameIndex(0);
    return p;
}

}  // namespace

TEST_CASE("Tier-2 GPU oracle: recovered femur silhouette matches the label",
          "[oracle][gpu]") {
    // Load the three candidate labels once (GPU). The per-frame correspondence
    // is resolved empirically below (binary label TIFFs are bottom-left
    // y-origin, so we vertically flip them).
    std::vector<std::vector<unsigned char>> label_bufs;
    std::vector<GPUImage*> label_gpus;
    for (const auto& path : kLabels) {
        label_bufs.push_back(GrayscaleUchar(path, /*flip_vertical=*/true));
        label_gpus.push_back(new GPUImage(kWidth, kHeight, kDevice,
                                          label_bufs.back().data()));
        REQUIRE(label_gpus.back()->IsInitializedCorrectly());
    }

    auto start_poses = StartPoses();
    REQUIRE(start_poses.size() == kBaseImages.size());
    const unsigned int kBudget = 3000;  // few minutes; production uses 20k/25k/30k
    const double kIouGate = 0.85;

    // Loop over all three Kneel_1 frames: per-frame pipeline, per-frame
    // empirical label correspondence, per-frame optimize, per-frame IoU gate.
    // This closes the original frame-0-only coverage gap.
    for (int f = 0; f < (int)kBaseImages.size(); ++f) {
        std::cout << "[oracle] --- frame " << f << " (" << kBaseImages[f]
                  << ") ---" << std::endl;

        Pipeline p = BuildFramePipeline(kBaseImages[f]);
        std::string err;
        REQUIRE(p.trunk->InitializeActiveCostFunction(err));
        Point6D start = start_poses[f];

        /*--- Diagnostic: render at the fem.jts start pose, find the label
         * whose silhouette the start pose matches best. Pins this frame's
         * correspondence empirically (names are not aligned). ---*/
        Pose golden = ToPose(start);
        p.model->SetCurrentPrimaryCameraPose(golden);
        REQUIRE(p.model->RenderPrimaryCamera(golden));
        GPUImage* golden_render = p.model->GetPrimaryCameraRenderedImage();
        int best_label = 0;
        double best_iou = -1.0;
        std::vector<double> start_iou;
        for (int i = 0; i < (int)label_gpus.size(); ++i) {
            double v = p.metrics->IOU(golden_render, label_gpus[i]);
            start_iou.push_back(v);
            std::cout << "[oracle] fem.jts-pose IoU vs label[" << i
                      << "] = " << v << std::endl;
            if (v > best_iou) {
                best_iou = v;
                best_label = i;
            }
        }
        std::cout << "[oracle] selected label index for frame " << f << " = "
                  << best_label << " (" << kLabels[best_label] << ")"
                  << std::endl;
        // Each frame's start pose must match SOME label well; otherwise the
        // pose<->image correspondence is broken and the gate is meaningless.
        REQUIRE(best_iou > 0.50);

        /*--- Optimize: DirectOptimizer bound to the real GPU DIRECT_DILATION
         * cost, exactly as OptimizerManager::RunDirectStage does (U6). ---*/
        auto cost = [&p](const Point6D& physical) -> double {
            p.model->SetCurrentPrimaryCameraPose(ToPose(physical));
            return p.trunk->callActiveCostFunction();
        };
        DirectOptimizer opt(cost, SearchRange(), start, kBudget);
        REQUIRE(opt.Run());
        Point6D recovered = opt.GetOptimumLocation();

        std::cout << "[oracle] frame " << f << " recovered pose: ("
                  << recovered.x << ", " << recovered.y << ", "
                  << recovered.z << ", " << recovered.xa << ", "
                  << recovered.ya << ", " << recovered.za << ")"
                  << std::endl;
        std::cout << "[oracle] frame " << f << " cost calls: "
                  << opt.GetCostFunctionCalls() << std::endl;
        std::cout << "[oracle] frame " << f << " gap vs fem.jts: ("
                  << recovered.x - start.x << ", " << recovered.y - start.y
                  << ", " << recovered.z - start.z << ", "
                  << recovered.xa - start.xa << ", "
                  << recovered.ya - start.ya << ", "
                  << recovered.za - start.za << ")" << std::endl;

        /*--- THE LOAD-BEARING APPEARANCE GATE ---*/
        Pose final_pose = ToPose(recovered);
        p.model->SetCurrentPrimaryCameraPose(final_pose);
        REQUIRE(p.model->RenderPrimaryCamera(final_pose));
        GPUImage* final_render = p.model->GetPrimaryCameraRenderedImage();

        double iou = p.metrics->IOU(final_render, label_gpus[best_label]);
        double l1 = p.metrics->L_1_1_MatrixDifferenceNorm(
            final_render, label_gpus[best_label]);
        double per_px = l1 / (double)(kWidth * kHeight);
        std::cout << "[oracle] frame " << f << " recovered-pose IoU vs label["
                  << best_label << "] = " << iou << std::endl;
        std::cout << "[oracle] frame " << f << " recovered-pose L1 pixel-diff = "
                  << l1 << " (per-px " << per_px << ")" << std::endl;
        CAPTURE(f, iou, best_iou, per_px);

        // The recovered femur silhouette must substantially overlap its
        // known-good label. Gate set from the first measured run on the RTX
        // 3090 (frame 0 recovered IoU = 0.9936): we assert with a healthy
        // margin below that, so a regression in the rewire fails loudly while
        // genuine hardware/float variance passes. IoU in [0,1].
        std::cout << "[oracle] frame " << f << " appearance gate: recovered IoU "
                  << iou << " vs threshold " << kIouGate << "" << std::endl;
        REQUIRE(iou > kIouGate);
    }
}

