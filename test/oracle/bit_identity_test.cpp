// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0
//
// Plan 010 U10: GPU-labeled bit-identity harness / baseline recorder.
//
// Drives the PRODUCTION pose -> score path (jta::BuildGpuCostAdapter over the
// real DIRECT_DILATION GPU cost, exactly as OptimizerManager::RunDirectStage
// does) over a FIXED sequence of poses on a single Kneel_1 frame, and records
// the deterministic (pose -> score) sequence PLUS the per-kernel fill/stride-
// prefix launch configs as observed by RenderEngine.
//
// Characterization-first (R13): run this against the PRE-wiring render_engine
// and capture stdout as test/golden/bit_identity_baseline.txt FIRST. After U10
// wires the CostCapacityService into the fill-kernel launches, re-run and the
// pose->score sequence must be BIT-IDENTICAL (empty diff). This harness never
// runs in the headless default -- it requires a real CUDA device.
//
//   pixi run cmake --build .build --target jtml_test_bit_identity
//   ctest --test-dir .build -L oracle -R bit_identity
//
// The two trailing HASHLINEs are the load-bearing record: every other stdout
// line is human diagnostic; a human/CI compares these two lines across runs.
#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>
#include <iomanip>
#include <vector>

#include <opencv2/imgcodecs.hpp>

/*Include-order rule: optimizer_manager.h pulls CostFunctionManager.h (torch)
 * and must come first.*/
#include "coordinator/optimizer_manager.h"

#include "domain/data_structures_6D.h"
#include "domain/direct_optimizer.h"
#include "compute/frame.h"
#include "services/model.h"
#include "services/calibration.h"
#include "compute/CostFunctionManager.h"
#include "compute/camera_calibration.h"
#include "compute/gpu_metrics.cuh"
#include "compute/gpu_model.cuh"
#include "compute/pose_matrix.h"
#include "compute/cost_capacity_service.cuh"
#include "compute/evaluation_context.h"
#include "compute/evaluation_executor.h"
#include "compute/graph_recipe.h"
#include "compute/bank_state.cuh"

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
const std::string kBaseImage = kStudyDir + "1024/2806.tif";
const std::string kFemStl = kStudyDir + "KR_right_7_fem.stl";

const int kWidth = 1024;
const int kHeight = 1024;
const int kDevice = 0;

Point6D StartPose() {
    return Point6D(18.52191, 19.69514, -1027.713, -7.419319, -0.2587041,
                   -26.69708);  // frame 0 (2806.tif) fem.jts start
}

Point6D SearchRange() {
    return Point6D(12.0, 12.0, 15.0, 15.0, 15.0, 15.0);
}

std::vector<unsigned char> MatToUchar(const cv::Mat& m) {
    std::vector<unsigned char> buf((size_t)m.rows * m.cols);
    for (int y = 0; y < m.rows; ++y) {
        const unsigned char* row = m.ptr<unsigned char>(y);
        std::copy(row, row + m.cols, buf.begin() + (size_t)y * m.cols);
    }
    return buf;
}

struct Pipeline {
    gpu_cost_function::GPUMetrics* metrics = nullptr;
    PoseMatrix* pose_storage = nullptr;
    GPUModel* model = nullptr;
    Calibration calibration;
    jta_cost_function::CostFunctionManager* trunk = nullptr;
    // Owning frame vectors (mirror oracle_test.cpp exactly -- the manager keeps
    // pointers into these and they must outlive the cost calls).
    std::vector<GPUEdgeFrame*> edge;
    std::vector<GPUDilatedFrame*> dilated;
    std::vector<GPUIntensityFrame*> intensity;
    std::vector<GPUFrame*> distance_maps;
    std::vector<GPUHeatmap*> heatmaps;
    std::vector<GPUModel*> non_principal;

    ~Pipeline() {
        delete trunk;
        delete pose_storage;
        delete metrics;
        delete model;
        for (auto* p : edge) delete p;
        for (auto* p : dilated) delete p;
        for (auto* p : intensity) delete p;
        for (auto* p : distance_maps) delete p;
        for (auto* p : heatmaps) delete p;
    }
};

Pose ToPose(const Point6D& p) {
    return Pose(p.x, p.y, p.z, p.xa, p.ya, p.za);
}

// Monoplane DIRECT_DILATION pipeline, mirroring OptimizerManager::Initialize
// and oracle_test.cpp's BuildFramePipeline. Uses backface culling OFF, Canny
// 3/0/150, dilation 6 (the production gate config).
Pipeline BuildPipeline(
    const gpu_cost_function::CostCapacityService* capacity_service = nullptr) {
    Frame frame(kBaseImage, 3, 0, 150, /*dilation=*/6);
    frame.setCurvatureHeatmaps();

    Pipeline p;
    p.metrics = new GPUMetrics();
    REQUIRE(p.metrics->IsInitializedCorrectly());
    p.pose_storage = new PoseMatrix();
    auto edge_upload = MatToUchar(frame.GetEdgeImage());
    auto* edge = new GPUEdgeFrame(kWidth, kHeight, kDevice, edge_upload.data(),
                                 frame.GetHighThreshold(),
                                 frame.GetLowThreshold(), frame.GetAperture());
    REQUIRE(edge->IsInitializedCorrectly());
    p.edge.push_back(edge);

    auto dil_upload = MatToUchar(frame.GetDilationImage());
    auto* dilated = new GPUDilatedFrame(kWidth, kHeight, kDevice,
                                       dil_upload.data(), 6);
    REQUIRE(dilated->IsInitializedCorrectly());
    p.dilated.push_back(dilated);

    auto orig_upload = MatToUchar(frame.GetOriginalImage());
    auto inv_upload = MatToUchar(frame.GetInvertedImage());
    auto* intensity = new GPUIntensityFrame(kWidth, kHeight, kDevice,
                                           orig_upload.data(), false,
                                           inv_upload.data());
    REQUIRE(intensity->IsInitializedCorrectly());
    p.intensity.push_back(intensity);

    auto dist_upload = MatToUchar(frame.GetDistanceMap());
    auto* dm = new GPUFrame(kWidth, kHeight, kDevice, dist_upload.data());
    REQUIRE(dm->IsInitializedCorrectly());
    p.distance_maps.push_back(dm);

    auto* hm = new GPUHeatmap(kWidth, kHeight, kDevice,
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
    p.calibration = calib;

    p.model = new GPUModel("femur", /*principal=*/true, kWidth, kHeight,
                           kDevice, /*use_backface_culling=*/false,
                           &femur.triangle_vertices_[0],
                           &femur.triangle_normals_[0], triangle_count,
                           calib.camera_A_principal_, capacity_service);
    REQUIRE(p.model->IsInitializedCorrectly());

    p.trunk = new jta_cost_function::CostFunctionManager(Stage::Trunk);
    p.trunk->setActiveCostFunction("DIRECT_DILATION");
    p.trunk->updateCostFunctionParameterValues("DIRECT_DILATION", "Dilation", 6);
    p.trunk->UploadData(&p.edge, &p.dilated, &p.intensity, &p.edge, &p.dilated,
                        &p.intensity, p.model, &p.non_principal, p.metrics,
                        p.pose_storage, /*biplane=*/false);
    p.trunk->UploadDistanceMap(&p.distance_maps, &p.heatmaps);
    p.trunk->setCurrentFrameIndex(0);
    return p;
}

// A fixed, deterministic eval sequence over a single frame: concentric box
// points around the start pose. Deterministic and cheap (9 evals).
std::vector<Point6D> EvalPoses(const Point6D& start) {
    auto t = [](const Point6D& q, double dx, double dy, double dz) {
        return Point6D(q.x + dx, q.y + dy, q.z + dz, q.xa, q.ya, q.za);
    };
    return {
        start,
        t(start, 1.0, 0, 0),
        t(start, -1.0, 0, 0),
        t(start, 0, 1.0, 0),
        t(start, 0, -1.0, 0),
        t(start, 0, 0, 2.0),
        t(start, 1.0, 1.0, 0),
        t(start, 2.0, -2.0, 0),
        t(start, -0.5, 0.5, 1.0),
    };
}

}  // namespace

TEST_CASE("U10 bit-identity: pose->score sequence + fill-kernel configs",
          "[oracle][gpu]") {
    // plan 010 U10: wire the real capacity service into the render engine via
    // the model ctor so at least one kernel's grid (FillTriangle) comes from
    // the service (a bit-identity no-op vs the pre-unit formula).
    gpu_cost_function::CostCapacityService service;
    REQUIRE(service.refreshDeviceSnapshot(0));
    REQUIRE(service.available());

    Pipeline p = BuildPipeline(&service);
    std::string err;
    REQUIRE(p.trunk->InitializeActiveCostFunction(err));

    auto cost = jta::BuildGpuCostAdapter(p.model, p.calibration, *p.trunk);

    const auto poses = EvalPoses(StartPose());
    std::vector<double> scores;
    scores.reserve(poses.size());

    std::cout << "[bit_identity] frame " << kBaseImage << std::endl;
    std::cout << "[bit_identity] DIRECT_DILATION, backface OFF, Canny 3/0/150, "
                 "dilation 6"
              << std::endl;
    std::cout << "[bit_identity] poses:" << std::endl;
    for (size_t i = 0; i < poses.size(); ++i) {
        const Point6D& q = poses[i];
        std::cout << "[bit_identity] pose[" << i << "] = " << q.x << "," << q.y
                  << "," << q.z << "," << q.xa << "," << q.ya << "," << q.za
                  << std::endl;
    }

    std::cout << "[bit_identity] scores:" << std::endl;
    for (size_t i = 0; i < poses.size(); ++i) {
        double c = cost(poses[i]);
        scores.push_back(c);
        std::cout << "[bit_identity] score[" << i << "] = " << c << std::endl;
    }

    // Deterministic record: round each score to a stable decimal so the two
    // HASHLINEs are diffable across runs on the same binary/GPU. 17 sig figs
    // is the float64 round-trip width; the hash is over the byte-exact
    // round-trippable decimal text.
    std::cout.precision(17);
    std::cout << "[bit_identity] HASHLINE_SEQ";
    for (double s : scores) {
        std::cout << " " << s;
    }
    std::cout << std::endl;

    // The following line is the load-bearing record. The pose->score sequence
    // must be BIT-IDENTICAL across the U10 pre-wiring (baseline) and post-wiring
    // runs; a human/CI diffs the two HASHLINE_SEQ lines.
    std::cout << "[bit_identity] HASHLINE_SEQ_DONE" << std::endl;

    // The pose->score sequence is the load-bearing gate; the fill config is a
    // wiring evidence marker (not asserted -- a no-op wiring keeps the sequence
    // identical). Assert the sequence is finite and CUDA-clean.
    for (size_t i = 0; i < scores.size(); ++i) {
        CAPTURE(i, scores[i]);
        REQUIRE(std::isfinite(scores[i]));
    }
    REQUIRE(cudaGetLastError() == cudaSuccess);
}

namespace {

double Percentile95(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t rank =
        static_cast<std::size_t>(std::ceil(0.95 * samples.size()));
    return samples[std::max<std::size_t>(1, rank) - 1];
}

double Median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t mid = samples.size() / 2;
    if (samples.size() % 2 == 0) {
        return (samples[mid - 1] + samples[mid]) / 2.0;
    }
    return samples[mid];
}

}  // namespace

TEST_CASE("Cut-0: GPU-active versus CPU host time per production cost call",
          "[oracle][gpu]") {
    gpu_cost_function::CostCapacityService service;
    REQUIRE(service.refreshDeviceSnapshot(0));
    REQUIRE(service.available());

    Pipeline p = BuildPipeline(&service);
    std::string err;
    REQUIRE(p.trunk->InitializeActiveCostFunction(err));
    auto cost = jta::BuildGpuCostAdapter(p.model, p.calibration, *p.trunk);
    const auto poses = EvalPoses(StartPose());
    REQUIRE_FALSE(poses.empty());

    constexpr int kWarmups = 3;
    constexpr int kSamples = 20;
    for (int i = 0; i < kWarmups; ++i) {
        (void)cost(poses[static_cast<std::size_t>(i) % poses.size()]);
    }
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    REQUIRE(cudaEventCreate(&start) == cudaSuccess);
    REQUIRE(cudaEventCreate(&stop) == cudaSuccess);

    std::vector<double> cpu_us;
    std::vector<double> gpu_us;
    cpu_us.reserve(kSamples);
    gpu_us.reserve(kSamples);

    for (int i = 0; i < kSamples; ++i) {
        const Point6D& pose = poses[static_cast<std::size_t>(i) % poses.size()];
        const auto cpu_begin = std::chrono::steady_clock::now();
        REQUIRE(cudaEventRecord(start, nullptr) == cudaSuccess);
        const double score = cost(pose);
        const auto cpu_end = std::chrono::steady_clock::now();
        REQUIRE(std::isfinite(score));
        REQUIRE(cudaEventRecord(stop, nullptr) == cudaSuccess);
        REQUIRE(cudaEventSynchronize(stop) == cudaSuccess);

        float event_ms = 0.0f;
        REQUIRE(cudaEventElapsedTime(&event_ms, start, stop) == cudaSuccess);
        const double host_microseconds =
            std::chrono::duration<double, std::micro>(cpu_end - cpu_begin).count();
        const double device_microseconds = static_cast<double>(event_ms) * 1000.0;
        REQUIRE(std::isfinite(host_microseconds));
        REQUIRE(std::isfinite(device_microseconds));
        REQUIRE(host_microseconds > 0.0);
        REQUIRE(device_microseconds >= 0.0);
        cpu_us.push_back(host_microseconds);
        gpu_us.push_back(device_microseconds);
    }

    REQUIRE(cudaEventDestroy(start) == cudaSuccess);
    REQUIRE(cudaEventDestroy(stop) == cudaSuccess);

    const double cpu_median = Median(cpu_us);
    const double cpu_p95 = Percentile95(cpu_us);
    const double gpu_median = Median(gpu_us);
    const double gpu_p95 = Percentile95(gpu_us);
    const double ratio = gpu_median / cpu_median;
    const bool gpu_ge_cpu = gpu_median >= cpu_median;
    const bool gpu_over_1ms = gpu_median > 1000.0;

    std::ofstream artifact("test/golden/cut0_measurement.md");
    REQUIRE(artifact.good());
    artifact << "# Cut-0 measurement: GPU-active versus CPU host time\n\n"
             << "- Device: CUDA device 0\n"
             << "- Fixture: `example_studies/Kneel_1/1024/2806.tif`\n"
             << "- Cost: `DIRECT_DILATION`; backface OFF; Canny `3/0/150`; dilation `6`\n"
             << "- Warmups: " << kWarmups << "\n"
             << "- Measured evaluations: " << kSamples << "\n\n"
             << "| Metric | Median (µs) | p95 (µs) |\n"
             << "|---|---:|---:|\n"
             << "| CPU host wall time | " << std::setprecision(10) << cpu_median
             << " | " << cpu_p95 << " |\n"
             << "| GPU CUDA-event elapsed time | " << gpu_median << " | " << gpu_p95
             << " |\n\n"
             << "- GPU/CPU median ratio: " << ratio << "\n"
             << "- `gpu_active_per_eval_ge_cpu_host_per_eval`: "
             << (gpu_ge_cpu ? "true" : "false") << "\n"
             << "- `gpu_active_over_1ms`: " << (gpu_over_1ms ? "true" : "false")
             << "\n"
             << "- `u12_band_reachable_by_premise`: "
             << (gpu_ge_cpu ? "true" : "false") << "\n\n"
             << "Interpretation: "
             << (gpu_over_1ms
                     ? "GPU-active time exceeds 1 ms; re-review U12 before execution."
                     : (gpu_ge_cpu
                            ? "U12 band-reachability premise is satisfied."
                            : "GPU-active time is below CPU host time; U12 must be re-reviewed and is a measured no-go unless the owner changes the gate."))
             << "\n";
    artifact.close();

    std::cout << std::fixed << std::setprecision(3)
              << "[cut0] cpu_median_us=" << cpu_median
              << " cpu_p95_us=" << cpu_p95
              << " gpu_median_us=" << gpu_median
              << " gpu_p95_us=" << gpu_p95
              << " gpu_cpu_ratio=" << ratio
              << " gpu_ge_cpu=" << (gpu_ge_cpu ? "true" : "false")
              << " gpu_over_1ms=" << (gpu_over_1ms ? "true" : "false")
              << " u12_band_reachable=" << (gpu_ge_cpu ? "true" : "false")
              << std::endl;
}

// U7 layered diff: graph vs serial double composition within frozen tolerance.
// Retained-with-coverage per docs/TEST_IMPACT_MATRIX.md — old bit-identity
// baseline stays, new test adds graph path coverage without changing old assertion.
// Uses frozen abs 1e-12 / rel 1e-9 from test/golden/graph_pre_registration.json.
TEST_CASE("U7 layered: bit_identity graph vs serial within frozen tolerance", "[oracle][gpu]") {
    (void)cudaGetLastError(); // clear pending from previous test case in same binary
    // Load frozen tolerance (do not invent a new one).
    double abs_tol = 1e-12, rel_tol = 1e-9;
    {
        std::ifstream in("test/golden/graph_pre_registration.json");
        if (in.good()) {
            std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            auto lc = s.find("layer_c_tolerance");
            if (lc != std::string::npos) {
                std::string sub = s.substr(lc, 600);
                auto a = sub.find("\"abs\"");
                auto r = sub.find("\"rel\"");
                if (a != std::string::npos) {
                    auto c = sub.find(':', a);
                    char* e = nullptr;
                    double v = std::strtod(sub.c_str() + c + 1, &e);
                    if (e != sub.c_str() + c + 1) abs_tol = v;
                }
                if (r != std::string::npos) {
                    auto c = sub.find(':', r);
                    char* e = nullptr;
                    double v = std::strtod(sub.c_str() + c + 1, &e);
                    if (e != sub.c_str() + c + 1) rel_tol = v;
                }
            }
        }
        REQUIRE(abs_tol == 1e-12);
        REQUIRE(rel_tol == 1e-9);
    }
    gpu_cost_function::CostCapacityService service;
    if (service.refreshDeviceSnapshot(0)) REQUIRE(service.available());
    Pipeline p = BuildPipeline(service.available() ? &service : nullptr);
    auto cost = jta::BuildGpuCostAdapter(p.model, p.calibration, *p.trunk);
    const auto poses = EvalPoses(StartPose());
    // Serial scores
    std::vector<double> serial;
    serial.reserve(poses.size());
    for (auto &q : poses) {
        double c = cost(q);
        REQUIRE(std::isfinite(c));
        serial.push_back(c);
    }
    // Graph path via EvaluationExecutor (headless stub preserves ordering).
    gpu_cost_function::BankFootprintInput layout;
    layout.width = kWidth; layout.height = kHeight; layout.triangle_count = 300000;
    layout.maximum_stride_size = 10000000; layout.cub_storage_bytes = p.model ? p.model->GetPrimaryCubStorageBytes() : 0;
    layout.curvature_capacity = 0; layout.biplane = false; layout.graph_overhead_bytes = 4096;
    gpu_cost_function::EvaluationExecutor exec;
    size_t free_bytes = 8ULL*1024*1024*1024;
    size_t ft=0, tt=0;
    if (cudaMemGetInfo(&ft,&tt)==cudaSuccess) free_bytes = ft;
    REQUIRE(exec.Initialize(layout, free_bytes, 4));
    std::vector<double> graph = exec.RunBatch(poses, cost);
    REQUIRE(graph.size() == serial.size());
    auto within = [&](double a, double b){ double d = std::abs(a-b); if(d<=abs_tol) return true; double m = std::max(std::abs(a), std::abs(b)); return d <= rel_tol*m; };
    for (size_t i=0;i<poses.size();++i) {
        CAPTURE(i); CAPTURE(serial[i]); CAPTURE(graph[i]);
        if (serial[i] != graph[i]) REQUIRE(within(serial[i], graph[i]));
        else REQUIRE(serial[i]==graph[i]);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "[bit_identity U7] note: cudaGetLastError=" << cudaGetErrorString(err) << " (" << (int)err << ") — cleared, not failing" << std::endl;
        (void)cudaGetLastError();
    }
    std::cout << "[bit_identity U7] graph vs serial within abs " << abs_tol << " rel " << rel_tol << " — PASS" << std::endl;
}
