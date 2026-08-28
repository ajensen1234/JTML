/*
 * Plan 014 POC — graph-free multi-stream overlap proof-of-concept.
 *
 * Question: can two DIRECT_DILATION evals, enqueued on two per-context
 * non-blocking streams, overlap on the GPU (window < 2 x serial 97us)?
 * This drives the REAL U4 pure-enqueue path (EnqueueRenderPhase +
 * EnqueueFastImplantDilationMetric + EnqueueDistanceMapMetric) directly —
 * NO graph recipe, NO cudaGraphLaunch — exactly the graph-free replay body
 * (graph_recipe_direct_dilation.cu:354-363) sans capture.
 *
 * oracle;gpu — not in the headless default.
 *
 * Measure: enqueue eval A then eval B from one thread, cudaEventRecord both,
 * poll both events; 2-eval window vs 194us (2 x 97us serial). Overlap factor
 * f = 1 - window/(2*97). If window ~120-135 us, f ~ 0.3 => overlap real.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "compute/camera_calibration.h"
#include "compute/gpu_metrics.cuh"
#include "compute/render_engine.cuh"
#include "domain/data_structures_6D.h"

using gpu_cost_function::BankFootprintInput;
using gpu_cost_function::EvaluationContext;
using gpu_cost_function::EvaluationContextPool;
using gpu_cost_function::GPUDilatedFrame;
using gpu_cost_function::GPUFrame;
using gpu_cost_function::GPUImage;
using gpu_cost_function::GPUMetrics;
using gpu_cost_function::RenderEngine;

namespace {
constexpr int kW = 1024, kH = 1024, kTri = 12412, kDil = 6;
constexpr const char* kStl = "example_studies/Kneel_1/KR_right_7_fem.stl";
constexpr std::size_t kMaxStride = 10000000;

struct POCFixture {
    std::vector<float> tris{};
    std::vector<float> norms{};
    std::unique_ptr<RenderEngine> eng{};
    std::unique_ptr<GPUMetrics> met{};
    std::unique_ptr<GPUImage> compImg{};
    std::unique_ptr<GPUDilatedFrame> compFrm{};
    std::unique_ptr<GPUFrame> distMap{};

    bool loadStl() {
        std::ifstream file(kStl);
        if (!file) {
            return false;
        }
        tris.reserve(static_cast<std::size_t>(kTri) * 9);
        norms.reserve(static_cast<std::size_t>(kTri) * 3);
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream facet(line);
            std::string keyword, normal_keyword;
            float nx = 0.0f, ny = 0.0f, nz = 0.0f;
            facet >> keyword >> normal_keyword >> nx >> ny >> nz;
            if (keyword != "facet" || normal_keyword != "normal") {
                continue;
            }
            std::vector<float> vertices;
            while (vertices.size() < 9 && std::getline(file, line)) {
                std::istringstream vertex_line(line);
                std::string vkw;
                vertex_line >> vkw;
                if (vkw != "vertex") {
                    continue;
                }
                float x = 0.0f, y = 0.0f, z = 0.0f;
                vertex_line >> x >> y >> z;
                vertices.insert(vertices.end(), {x, y, z});
            }
            if (vertices.size() != 9) {
                return false;
            }
            tris.insert(tris.end(), vertices.begin(), vertices.end());
            norms.insert(norms.end(), {nx, ny, nz});
        }
        return tris.size() == static_cast<std::size_t>(kTri) * 9 &&
            norms.size() == static_cast<std::size_t>(kTri) * 3;
    }

    bool setup() {
        int dc = 0;
        if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) {
            return false;
        }
        if (!loadStl()) {
            return false;
        }
        CameraCalibration cal(1198.0f, 0.0f, 0.0f, 0.373f);
        eng = std::make_unique<RenderEngine>(
            kW, kH, 0, false, tris.data(), norms.data(), kTri, cal);
        if (!eng->IsInitializedCorrectly()) {
            return false;
        }
        met = std::make_unique<GPUMetrics>();
        if (!met->IsInitializedCorrectly()) {
            return false;
        }
        std::vector<unsigned char> host(kW * kH, 0);
        for (int y = kH / 4; y < 3 * kH / 4; ++y) {
            for (int x = kW / 4; x < 3 * kW / 4; ++x) {
                host[y * kW + x] = 255;
            }
        }
        compImg = std::make_unique<GPUImage>(kW, kH, 0, host.data());
        compFrm =
            std::make_unique<GPUDilatedFrame>(kW, kH, 0, host.data(), kDil);
        distMap = std::make_unique<GPUFrame>(kW, kH, 0, host.data());
        return compImg->IsInitializedCorrectly() &&
            compFrm->IsInitializedCorrectly() &&
            distMap->IsInitializedCorrectly();
    }
};

// Enqueue the full DIRECT_DILATION eval chain on ctx.stream (graph-free replay
// body).
static bool EnqueueEval(POCFixture& fix, EvaluationContext& ctx, Point6D pose) {
    if (!ctx.initialized_correctly || !ctx.in_flight || ctx.stream == nullptr ||
        ctx.completion_event == nullptr) {
        return false;
    }
    ctx.x_location = pose.x;
    ctx.y_location = pose.y;
    ctx.z_location = pose.z;
    ctx.x_angle = pose.xa;
    ctx.y_angle = pose.ya;
    ctx.z_angle = pose.za;

    cudaError_t err = fix.eng->EnqueueRenderPhase(ctx);
    if (err != cudaSuccess) {
        return false;
    }
    err = fix.met->EnqueueFastImplantDilationMetric(
        fix.compImg.get(), fix.compFrm.get(), kDil, ctx);
    if (err != cudaSuccess) {
        return false;
    }
    err = fix.met->EnqueueDistanceMapMetric(
        fix.compImg.get(), fix.distMap.get(), kDil, ctx);
    if (err != cudaSuccess) {
        return false;
    }
    // Record completion AFTER the last async op so EventQuery gates the pins.
    err = cudaEventRecord(
        reinterpret_cast<cudaEvent_t>(ctx.completion_event),
        reinterpret_cast<cudaStream_t>(ctx.stream));
    return err == cudaSuccess;
}

static bool IsDone(EvaluationContext& ctx) {
    cudaError_t q =
        cudaEventQuery(reinterpret_cast<cudaEvent_t>(ctx.completion_event));
    return q == cudaSuccess;
}

static double WaitFor(EvaluationContext& a, EvaluationContext& b) {
    // Event-based completion with bounded backoff (plan-013 U1 discipline:
    // never hot-spin cudaEventQuery). Poll both, then sleep the bounded pacing
    // window.
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 200000; ++i) {
        if (IsDone(a) && IsDone(b)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

}  // namespace

TEST_CASE(
    "graph-free two-stream overlap window (POC)",
    "[graph_free][oracle][gpu][p014]") {
    POCFixture fix;
    REQUIRE(fix.setup());

    EvaluationContextPool pool;
    gpu_cost_function::BankFootprintInput layout0{};
    layout0.width = kW;
    layout0.height = kH;
    layout0.triangle_count = kTri;
    layout0.maximum_stride_size = kMaxStride;
    layout0.cub_storage_bytes =
        (fix.eng ? fix.eng->GetCubStorageBytes() : 4096);
    layout0.curvature_capacity = 0;
    layout0.graph_overhead_bytes = 0;
    layout0.biplane = false;
    std::size_t freeB = 0, totB = 0;
    REQUIRE(cudaMemGetInfo(&freeB, &totB) == cudaSuccess);
    REQUIRE(pool.Initialize(layout0, freeB, 2));
    REQUIRE(pool.size() >= 2);

    int idxA = pool.Checkout();
    int idxB = pool.Checkout();
    REQUIRE(idxA >= 0);
    REQUIRE(idxB >= 0);
    REQUIRE(idxA != idxB);
    EvaluationContext* cA = pool.context(idxA);
    EvaluationContext* cB = pool.context(idxB);
    REQUIRE(cA != nullptr);
    REQUIRE(cB != nullptr);
    cA->in_flight = true;
    cB->in_flight = true;
    (void)idxA;
    (void)idxB;

    // Warmup one eval first (cold caches / first-launch of persistent workers).
    {
        REQUIRE(EnqueueEval(
            fix, *cA, Point6D(0.0f, 0.0f, -900.0f, 0.0f, 0.0f, 0.0f)));
        WaitFor(*cA, *cA);
    }

    // --- measured N=2 overlap ---
    Point6D pa(0.0f, 0.0f, -900.0f, 0.0f, 0.0f, 0.0f);
    Point6D pb(2.0f, 1.0f, -900.0f, 0.0f, 0.0f, 0.0f);
    auto t0 = std::chrono::steady_clock::now();
    REQUIRE(EnqueueEval(fix, *cA, pa));
    REQUIRE(EnqueueEval(fix, *cB, pb));
    // Measure the two-eval window with bounded backoff (not a hot-spin).
    double windowUs = WaitFor(*cA, *cB);
    REQUIRE(windowUs > 0);

    // Serial 2-pose baseline is 2 x ~97us = ~194us single-stream.
    double serial2Us = 2.0 * 97.0;
    double f = 1.0 - windowUs / serial2Us;  // overlap fraction
    std::string info = "[p014] 2-window " + std::to_string(windowUs) +
        " us; f=" + std::to_string(f);
    INFO(info);

    // Anti-stub: both events really fired (not fabricated).
    REQUIRE(IsDone(*cA));
    REQUIRE(IsDone(*cB));

    std::cerr << "\n[P014-RESULT] two-eval window=" << windowUs
              << " us  f=" << f << "  (serial2=" << serial2Us << " us)\n\n";

    // Report (non-gating evidence): if f ~0.3+, overlap is real and plan-014
    // proceeds. We do NOT hard-fail on the window here — the POC records it
    // for the plan-014 admission decision. SUCCEED with the measured f.
    SUCCEED(
        "graph-free N=2 overlap window measured: " + std::to_string(windowUs) +
        " us, f=" + std::to_string(f));
}
