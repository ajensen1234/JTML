/* * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/* @file graph_capture_probe_test.cu
 *
 * Plan 011 U3 (Phase A). GPU-labeled oracle for the preflight probe
 * in src/compute/graph_preflight.cu. Requires a CUDA device (oracle label).
 * Run explicitly with: ctest --test-dir .build -L oracle -R graph_capture_probe
 *
 * Test scenarios (from plan's U3):
 *  - Serial path: real RenderPhase callback discovers cudaStreamSynchronize
 *    blocker via actual capture attempt (no hardcoded constant).
 *  - Synthetic capturable micro-graph: dummy kernel + Memset captures and
 *    instantiates, proving the toolchain can capture at all.
 *  - ProbeCapturableOpSet: callback-based probe with real capturable and
 *    non-capturable operation sets.
 *  - Edge case: 0-triangle or maximum_stride_size overflow reports
 *    capturable=false with correct reasonCode.
 *  - Error path: probe cleans up after actual capture invalidation,
 *    then verifies subsequent probes still work (no global state pollution).
 */

#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>

#include "compute/camera_calibration.h"
#include "compute/cost_capacity_service.cuh"
#include "compute/graph_preflight.h"
#include "compute/render_engine.cuh"

// ---------------------------------------------------------------------------
// RenderPhase capture callback context — saves/restores bank.stream so the
// CostCapacityService pool is never corrupted by the capture stream.
// ---------------------------------------------------------------------------
struct RenderPhaseProbeCtx {
    gpu_cost_function::RenderEngine* engine = nullptr;
    gpu_cost_function::BankState* bank = nullptr;
};

static int renderPhaseCaptureOp(void* stream_ptr, void* ctx_ptr) {
    auto* c = static_cast<RenderPhaseProbeCtx*>(ctx_ptr);
    // Swap bank.stream to the capture stream so RenderPhase enqueues all
    // kernels (memset, ResetKernel, WorldToPixel, BoundingBoxForTriangles,
    // BoundingBoxSizes, CUB scan, PrepareLaunchPacket, memcpy D2H,
    // cudaStreamSynchronize) onto the captured stream.
    void* original_stream = c->bank->stream;
    c->bank->stream = stream_ptr;
    // RenderPhase calls BindBankPointers which rebinds the engine's internal
    // pointers to the bank's buffers, then launches the real kernel chain.
    // The cudaStreamSynchronize inside RenderPhase will invalidate capture.
    cudaError_t err = c->engine->RenderPhase(*c->bank);
    // Restore original stream so pool ownership is not corrupted.
    c->bank->stream = original_stream;
    // Restore bank-0 pointers so the engine is in a clean state for subsequent
    // calls (e.g. the synthetic probe in the cleanup test).
    c->engine->SetActiveBank(nullptr);
    return static_cast<int>(err);
}

// Helper: minimal RenderEngine + bank from CostCapacityService.
// The engine and service must outlive the returned context.
struct RenderPhaseFixture {
    gpu_cost_function::CostCapacityService service;
    std::unique_ptr<gpu_cost_function::RenderEngine> engine;
    int bank_idx = -1;
    gpu_cost_function::BankState* bank = nullptr;
    RenderPhaseProbeCtx ctx{};

    bool setup() {
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
            return false;

        // Minimal single triangle (right-hand, CCW winding).
        // clang-format off
        static float s_triangles[] = {
            0.f, 0.f, 0.f,
            1.f, 0.f, 0.f,
            0.f, 1.f, 0.f,
        };
        static float s_normals[] = {0.f, 0.f, 1.f};
        // clang-format on

        CameraCalibration cal(100.f, 4.f, 4.f, 0.1f);
        constexpr int kW = 8, kH = 8;
        engine = std::make_unique<gpu_cost_function::RenderEngine>(
            kW, kH, 0, false, s_triangles, s_normals, 1, cal);
        if (!engine->IsInitializedCorrectly()) return false;

        // Configure CostCapacityService with enough free memory for 2 banks.
        service.setSnapshot(gpu_cost_function::DeviceCapacitySnapshot{
            .sm_count = 1,
            .max_threads_per_sm = 1024,
            .safe_cap = 1000000LL,
            .grid_dim_limit = 100000LL,
            .free_device_bytes = 1024LL * 1024 * 1024,
            .per_bank_footprint_bytes = 0,
            .n_max = 0,
        });

        gpu_cost_function::BankFootprintInput layout{};
        layout.width = kW;
        layout.height = kH;
        layout.triangle_count = 1;
        layout.maximum_stride_size = 10000;
        layout.cub_storage_bytes = engine->GetCubStorageBytes();
        layout.curvature_capacity = 0;
        layout.graph_overhead_bytes = 0;
        layout.biplane = false;

        if (!service.ConfigurePool(layout, 2)) return false;

        bank_idx = service.CheckoutBank();
        if (bank_idx < 1) return false;

        bank = service.bankState(static_cast<std::size_t>(bank_idx));
        if (bank == nullptr || bank->stream == nullptr) return false;

        // Bind the bank to the engine so BindBankPointers succeeds inside
        // RenderPhase.
        engine->SetActiveBank(bank);

        ctx.engine = engine.get();
        ctx.bank = bank;
        return true;
    }

    void teardown() {
        if (bank_idx >= 0)
            service.RecycleBank(static_cast<std::size_t>(bank_idx), true);
    }
};

TEST_CASE(
    "U3 probe: serial path is non-capturable — real RenderPhase discovers "
    "sync blocker",
    "[preflight][gpu]") {
    using namespace gpu_cost_function;

    RenderPhaseFixture fix;
    REQUIRE(fix.setup());

    // ProbeCurrentSerialPath forwards to ProbeCapturableOpSet with the real
    // RenderPhase callback. RenderPhase executes the actual kernel chain
    // (memset → ResetKernel → WorldToPixel → BoundingBoxForTriangles →
    // BoundingBoxSizes → CUB scan → PrepareLaunchPacket → memcpy D2H →
    // cudaStreamSynchronize), and the sync invalidates Global capture.
    const auto result = ProbeCurrentSerialPath(renderPhaseCaptureOp, &fix.ctx);
    INFO(
        "ProbeCurrentSerialPath (real RenderPhase): "
        << FormatPreflightResult(result));
    REQUIRE_FALSE(result.capturable);
    REQUIRE(result.reasonCode == kSyncBlocker);
    REQUIRE_FALSE(result.failingNodeHint.empty());
    REQUIRE_FALSE(result.reasonString.empty());

    // Must not crash and must be deterministic.
    fix.engine->SetActiveBank(fix.bank);
    const auto again = ProbeCurrentSerialPath(renderPhaseCaptureOp, &fix.ctx);
    REQUIRE(again.capturable == result.capturable);
    REQUIRE(again.reasonCode == result.reasonCode);

    fix.teardown();
}

TEST_CASE("U3 probe: synthetic micro-graph is capturable", "[preflight][gpu]") {
    using namespace gpu_cost_function;

    int deviceCount = 0;
    cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
    if (countErr != cudaSuccess || deviceCount == 0) {
        WARN("No CUDA device — skipping synthetic capture assertion");
        return;
    }

    const auto result = ProbeSyntheticMicroGraph(nullptr);
    INFO("ProbeSyntheticMicroGraph: " << FormatPreflightResult(result));
    REQUIRE(result.capturable);
    REQUIRE(result.reasonCode == kPreflightOk);
    REQUIRE(result.failingNodeHint.empty());
}

TEST_CASE(
    "U3 probe: synthetic micro-graph cleans up and is reusable",
    "[preflight][gpu]") {
    using namespace gpu_cost_function;
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        WARN("No CUDA device — skipping");
        return;
    }

    const auto r1 = ProbeSyntheticMicroGraph(nullptr);
    const auto r2 = ProbeSyntheticMicroGraph(nullptr);
    INFO("r1: " << FormatPreflightResult(r1));
    INFO("r2: " << FormatPreflightResult(r2));
    REQUIRE(r1.capturable);
    REQUIRE(r2.capturable);
    REQUIRE(r1.reasonCode == kPreflightOk);
    REQUIRE(r2.reasonCode == kPreflightOk);
}

TEST_CASE(
    "U3 probe: ProbeCapturableOpSet with capturable callback",
    "[preflight][gpu]") {
    using namespace gpu_cost_function;
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        WARN("No CUDA device — skipping");
        return;
    }

    int* d_val = nullptr;
    cudaError_t allocErr = cudaMalloc(&d_val, sizeof(int));
    REQUIRE(allocErr == cudaSuccess);

    auto capturable_cb = [](void* stream_ptr, void* ctx) -> int {
        auto* d = static_cast<int*>(ctx);
        cudaStream_t s = static_cast<cudaStream_t>(stream_ptr);
        cudaMemsetAsync(d, 0, sizeof(int), s);
        return 0;
    };

    auto result = ProbeCapturableOpSet(capturable_cb, d_val);
    INFO("ProbeCapturableOpSet capturable: " << FormatPreflightResult(result));
    REQUIRE(result.capturable);
    REQUIRE(result.reasonCode == kPreflightOk);
    REQUIRE(result.failingNodeHint.empty());
    cudaFree(d_val);
}

TEST_CASE(
    "U3 probe: ProbeCapturableOpSet detects sync blocker", "[preflight][gpu]") {
    using namespace gpu_cost_function;
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        WARN("No CUDA device — skipping");
        return;
    }

    int* d_val = nullptr;
    cudaError_t allocErr = cudaMalloc(&d_val, sizeof(int));
    REQUIRE(allocErr == cudaSuccess);

    auto sync_cb = [](void* stream_ptr, void* ctx) -> int {
        auto* d = static_cast<int*>(ctx);
        cudaStream_t s = static_cast<cudaStream_t>(stream_ptr);
        cudaMemsetAsync(d, 0, sizeof(int), s);
        cudaStreamSynchronize(s);
        return 0;
    };

    auto result = ProbeCapturableOpSet(sync_cb, d_val);
    INFO("ProbeCapturableOpSet sync: " << FormatPreflightResult(result));
    REQUIRE_FALSE(result.capturable);
    REQUIRE(result.reasonCode == kSyncBlocker);
    REQUIRE_FALSE(result.failingNodeHint.empty());
    cudaFree(d_val);
}

TEST_CASE(
    "U3 probe: ProbeCapturableOpSet with null callback", "[preflight][gpu]") {
    using namespace gpu_cost_function;
    auto result = ProbeCapturableOpSet(nullptr, nullptr);
    REQUIRE_FALSE(result.capturable);
    REQUIRE(result.reasonCode == kNoEligibleRecipe);
}

TEST_CASE(
    "U3 probe: zero-triangle edge case reports capturable=false",
    "[preflight][gpu]") {
    using namespace gpu_cost_function;
    const auto result = ProbeZeroTriangle(nullptr);
    REQUIRE_FALSE(result.capturable);
    REQUIRE(result.reasonCode == kZeroTriangle);
    REQUIRE(result.failingNodeHint.find("zero") != std::string::npos);
    const auto again = ProbeZeroTriangle(nullptr);
    REQUIRE(again.reasonCode == result.reasonCode);
}

TEST_CASE(
    "U3 probe: overflow edge case reports capturable=false",
    "[preflight][gpu]") {
    using namespace gpu_cost_function;
    const auto result = ProbeOverflowCase(nullptr);
    REQUIRE_FALSE(result.capturable);
    REQUIRE(result.reasonCode == kOverflow);
    REQUIRE(
        result.failingNodeHint.find("maximum_stride_size") !=
        std::string::npos);
}

TEST_CASE("U3 probe: FormatPreflightResult is stable", "[preflight][gpu]") {
    using namespace gpu_cost_function;

    RenderPhaseFixture fix;
    REQUIRE(fix.setup());

    const auto r = ProbeCurrentSerialPath(renderPhaseCaptureOp, &fix.ctx);
    const std::string s = FormatPreflightResult(r);
    REQUIRE(s.find("capturable=false") != std::string::npos);
    REQUIRE(s.find("reasonCode=") != std::string::npos);
    REQUIRE(s.find("hint=") != std::string::npos);

    fix.teardown();
}

TEST_CASE(
    "U3 probe: error path cleans up after actual capture invalidation",
    "[preflight][gpu]") {
    using namespace gpu_cost_function;
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        WARN("No CUDA device — skipping");
        return;
    }

    RenderPhaseFixture fix;
    REQUIRE(fix.setup());

    // ProbeCurrentSerialPath causes a capture invalidation via real
    // cudaStreamSynchronize on a captured stream (Global mode).
    const auto fail = ProbeCurrentSerialPath(renderPhaseCaptureOp, &fix.ctx);
    REQUIRE_FALSE(fail.capturable);
    REQUIRE(fail.reasonCode == kSyncBlocker);

    // Verify no global state pollution: synthetic probe must still succeed.
    const auto ok = ProbeSyntheticMicroGraph(nullptr);
    INFO("after invalidation, synthetic: " << FormatPreflightResult(ok));
    REQUIRE(ok.capturable);
    REQUIRE(ok.reasonCode == kPreflightOk);

    fix.teardown();
}
