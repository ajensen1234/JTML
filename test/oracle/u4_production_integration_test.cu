/* * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/* @file u4_production_integration_test.cu
 *
 * Plan 011 U4: REAL production-path GPU integration oracle.
 * Exercises the actual production methods — NOT reimplemented math — to
 * prove the EvaluationContext pipeline matches the legacy BankState path.
 *
 * 1) RenderPhase(EvaluationContext&) vs legacy Render()/BankState:
 * byte-identical output. 2) EnqueueFastImplantDilationMetric(ctx) +
 * EnqueueDistanceMapMetric(ctx) vs legacy serial metrics: matching int
 * reductions.  No host bbox packet barrier. 3) ProbeCapturableOpSet over
 * RenderPhase(ctx) + both EvaluationContext metric enqueues. 4) Edge/edge
 * behavior: degenerate crop, overflow flag comparison.
 *
 * Run: ctest --test-dir .build -L oracle -R u4_production_integration
 */

#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <memory>
#include <vector>

#include "compute/bank_state.cuh"
#include "compute/camera_calibration.h"
#include "compute/cost_capacity_service.cuh"
#include "compute/evaluation_context.h"
#include "compute/fast_implant_dilation_metric.cuh"
#include "compute/gpu_dilated_frame.cuh"
#include "compute/gpu_frame.cuh"
#include "compute/gpu_image.cuh"
#include "compute/gpu_metrics.cuh"
#include "compute/graph_preflight.h"
#include "compute/render_engine.cuh"

// ---------------------------------------------------------------------------
// Minimal fixture: 1-triangle right-hand CCW, small frame.
// ---------------------------------------------------------------------------

namespace {

constexpr int kW = 32;
constexpr int kH = 32;
constexpr int kDilation = 2;

// Single triangle (right-hand CCW winding).
// clang-format off
static float s_triangles[] = {
     0.f,  0.f, 0.f,
     10.f,  0.f, 0.f,
      5.f, 10.f, 0.f,
};
// clang-format on
static float s_normals[] = {0.f, 0.f, 1.f};
static constexpr int kTriangleCount = 1;

struct U4Fixture {
    gpu_cost_function::CostCapacityService service;
    std::unique_ptr<gpu_cost_function::RenderEngine> engine;
    std::unique_ptr<gpu_cost_function::GPUMetrics> metrics;
    std::unique_ptr<gpu_cost_function::GPUImage> comparisonImage;
    std::unique_ptr<gpu_cost_function::GPUDilatedFrame> cf;
    std::unique_ptr<gpu_cost_function::GPUFrame> dm;
    gpu_cost_function::EvaluationContextPool pool;
    int bank_idx = -1;
    gpu_cost_function::BankState* bank = nullptr;
    bool ok = false;

    bool setup() {
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
            return false;

        CameraCalibration cal(100.f, 4.f, 4.f, 0.1f);
        engine = std::make_unique<gpu_cost_function::RenderEngine>(
            kW, kH, 0, false, s_triangles, s_normals, kTriangleCount, cal);
        if (!engine->IsInitializedCorrectly()) return false;

        metrics = std::make_unique<gpu_cost_function::GPUMetrics>();
        if (!metrics->IsInitializedCorrectly()) return false;

        // Create comparison images (same dims as render output).
        // Fill comparisonImage with a known pattern.
        std::vector<unsigned char> hostComp(kW * kH, 0);
        // Put a small white patch in the center.
        for (int y = kH / 4; y < 3 * kH / 4; ++y)
            for (int x = kW / 4; x < 3 * kW / 4; ++x)
                hostComp[y * kW + x] = 255;

        comparisonImage = std::make_unique<gpu_cost_function::GPUImage>(
            kW, kH, 0, hostComp.data());
        if (!comparisonImage->IsInitializedCorrectly()) return false;

        // GPUDilatedFrame and GPUFrame from the same host data.
        cf = std::make_unique<gpu_cost_function::GPUDilatedFrame>(
            kW, kH, 0, hostComp.data(), kDilation);
        if (!cf->IsInitializedCorrectly()) return false;

        dm = std::make_unique<gpu_cost_function::GPUFrame>(
            kW, kH, 0, hostComp.data());
        if (!dm->IsInitializedCorrectly()) return false;

        // Configure CostCapacityService + bank pool.
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
        layout.triangle_count = kTriangleCount;
        layout.maximum_stride_size = 10000;
        layout.cub_storage_bytes = engine->GetCubStorageBytes();
        layout.curvature_capacity = 0;
        layout.graph_overhead_bytes = 0;
        layout.biplane = false;

        if (!service.ConfigurePool(layout, 2)) return false;

        // Bank 0 is the engine's own; bank 1 is the external bank.
        bank_idx = service.CheckoutBank();
        if (bank_idx < 1) return false;
        bank = service.bankState(static_cast<std::size_t>(bank_idx));
        if (bank == nullptr || bank->stream == nullptr) return false;

        // Initialize EvaluationContextPool with the same layout.
        if (!pool.Initialize(layout, 1024ULL * 1024 * 1024, 2)) return false;

        // Set a fixed pose.
        gpu_cost_function::Pose pose(0.5f, 0.3f, -5.0f, 0.f, 0.f, 15.f);
        engine->SetPose(pose);

        ok = true;
        return true;
    }

    void teardown() {
        if (bank_idx >= 0)
            REQUIRE(
                service.RecycleBank(static_cast<std::size_t>(bank_idx), true));
    }
};

// Helper: copy device image to host for comparison.
std::vector<unsigned char> copyImageToHost(gpu_cost_function::GPUImage& img) {
    std::vector<unsigned char> host(kW * kH);
    REQUIRE(
        cudaMemcpy(
            host.data(),
            img.GetDeviceImagePointer(),
            kW * kH * sizeof(unsigned char),
            cudaMemcpyDeviceToHost) == cudaSuccess);
    return host;
}

// Helper: copy raw device image pointer to host.
std::vector<unsigned char>
copyDevicePtrToHost(unsigned char* dev_ptr, std::size_t bytes) {
    std::vector<unsigned char> host(bytes);
    REQUIRE(
        cudaMemcpy(host.data(), dev_ptr, bytes, cudaMemcpyDeviceToHost) ==
        cudaSuccess);
    return host;
}

bool hasContextResources(const gpu_cost_function::EvaluationContext& ctx) {
    const auto& r = ctx.primary;
    const auto& m = ctx.metrics;
    return ctx.initialized_correctly && ctx.in_flight &&
           ctx.stream != nullptr && ctx.completion_event != nullptr &&
           ctx.dev_nextCandidate != nullptr && ctx.dev_nextChunk != nullptr &&
           ctx.dev_overflowFlag != nullptr &&
           ctx.host_overflowFlag != nullptr && r.output != nullptr &&
           r.host_bounding_box != nullptr && r.dev_backface != nullptr &&
           r.dev_transformed_vertex_zs != nullptr &&
           r.dev_tangent_triangle != nullptr &&
           r.dev_projected_triangles != nullptr &&
           r.dev_projected_triangles_snapped != nullptr &&
           r.dev_bounding_box_triangles != nullptr &&
           r.dev_bounding_box_triangles_sizes != nullptr &&
           r.dev_bounding_box_triangles_sizes_prefix != nullptr &&
           r.dev_bounding_box != nullptr && r.dev_fragment_fill != nullptr &&
           r.host_fragment_fill != nullptr &&
           r.dev_stride_prefixes != nullptr && r.dev_cub_storage != nullptr &&
           r.dev_metric_crop != nullptr && m.host_pixel_score != nullptr &&
           m.dev_pixel_score != nullptr && m.host_distance_score != nullptr &&
           m.dev_distance_score != nullptr && m.host_edge_count != nullptr &&
           m.dev_edge_count != nullptr;
}

} // anonymous namespace

// ===========================================================================
// TEST 1: RenderPhase(EvaluationContext&) vs legacy Render() = byte-identical
// output
// ===========================================================================

TEST_CASE(
    "U4 integration: RenderPhase(ctx) produces byte-identical image to legacy "
    "Render()",
    "[u4][integration][gpu]") {
    using namespace gpu_cost_function;

    U4Fixture fix;
    REQUIRE(fix.setup());

    // --- Path A: Legacy serial Render() on bank 0 (engine's own buffers) ---
    fix.engine->SetActiveBank(nullptr); // restore bank 0
    cudaError_t err = fix.engine->Render();
    REQUIRE(err == cudaSuccess);

    // Copy bank-0 output image to host.
    auto legacyImage = copyImageToHost(*fix.engine->GetRenderOutput());

    // --- Path B: RenderPhase(EvaluationContext&) → CompleteRenderPhase ---
    int ctx_idx = fix.pool.Checkout();
    REQUIRE(ctx_idx >= 0);
    EvaluationContext* ctx =
        fix.pool.context(static_cast<std::size_t>(ctx_idx));
    REQUIRE(ctx != nullptr);
    REQUIRE(ctx->initialized_correctly);

    // The context is checked out and owns the complete U4 write set.  Exercise
    // the EvaluationContext overload directly; do not construct a BankState
    // compatibility view for this path.
    REQUIRE(fix.pool.IsInFlight(static_cast<std::size_t>(ctx_idx)));
    REQUIRE(hasContextResources(*ctx));
    err = fix.engine->RenderPhase(*ctx);
    REQUIRE(err == cudaSuccess);
    err = fix.engine->CompleteRenderPhase(*ctx);
    REQUIRE(err == cudaSuccess);
    REQUIRE(fix.engine->GetActiveBank() == nullptr);

    // Copy the EvaluationContext's output image to host.
    auto ctxImage = copyDevicePtrToHost(
        static_cast<unsigned char*>(ctx->primary.output), kW * kH);

    // They must be byte-identical.
    REQUIRE(legacyImage == ctxImage);

    // Restore bank 0 and recycle.
    fix.engine->SetActiveBank(nullptr);
    REQUIRE(fix.pool.Recycle(static_cast<std::size_t>(ctx_idx), true));

    fix.teardown();
}

// ===========================================================================
// TEST 2: EvaluationContext metric paths vs legacy serial metric paths
// ===========================================================================

TEST_CASE(
    "U4 integration: EvaluationContext FID metric matches legacy serial FID",
    "[u4][integration][gpu]") {
    using namespace gpu_cost_function;

    U4Fixture fix;
    REQUIRE(fix.setup());

    // --- Path A: explicit legacy BankState path (baseline only) ---
    // The extra checked-out bank is a complete legacy allocation.  RenderPhase
    // + CompleteRenderPhase and the host-scalar metric APIs are intentionally
    // kept here as the baseline; the optimized path below never sees this view.
    REQUIRE(fix.bank != nullptr);
    REQUIRE(fix.bank->in_flight);
    REQUIRE(fix.bank->stream != nullptr);
    cudaError_t err = fix.engine->RenderPhase(*fix.bank);
    REQUIRE(err == cudaSuccess);
    err = fix.engine->CompleteRenderPhase(*fix.bank);
    REQUIRE(err == cudaSuccess);
    REQUIRE(fix.metrics->TrySetActiveBank(fix.bank));
    auto legacyStream = reinterpret_cast<cudaStream_t>(fix.bank->stream);
    cudaError_t legacyMetricErr = fix.metrics->EnqueueFastImplantDilationMetric(
        nullptr, fix.cf.get(), kDilation, legacyStream);
    REQUIRE(legacyMetricErr == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(legacyStream) == cudaSuccess);
    const int legacyPixelScore =
        *static_cast<int*>(fix.bank->metrics.host_pixel_score);
    const double legacyFID = -1.0 * legacyPixelScore;
    REQUIRE(fix.metrics->TrySetActiveBank(nullptr));
    fix.engine->SetActiveBank(nullptr);

    // --- Path B: EvaluationContext path ---
    int ctx_idx = fix.pool.Checkout();
    REQUIRE(ctx_idx >= 0);
    EvaluationContext* ctx =
        fix.pool.context(static_cast<std::size_t>(ctx_idx));
    REQUIRE(ctx != nullptr);

    // The context is checked out and owns the complete U4 write set.
    REQUIRE(fix.pool.IsInFlight(static_cast<std::size_t>(ctx_idx)));
    REQUIRE(hasContextResources(*ctx));
    err = fix.engine->RenderPhase(*ctx);
    REQUIRE(err == cudaSuccess);

    // Enqueue FID before CompleteRenderPhase — proving the render packet has
    // no host barrier between it and the metric packet.
    cudaError_t metricErr = fix.metrics->EnqueueFastImplantDilationMetric(
        fix.engine->GetRenderOutput(), fix.cf.get(), kDilation, *ctx);
    REQUIRE(metricErr == cudaSuccess);
    err = fix.engine->CompleteRenderPhase(*ctx);
    REQUIRE(err == cudaSuccess);

    // Read the raw reduction from the context-owned pinned host twin.
    int ctxPixelScore = *static_cast<int*>(ctx->metrics.host_pixel_score);
    double ctxFID = -1.0 * ctxPixelScore;

    // Both the legacy result and the raw context reduction must match.
    REQUIRE(legacyFID == ctxFID);
    REQUIRE(ctxPixelScore == legacyPixelScore);
    REQUIRE(fix.engine->GetActiveBank() == nullptr);
    REQUIRE(fix.metrics->GetActiveBank() == nullptr);

    fix.engine->SetActiveBank(nullptr);
    REQUIRE(fix.pool.Recycle(static_cast<std::size_t>(ctx_idx), true));
    fix.teardown();
}

// ===========================================================================
// TEST 3: EvaluationContext DistanceMapMetric matches legacy
// ===========================================================================

TEST_CASE(
    "U4 integration: EvaluationContext DM metric matches legacy serial DM",
    "[u4][integration][gpu]") {
    using namespace gpu_cost_function;

    U4Fixture fix;
    REQUIRE(fix.setup());

    // --- Path A: explicit legacy BankState path (baseline only) ---
    REQUIRE(fix.bank != nullptr);
    REQUIRE(fix.bank->in_flight);
    REQUIRE(fix.bank->stream != nullptr);
    cudaError_t err = fix.engine->RenderPhase(*fix.bank);
    REQUIRE(err == cudaSuccess);
    err = fix.engine->CompleteRenderPhase(*fix.bank);
    REQUIRE(err == cudaSuccess);
    REQUIRE(fix.metrics->TrySetActiveBank(fix.bank));
    auto legacyStream = reinterpret_cast<cudaStream_t>(fix.bank->stream);
    cudaError_t legacyMetricErr = fix.metrics->EnqueueDistanceMapMetric(
        nullptr, fix.dm.get(), kDilation, legacyStream);
    REQUIRE(legacyMetricErr == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(legacyStream) == cudaSuccess);
    const int legacyDMScore =
        *static_cast<int*>(fix.bank->metrics.host_distance_score);
    const int legacyEdgeCount =
        *static_cast<int*>(fix.bank->metrics.host_edge_count);
    const double legacyDM =
        static_cast<double>(legacyDMScore) / (legacyEdgeCount + 0.1);
    REQUIRE(fix.metrics->TrySetActiveBank(nullptr));
    fix.engine->SetActiveBank(nullptr);

    // --- Path B: EvaluationContext ---
    int ctx_idx = fix.pool.Checkout();
    REQUIRE(ctx_idx >= 0);
    EvaluationContext* ctx =
        fix.pool.context(static_cast<std::size_t>(ctx_idx));
    REQUIRE(ctx != nullptr);

    REQUIRE(fix.pool.IsInFlight(static_cast<std::size_t>(ctx_idx)));
    REQUIRE(hasContextResources(*ctx));
    err = fix.engine->RenderPhase(*ctx);
    REQUIRE(err == cudaSuccess);

    // Enqueue DM before CompleteRenderPhase so the test covers the real
    // render→metric stream ordering without a packet barrier.
    cudaError_t metricErr = fix.metrics->EnqueueDistanceMapMetric(
        fix.engine->GetRenderOutput(), fix.dm.get(), kDilation, *ctx);
    REQUIRE(metricErr == cudaSuccess);
    err = fix.engine->CompleteRenderPhase(*ctx);
    REQUIRE(err == cudaSuccess);

    const int ctxDMScore = *static_cast<int*>(ctx->metrics.host_distance_score);
    const int ctxEdgeCount = *static_cast<int*>(ctx->metrics.host_edge_count);
    // Compare both raw host reductions and the legacy scalar reduction.
    REQUIRE(ctxDMScore == legacyDMScore);
    REQUIRE(ctxEdgeCount == legacyEdgeCount);
    REQUIRE(fix.engine->GetActiveBank() == nullptr);
    REQUIRE(fix.metrics->GetActiveBank() == nullptr);
    REQUIRE(legacyDM == static_cast<double>(ctxDMScore) / (ctxEdgeCount + 0.1));

    fix.engine->SetActiveBank(nullptr);
    REQUIRE(fix.pool.Recycle(static_cast<std::size_t>(ctx_idx), true));
    fix.teardown();
}

// ===========================================================================
// TEST 4: Capture smoke — ProbeCapturableOpSet over ctx metric enqueues
// ===========================================================================

TEST_CASE(
    "U4 integration: ProbeCapturableOpSet over RenderPhase(ctx) + metric "
    "enqueues succeeds",
    "[u4][integration][gpu]") {
    using namespace gpu_cost_function;

    U4Fixture fix;
    REQUIRE(fix.setup());

    int ctx_idx = fix.pool.Checkout();
    REQUIRE(ctx_idx >= 0);
    EvaluationContext* ctx =
        fix.pool.context(static_cast<std::size_t>(ctx_idx));
    REQUIRE(ctx != nullptr);
    REQUIRE(fix.pool.IsInFlight(static_cast<std::size_t>(ctx_idx)));
    REQUIRE(hasContextResources(*ctx));

    // Build a callback that enqueues the full production operation set:
    //   RenderPhase(ctx) → FID enqueue → DM enqueue
    // All on ctx.stream — no sync, no D2H.
    struct CaptureCtx {
        RenderEngine* engine;
        EvaluationContext* ctx;
        GPUMetrics* metrics;
        GPUImage* renderedImage;
        GPUDilatedFrame* cf;
        GPUFrame* dm;
        int dilation;
    };
    CaptureCtx cctx;
    cctx.engine = fix.engine.get();
    cctx.ctx = ctx;
    cctx.metrics = fix.metrics.get();
    cctx.renderedImage = fix.engine->GetRenderOutput();
    cctx.cf = fix.cf.get();
    cctx.dm = fix.dm.get();
    cctx.dilation = kDilation;

    auto captureOp = [](void* stream_ptr, void* user_ctx) -> int {
        auto* c = static_cast<CaptureCtx*>(user_ctx);

        // Save original context stream and swap to the capture stream.
        void* original_stream = c->ctx->stream;
        c->ctx->stream = stream_ptr;

        // Use the EvaluationContext overload of RenderPhase (no internal sync).
        cudaError_t err = c->engine->RenderPhase(*c->ctx);
        if (err != cudaSuccess) {
            c->ctx->stream = original_stream;
            return static_cast<int>(err);
        }

        // Enqueue FID metric on the capture stream.
        err = c->metrics->EnqueueFastImplantDilationMetric(
            c->renderedImage, c->cf, c->dilation, *c->ctx);
        if (err != cudaSuccess) {
            c->ctx->stream = original_stream;
            return static_cast<int>(err);
        }

        // Enqueue DM metric on the capture stream.
        err = c->metrics->EnqueueDistanceMapMetric(
            c->renderedImage, c->dm, c->dilation, *c->ctx);
        if (err != cudaSuccess) {
            c->ctx->stream = original_stream;
            return static_cast<int>(err);
        }

        // Restore original stream and bank 0.
        c->ctx->stream = original_stream;
        c->engine->SetActiveBank(nullptr);

        return 0;
    };

    auto result = ProbeCapturableOpSet(captureOp, &cctx);
    INFO("U4 capture smoke: " << FormatPreflightResult(result));
    REQUIRE(result.capturable);
    REQUIRE(result.reasonCode == kPreflightOk);
    REQUIRE(result.failingNodeHint.empty());

    fix.engine->SetActiveBank(nullptr);
    REQUIRE(fix.pool.Recycle(static_cast<std::size_t>(ctx_idx), true));
    fix.teardown();
}

// ===========================================================================
// TEST 5: Edge/edge behavior — degenerate crop must not divide by zero
// ===========================================================================

TEST_CASE(
    "U4 integration: ComputeMetricCropKernel with zero width produces zeroed "
    "crop",
    "[u4][integration][gpu]") {
    using namespace gpu_cost_function;

    // Allocate a device MetricCropParams and a fake bounding box.
    MetricCropParams* dev_crop = nullptr;
    int* dev_bb = nullptr;
    cudaMalloc(&dev_crop, sizeof(MetricCropParams));
    cudaMalloc(&dev_bb, 4 * sizeof(int));
    REQUIRE(dev_crop != nullptr);
    REQUIRE(dev_bb != nullptr);

    // Set bounding box to something valid.
    int bb_host[4] = {5, 5, 20, 20};
    REQUIRE(
        cudaMemcpy(dev_bb, bb_host, 4 * sizeof(int), cudaMemcpyHostToDevice) ==
        cudaSuccess);

    // Launch ComputeMetricCropKernel with width=0, height=0.
    ComputeMetricCropKernel<<<1, 1>>>(dev_bb, kDilation, 0, 0, dev_crop);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    // Read back — all fields should be zero.
    MetricCropParams crop_host;
    REQUIRE(
        cudaMemcpy(
            &crop_host,
            dev_crop,
            sizeof(MetricCropParams),
            cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(crop_host.sub_cropped_width == 0);
    REQUIRE(crop_host.sub_cropped_height == 0);
    REQUIRE(crop_host.diff_cropped_width == 0);
    REQUIRE(crop_host.diff_cropped_height == 0);

    cudaFree(dev_crop);
    cudaFree(dev_bb);
}

// ===========================================================================
// TEST 6: Overflow flag comparison must not mark normal fills
// ===========================================================================

TEST_CASE(
    "U4 integration: overflow flag is 0 for normal (non-overflowing) render",
    "[u4][integration][gpu]") {
    using namespace gpu_cost_function;

    U4Fixture fix;
    REQUIRE(fix.setup());

    int ctx_idx = fix.pool.Checkout();
    REQUIRE(ctx_idx >= 0);
    EvaluationContext* ctx =
        fix.pool.context(static_cast<std::size_t>(ctx_idx));
    REQUIRE(ctx != nullptr);

    // RenderPhase on a small image with 1 triangle — should NOT overflow.
    REQUIRE(fix.pool.IsInFlight(static_cast<std::size_t>(ctx_idx)));
    REQUIRE(hasContextResources(*ctx));
    cudaError_t err = fix.engine->RenderPhase(*ctx);
    REQUIRE(err == cudaSuccess);
    err = fix.engine->CompleteRenderPhase(*ctx);
    REQUIRE(err == cudaSuccess);
    REQUIRE(fix.engine->GetActiveBank() == nullptr);

    // The overflow flag (pinned host twin) must be 0.
    int overflow = *static_cast<int*>(ctx->host_overflowFlag);
    REQUIRE(overflow == 0);

    fix.engine->SetActiveBank(nullptr);
    REQUIRE(fix.pool.Recycle(static_cast<std::size_t>(ctx_idx), true));
    fix.teardown();
}

// ===========================================================================
// TEST 7: EvaluationContext metric rejects null cf/dm gracefully
// ===========================================================================

TEST_CASE(
    "U4 integration: EvaluationContext FID metric rejects null cf",
    "[u4][integration][gpu]") {
    using namespace gpu_cost_function;

    U4Fixture fix;
    REQUIRE(fix.setup());

    int ctx_idx = fix.pool.Checkout();
    REQUIRE(ctx_idx >= 0);
    EvaluationContext* ctx =
        fix.pool.context(static_cast<std::size_t>(ctx_idx));
    REQUIRE(ctx != nullptr);
    REQUIRE(fix.pool.IsInFlight(static_cast<std::size_t>(ctx_idx)));
    REQUIRE(hasContextResources(*ctx));

    // Enqueue with null cf — must return error, not crash.
    cudaError_t err = fix.metrics->EnqueueFastImplantDilationMetric(
        fix.engine->GetRenderOutput(), nullptr, kDilation, *ctx);
    REQUIRE(err == cudaErrorInvalidValue);

    fix.engine->SetActiveBank(nullptr);
    REQUIRE(fix.pool.Recycle(static_cast<std::size_t>(ctx_idx), true));
    fix.teardown();
}

TEST_CASE(
    "U4 integration: EvaluationContext DM metric rejects null dm",
    "[u4][integration][gpu]") {
    using namespace gpu_cost_function;

    U4Fixture fix;
    REQUIRE(fix.setup());

    int ctx_idx = fix.pool.Checkout();
    REQUIRE(ctx_idx >= 0);
    EvaluationContext* ctx =
        fix.pool.context(static_cast<std::size_t>(ctx_idx));
    REQUIRE(ctx != nullptr);
    REQUIRE(fix.pool.IsInFlight(static_cast<std::size_t>(ctx_idx)));
    REQUIRE(hasContextResources(*ctx));

    cudaError_t err = fix.metrics->EnqueueDistanceMapMetric(
        fix.engine->GetRenderOutput(), nullptr, kDilation, *ctx);
    REQUIRE(err == cudaErrorInvalidValue);

    fix.engine->SetActiveBank(nullptr);
    REQUIRE(fix.pool.Recycle(static_cast<std::size_t>(ctx_idx), true));
    fix.teardown();
}
