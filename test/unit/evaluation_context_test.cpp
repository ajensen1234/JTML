/* Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/* U2: EvaluationContext — private mutable state per in-flight evaluation.
 * Covers R4/R6: null-init, half-memory admission, real CUDA ownership, and
 * Checkout/Recycle. These tests are compute-only and remain headless.
 */

#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

using gpu_cost_function::BankFootprintInput;
using gpu_cost_function::EvaluationContext;
using gpu_cost_function::EvaluationContextPool;
using gpu_cost_function::EvaluationStatus;

TEST_CASE(
    "EvaluationContext null-init leaves every destructor-freed pointer "
    "null/false",
    "[evaluation_context]") {
    EvaluationContext ctx;
    REQUIRE(ctx.stream == nullptr);
    REQUIRE(ctx.completion_event == nullptr);
    REQUIRE(ctx.graph_exec == nullptr);
    REQUIRE(ctx.dev_nextCandidate == nullptr);
    REQUIRE(ctx.dev_nextChunk == nullptr);
    REQUIRE(ctx.dev_overflowFlag == nullptr);
    REQUIRE(ctx.host_overflowFlag == nullptr);
    REQUIRE(ctx.primary.output == nullptr);
    REQUIRE(ctx.primary.host_bounding_box == nullptr);
    REQUIRE(ctx.metrics.host_pixel_score == nullptr);
    REQUIRE(ctx.status == EvaluationStatus::Idle);
    REQUIRE(ctx.in_flight == false);
    REQUIRE(ctx.initialized_correctly == false);
    REQUIRE(ctx.input_index == -1);
    REQUIRE(ctx.index == 0);
}

TEST_CASE(
    "EvaluationContext secondary empty for monoplane",
    "[evaluation_context]") {
    EvaluationContext ctx;
    REQUIRE(ctx.secondary.output == nullptr);
    REQUIRE(ctx.secondary.dev_projected_triangles == nullptr);
    REQUIRE(ctx.secondary.dev_cub_storage == nullptr);
}

TEST_CASE(
    "BankAdmission half-memory includes graph overhead and device counters",
    "[evaluation_context]") {
    BankFootprintInput in;
    in.width = 512;
    in.height = 512;
    in.triangle_count = 300000;
    in.maximum_stride_size = 10000000;
    in.cub_storage_bytes = 4096;
    in.curvature_capacity = 0;
    in.graph_overhead_bytes = 0;
    in.biplane = false;
    auto base = gpu_cost_function::bank_state_math::footprint(in);
    REQUIRE(base.valid);
    // U1: total includes 3*int device counters + 1*int host overflow +
    // graph_overhead
    in.graph_overhead_bytes = 1024 * 1024;  // 1 MB
    auto with_graph = gpu_cost_function::bank_state_math::footprint(in);
    REQUIRE(with_graph.valid);
    REQUIRE(with_graph.total_bytes == base.total_bytes + 1024 * 1024);
    // Half-memory admission clamps
    const std::uint64_t free = with_graph.total_bytes * 4;
    auto adm = gpu_cost_function::bank_state_math::admit(free, with_graph, 4);
    REQUIRE(adm.budget_bytes == free / 2);
    REQUIRE(adm.admitted);
    REQUIRE(adm.bank_count >= 1);
}

TEST_CASE(
    "EvaluationContextPool allocates real resources for zero-work layouts",
    "[evaluation_context][gpu]") {
    BankFootprintInput layout;
    layout.width = 1;
    layout.height = 1;
    layout.triangle_count = 0;
    layout.maximum_stride_size = 1;
    layout.cub_storage_bytes = 1;
    layout.curvature_capacity = 0;
    layout.graph_overhead_bytes = 0;
    layout.biplane = false;

    EvaluationContextPool pool;
    REQUIRE(pool.Initialize(layout, 1024ULL * 1024 * 1024, 4));
    REQUIRE(pool.size() >= 1);

    const auto* ctx = pool.context(0);
    REQUIRE(ctx != nullptr);
    REQUIRE(ctx->stream != nullptr);
    REQUIRE(ctx->completion_event != nullptr);
    REQUIRE(ctx->dev_nextCandidate != nullptr);
    REQUIRE(ctx->dev_nextChunk != nullptr);
    REQUIRE(ctx->dev_overflowFlag != nullptr);
    REQUIRE(ctx->host_overflowFlag != nullptr);
    REQUIRE(*static_cast<const std::int32_t*>(ctx->host_overflowFlag) == 0);
    REQUIRE(ctx->initialized_correctly);

    unsigned int stream_flags = 0;
    REQUIRE(
        cudaStreamGetFlags(
            reinterpret_cast<cudaStream_t>(ctx->stream), &stream_flags) ==
        cudaSuccess);
    REQUIRE((stream_flags & cudaStreamNonBlocking) != 0);

    std::int32_t device_value = -1;
    for (void* counter :
         {ctx->dev_nextCandidate, ctx->dev_nextChunk, ctx->dev_overflowFlag}) {
        REQUIRE(
            cudaMemcpy(
                &device_value,
                counter,
                sizeof(device_value),
                cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(device_value == 0);
        device_value = -1;
    }

    pool.Shutdown();
    REQUIRE(pool.size() == 0);
    pool.Shutdown();
    REQUIRE(pool.size() == 0);
}

TEST_CASE(
    "EvaluationContextPool Checkout/Recycle owns distinct CUDA contexts",
    "[evaluation_context][gpu]") {
    BankFootprintInput layout;
    layout.width = 512;
    layout.height = 512;
    layout.triangle_count = 300000;
    layout.maximum_stride_size = 10000000;
    layout.cub_storage_bytes = 4096;
    layout.curvature_capacity = 0;
    layout.graph_overhead_bytes = 0;
    layout.biplane = false;

    EvaluationContextPool pool;
    REQUIRE(pool.Initialize(layout, 24ULL * 1024 * 1024 * 1024, 4));
    REQUIRE(pool.size() >= 2);

    const int a = pool.Checkout();
    const int b = pool.Checkout();
    REQUIRE(a >= 0);
    REQUIRE(b >= 0);
    REQUIRE(a != b);

    const auto* context_a = pool.context(static_cast<std::size_t>(a));
    const auto* context_b = pool.context(static_cast<std::size_t>(b));
    REQUIRE(context_a != nullptr);
    REQUIRE(context_b != nullptr);
    REQUIRE(context_a->stream != nullptr);
    REQUIRE(context_b->stream != nullptr);
    REQUIRE(context_a->stream != context_b->stream);
    REQUIRE(context_a->completion_event != context_b->completion_event);
    REQUIRE(context_a->dev_nextCandidate != context_b->dev_nextCandidate);

    REQUIRE((pool.Checkout() >= 0 || pool.size() == 2));
    REQUIRE(pool.IsInFlight(static_cast<std::size_t>(a)));
    REQUIRE_FALSE(pool.Recycle(static_cast<std::size_t>(a), false));
    REQUIRE(pool.Recycle(static_cast<std::size_t>(a), true));
    REQUIRE_FALSE(pool.IsInFlight(static_cast<std::size_t>(a)));
}

TEST_CASE(
    "admission falls back to N=1 for insufficient free bytes",
    "[evaluation_context]") {
    BankFootprintInput in;
    in.width = 512;
    in.height = 512;
    in.triangle_count = 300000;
    in.maximum_stride_size = 10000000;
    in.cub_storage_bytes = 4096;
    in.curvature_capacity = 0;
    in.graph_overhead_bytes = 0;
    in.biplane = false;
    auto fp = gpu_cost_function::bank_state_math::footprint(in);
    REQUIRE(fp.valid);
    auto adm = gpu_cost_function::bank_state_math::admit(fp.total_bytes, fp, 4);
    REQUIRE(adm.bank_count == 1);
    REQUIRE_FALSE(adm.admitted);
}

TEST_CASE(
    "ForceRelease makes context checkout-able again; LeavePoisoned does not",
    "[evaluation_context][lease]") {
    EvaluationContextPool pool;
    pool.InitForTest(4);
    int a = pool.Checkout();
    REQUIRE(a >= 0);
    REQUIRE(pool.ForceRelease(static_cast<std::size_t>(a)));
    REQUIRE_FALSE(pool.IsPoisoned(static_cast<std::size_t>(a)));
    int b = pool.Checkout();
    REQUIRE(b >= 0);
    REQUIRE_FALSE(pool.IsPoisoned(static_cast<std::size_t>(b)));
    int c = pool.Checkout();
    REQUIRE(c >= 0);
    REQUIRE(pool.LeavePoisoned(static_cast<std::size_t>(c)));
    REQUIRE(pool.IsPoisoned(static_cast<std::size_t>(c)));
    // c is poisoned, further checkout should skip it
    int d = pool.Checkout();
    REQUIRE(d >= 0);
    REQUIRE(d != c);
}
