/* Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/* U2: EvaluationContext — private mutable state per in-flight evaluation.
 * Covers R4/R6: null-init, half-memory admission with graph overhead, pool Checkout/Recycle.
 * Headless: CUDA-free, header-only (evaluation_context.h is opaque void* handles).
 */

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "compute/bank_state.cuh"
#include "compute/evaluation_context.h"

using gpu_cost_function::BankFootprintInput;
using gpu_cost_function::EvaluationContext;
using gpu_cost_function::EvaluationContextPool;
using gpu_cost_function::EvaluationStatus;

TEST_CASE("EvaluationContext null-init leaves every destructor-freed pointer null/false", "[evaluation_context]") {
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

TEST_CASE("EvaluationContext secondary empty for monoplane", "[evaluation_context]") {
    EvaluationContext ctx;
    REQUIRE(ctx.secondary.output == nullptr);
    REQUIRE(ctx.secondary.dev_projected_triangles == nullptr);
    REQUIRE(ctx.secondary.dev_cub_storage == nullptr);
}

TEST_CASE("BankAdmission half-memory includes graph overhead and device counters", "[evaluation_context]") {
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
    // U1: total includes 3*int device counters + 1*int host overflow + graph_overhead
    in.graph_overhead_bytes = 1024 * 1024; // 1 MB
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

TEST_CASE("EvaluationContextPool zero-work construction is safely destructible", "[evaluation_context]") {
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
    const bool ok = pool.Initialize(layout, 1024ULL * 1024 * 1024, 4);
    REQUIRE(ok);
    REQUIRE(pool.size() >= 1);
    REQUIRE(pool.context(0) != nullptr);
    // Zero-work pool has at least one correctly null-initted context
    const auto* ctx = pool.context(0);
    REQUIRE(ctx->stream == nullptr);
    REQUIRE(ctx->completion_event == nullptr);
}

TEST_CASE("EvaluationContextPool Checkout/Recycle mirrors BankCheckoutTracker", "[evaluation_context]") {
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
    REQUIRE((pool.Checkout() >= 0 || pool.size() == 2)); // third checkout fails if N=2
    REQUIRE(pool.IsInFlight(static_cast<std::size_t>(a)) == true);
    REQUIRE(pool.Recycle(static_cast<std::size_t>(a), false) == false);
    REQUIRE(pool.Recycle(static_cast<std::size_t>(a), true) == true);
    REQUIRE(pool.IsInFlight(static_cast<std::size_t>(a)) == false);
}

TEST_CASE("admission falls back to N=1 for insufficient free bytes", "[evaluation_context]") {
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
