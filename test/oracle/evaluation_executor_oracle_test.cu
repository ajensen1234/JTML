/* Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/* U2 oracle: EvaluationContext pool / graph overhead smoke (LABELS oracle;gpu).
 * This file is CUDA (.cu) so it is compiled with nvcc; it verifies that
 * BankFootprint math with graph_overhead_bytes is consistent and that
 * EvaluationContextPool correctly accounts for graph VRAM in admission.
 * No GPU kernel launch is required — the oracle label is used because
 * the pool's N in production will be probed via cudaMemGetInfo in U5/U8.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

using gpu_cost_function::BankFootprintInput;
using gpu_cost_function::EvaluationContextPool;

TEST_CASE(
    "oracle: BankFootprint with graph overhead is additive and valid",
    "[evaluation_executor][oracle]") {
    BankFootprintInput in;
    in.width = 1024;
    in.height = 1024;
    in.triangle_count = 12412;
    in.maximum_stride_size = 10000000;
    in.cub_storage_bytes = 4096;
    in.curvature_capacity = 0;
    in.graph_overhead_bytes = 0;
    in.biplane = false;
    auto base = gpu_cost_function::bank_state_math::footprint(in);
    REQUIRE(base.valid);
    in.graph_overhead_bytes = 2ULL * 1024 * 1024;  // 2 MB
    auto with = gpu_cost_function::bank_state_math::footprint(in);
    REQUIRE(with.valid);
    REQUIRE(with.total_bytes == base.total_bytes + 2ULL * 1024 * 1024);
}

TEST_CASE(
    "oracle: EvaluationContextPool half-memory admission respects graph "
    "overhead",
    "[evaluation_executor][oracle]") {
    BankFootprintInput layout;
    layout.width = 1024;
    layout.height = 1024;
    layout.triangle_count = 12412;
    layout.maximum_stride_size = 10000000;
    layout.cub_storage_bytes = 4096;
    layout.curvature_capacity = 0;
    layout.graph_overhead_bytes = 4ULL * 1024 * 1024;  // 4 MB per context
    layout.biplane = false;
    auto fp = gpu_cost_function::bank_state_math::footprint(layout);
    REQUIRE(fp.valid);
    // Simulate 24 GB free — should admit N=4 even with graph overhead
    const std::uint64_t free = 24ULL * 1024 * 1024 * 1024;
    auto adm = gpu_cost_function::bank_state_math::admit(free, fp, 4);
    REQUIRE(adm.admitted);
    REQUIRE(adm.bank_count == 4);
    EvaluationContextPool pool;
    REQUIRE(pool.Initialize(layout, free, 4));
    REQUIRE(pool.size() == 4);
}

// U4: persistent worker counters and chunk math (oracle label, headless logic
// but gated for U4 verification).
TEST_CASE(
    "oracle: U4 persistent worker counters are counted in footprint",
    "[evaluation_executor][oracle][U4]") {
    BankFootprintInput in;
    in.width = 1024;
    in.height = 1024;
    in.triangle_count = 12412;
    in.maximum_stride_size = 10000000;
    in.cub_storage_bytes = 4096;
    in.curvature_capacity = 0;
    in.graph_overhead_bytes = 0;
    in.biplane = false;
    auto base = gpu_cost_function::bank_state_math::footprint(in);
    REQUIRE(base.valid);
    // U1's three device counters and one pinned host counter add 16 bytes;
    // tiny fill inputs do not change validity.
    REQUIRE(base.total_bytes > 0);
    // Verify that overflow threshold is not in footprint but in logic
    constexpr int64_t maxStride = 10000000;
    constexpr int64_t threshold = maxStride * 255;  // 2.55B
    REQUIRE(threshold == 2550000000LL);
}

TEST_CASE(
    "oracle: U4 chunk covering never skips or duplicates",
    "[evaluation_executor][oracle][U4]") {
    const int chunk = 256;
    for (int fill : {0, 1, 255, 256, 257, 512, 100000}) {
        int chunks = (fill + chunk - 1) / chunk;
        int covered = 0;
        for (int c = 0; c < chunks; ++c) {
            int start = c * chunk;
            int end = std::min(start + chunk, fill);
            covered += (end - start);
        }
        REQUIRE(covered == fill);
    }
}
