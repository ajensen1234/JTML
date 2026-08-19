/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */
/* Plan 010 U12 Stage 1: allocation-free bank-state contract and footprint math. */
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

#include "compute/bank_state.cuh"

using gpu_cost_function::BankFootprintInput;
using gpu_cost_function::BankState;
using gpu_cost_function::SharedReadOnlyGeometry;
using gpu_cost_function::bank_state_math::admit;
using gpu_cost_function::bank_state_math::footprint;

namespace {
BankFootprintInput fixture(bool biplane = false) {
    BankFootprintInput in;
    in.width = 1024;
    in.height = 1024;
    in.triangle_count = 300000;
    in.maximum_stride_size = 10000000;
    in.cub_storage_bytes = 4096;
    in.curvature_capacity = 0;
    in.biplane = biplane;
    return in;
}
}  // namespace

TEST_CASE("bank state represents the complete private write-set", "[bank_state]") {
    BankState bank;
    REQUIRE(bank.primary.output == nullptr);
    REQUIRE(bank.primary.host_bounding_box == nullptr);
    REQUIRE(bank.primary.dev_backface == nullptr);
    REQUIRE(bank.primary.dev_transformed_vertex_zs == nullptr);
    REQUIRE(bank.primary.dev_tangent_triangle == nullptr);
    REQUIRE(bank.primary.dev_projected_triangles == nullptr);
    REQUIRE(bank.primary.dev_projected_triangles_snapped == nullptr);
    REQUIRE(bank.primary.dev_bounding_box_triangles == nullptr);
    REQUIRE(bank.primary.dev_bounding_box_triangles_sizes == nullptr);
    REQUIRE(bank.primary.dev_bounding_box_triangles_sizes_prefix == nullptr);
    REQUIRE(bank.primary.dev_bounding_box == nullptr);
    REQUIRE(bank.primary.dev_fragment_fill == nullptr);
    REQUIRE(bank.primary.host_fragment_fill == nullptr);
    REQUIRE(bank.primary.dev_stride_prefixes == nullptr);
    REQUIRE(bank.primary.dev_cub_storage == nullptr);
    REQUIRE(bank.metrics.host_pixel_score == nullptr);
    REQUIRE(bank.metrics.dev_pixel_score == nullptr);
    REQUIRE(bank.metrics.host_intersection == nullptr);
    REQUIRE(bank.metrics.host_union == nullptr);
    REQUIRE(bank.metrics.dev_intersection == nullptr);
    REQUIRE(bank.metrics.dev_union == nullptr);
    REQUIRE(bank.metrics.host_white_count == nullptr);
    REQUIRE(bank.metrics.dev_white_count == nullptr);
    REQUIRE(bank.metrics.host_distance_score == nullptr);
    REQUIRE(bank.metrics.dev_distance_score == nullptr);
    REQUIRE(bank.metrics.host_edge_count == nullptr);
    REQUIRE(bank.metrics.dev_edge_count == nullptr);
    REQUIRE(bank.metrics.host_curvature == nullptr);
    REQUIRE(bank.metrics.dev_curvature == nullptr);
}

TEST_CASE("shared triangles and normals are excluded from footprint", "[bank_state]") {
    SharedReadOnlyGeometry geometry;
    geometry.triangles = reinterpret_cast<const void*>(0x1);
    geometry.normals = reinterpret_cast<const void*>(0x2);
    const auto mono = footprint(fixture(false));
    REQUIRE(mono.valid);
    REQUIRE(mono.total_bytes > 0);
    // The layout has no geometry fields, so changing only shared geometry cannot
    // affect the pure write-set accounting.
    REQUIRE(mono.total_bytes == footprint(fixture(false)).total_bytes);
    REQUIRE(geometry.triangles != nullptr);
    REQUIRE(geometry.normals != nullptr);
}

TEST_CASE("biplane doubles the render write-set but not shared metrics", "[bank_state]") {
    const auto mono = footprint(fixture(false));
    const auto bi = footprint(fixture(true));
    REQUIRE(mono.valid);
    REQUIRE(bi.valid);
    REQUIRE(bi.render_bytes == mono.render_bytes * 2);
    REQUIRE(bi.metric_bytes == mono.metric_bytes);
    // U1: total includes per-context counters (nextCandidate/nextChunk/overflowFlag) + host overflow + graph overhead
    REQUIRE(bi.total_bytes == mono.render_bytes * 2 + mono.metric_bytes + 3 * sizeof(std::int32_t) + 1 * sizeof(std::int32_t));
}

TEST_CASE("footprint arithmetic includes dimensions, scratch, metrics, curvature", "[bank_state]") {
    auto base = fixture();
    auto with_curvature = base;
    with_curvature.curvature_capacity = 17;
    auto with_scratch = base;
    with_scratch.cub_storage_bytes += 4096;

    const auto a = footprint(base);
    const auto b = footprint(with_curvature);
    const auto c = footprint(with_scratch);
    REQUIRE(a.valid);
    REQUIRE(b.valid);
    REQUIRE(c.valid);
    REQUIRE(b.total_bytes > a.total_bytes);
    REQUIRE(c.total_bytes > a.total_bytes);
    REQUIRE(a.render_bytes_per_camera > a.metric_bytes);
}

TEST_CASE("admission uses half free memory and clamps N_MAX", "[bank_state]") {
    BankFootprintInput in = fixture();
    in.width = 1;
    in.height = 1;
    in.triangle_count = 1;
    in.maximum_stride_size = 1;
    in.cub_storage_bytes = 1;
    const auto fp = footprint(in);
    REQUIRE(fp.valid);

    const std::uint64_t free_bytes = fp.total_bytes * 10;
    const auto admitted = admit(free_bytes, fp, 3);
    REQUIRE(admitted.budget_bytes == free_bytes / 2);
    REQUIRE(admitted.fitting_banks == 5);
    REQUIRE(admitted.bank_count == 3);
    REQUIRE(admitted.admitted);
}

TEST_CASE("admission falls back to N=1 for unavailable or insufficient capacity", "[bank_state]") {
    const auto fp = footprint(fixture());
    REQUIRE(fp.valid);
    REQUIRE(admit(0, fp, 4).bank_count == 1);
    REQUIRE_FALSE(admit(0, fp, 4).admitted);
    REQUIRE(admit(fp.total_bytes * 2 - 1, fp, 4).bank_count == 1);
    REQUIRE(admit(fp.total_bytes * 2, fp, 4).bank_count == 1);

    auto invalid = fp;
    invalid.valid = false;
    REQUIRE(admit(fp.total_bytes * 100, invalid, 4).bank_count == 1);
    REQUIRE(admit(fp.total_bytes * 100, fp, 0).bank_count == 1);
}

TEST_CASE("footprint rejects overflowing dimensions and admission arithmetic", "[bank_state]") {
    auto in = fixture();
    in.width = std::numeric_limits<std::uint64_t>::max();
    in.height = 2;
    const auto fp = footprint(in);
    REQUIRE_FALSE(fp.valid);

    auto valid = footprint(fixture());
    REQUIRE(valid.valid);
    const auto saturated = admit(std::numeric_limits<std::uint64_t>::max(), valid,
                                 std::numeric_limits<std::uint64_t>::max());
    REQUIRE(saturated.bank_count >= 1);
    REQUIRE(saturated.budget_bytes == std::numeric_limits<std::uint64_t>::max() / 2);
}

TEST_CASE("bank checkout is unique and recycle waits for completion", "[bank_state]") {
    gpu_cost_function::BankCheckoutTracker tracker(2);
    const int first = tracker.checkout();
    const int second = tracker.checkout();
    REQUIRE(first != second);
    REQUIRE(first >= 0);
    REQUIRE(second >= 0);
    REQUIRE(tracker.checkout() == -1);
    REQUIRE_FALSE(tracker.recycle(static_cast<std::size_t>(first), false));
    REQUIRE(tracker.checkedOut(static_cast<std::size_t>(first)));
    REQUIRE(tracker.recycle(static_cast<std::size_t>(first), true));
    REQUIRE_FALSE(tracker.checkedOut(static_cast<std::size_t>(first)));
    REQUIRE(tracker.recycle(static_cast<std::size_t>(second), true));
    REQUIRE(tracker.checkout() >= 0);
}
