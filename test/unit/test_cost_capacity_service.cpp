/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */
/* @file test_cost_capacity_service.cpp
 *
 * Plan 010 U9 (Phase B).  Headless unit tests for the PURE capacity sizing math
 * in cost_capacity_service.cuh (no CUDA runtime — injected fake device props).
 * The GPU-only occupancy/device queries are covered by the oracle test under
 * test/oracle/.
 */
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "compute/cost_capacity_service.cuh"

using gpu_cost_function::CapacityGrid;
using gpu_cost_function::DeviceCapacitySnapshot;
using gpu_cost_function::capacityGrid;
using gpu_cost_function::evalBankCount;
using gpu_cost_function::isCapacityAvailable;

namespace {

// Fake props: 80 SMs, 2048 threads/SM (the plan's U9 test-scenario fixture).
DeviceCapacitySnapshot fakeProps() {
    DeviceCapacitySnapshot s;
    s.sm_count = 80;
    s.max_threads_per_sm = 2048;
    s.safe_cap = 10000000LL * (256LL - 1);  // 2.55e9, the render_engine.cu:803 bound
    s.grid_dim_limit = 2147483647LL;        // 2^31-1 per axis (CC 3.0+)
    s.free_device_bytes = 24LL * 1024 * 1024 * 1024;  // ~24 GB (RTX 3090-class)
    s.per_bank_footprint_bytes = 75LL * 1024 * 1024;  // ~75 MB/bank (300k-triangle)
    s.n_max = 4;
    return s;
}

}  // namespace

TEST_CASE("capacityGrid returns the minimal covering grid (no up-pad)", "[capacity]") {
    const auto snap = fakeProps();  // 80 SMs, 2048 threads/SM

    // work <= resident capacity -> EXACTLY ceil(work/block), never padded up.
    const CapacityGrid g = capacityGrid(2048, 256, snap);
    REQUIRE(g.capacity_applicable);
    REQUIRE(g.block_threads == 256);
    REQUIRE(g.grid_blocks == 8);                    // 2048/256
    REQUIRE(static_cast<std::int64_t>(g.grid_blocks) * g.block_threads >= 2048);

    // A non-multiple: ceil.
    const CapacityGrid g2 = capacityGrid(1000, 256, snap);
    REQUIRE(g2.capacity_applicable);
    REQUIRE(g2.grid_blocks == 4);                   // ceil(1000/256) = 4
    REQUIRE(static_cast<std::int64_t>(g2.grid_blocks) * g2.block_threads >= 1000);
}

TEST_CASE("capacityGrid work-coverage invariant holds", "[capacity]") {
    const auto snap = fakeProps();
    for (std::int64_t work : {1LL, 255LL, 256LL, 257LL, 100000LL, 1300000LL, 2000000000LL}) {
        const CapacityGrid g = capacityGrid(work, 256, snap);
        REQUIRE(g.capacity_applicable);
        REQUIRE(g.grid_blocks >= 1);
        const std::int64_t covered =
            static_cast<std::int64_t>(g.grid_blocks) * g.block_threads;
        REQUIRE(covered >= work);               // work-coverage invariant
        // never above SAFE_CAP threads
        REQUIRE(covered <= snap.safe_cap);
        // never above the grid-dimension ceiling
        REQUIRE(g.grid_blocks <= snap.grid_dim_limit);
    }
}

TEST_CASE("capacityGrid SAFE_CAP clamps an over-capacity request to fallback", "[capacity]") {
    const auto snap = fakeProps();

    // Work exactly at SAFE_CAP stays coverable (grid*block == SAFE_CAP).
    const std::int64_t safe_work =
        static_cast<std::int64_t>(snap.safe_cap) / 256 * 256;
    const CapacityGrid at = capacityGrid(safe_work, 256, snap);
    REQUIRE(at.capacity_applicable);

    // Work beyond SAFE_CAP -> not applicable (caller uses today's sizing +
    // the existing synchronous overflow fallback, P2). No giant grid is produced.
    const CapacityGrid over = capacityGrid(3000000000LL, 256, snap);  // 3e9
    REQUIRE_FALSE(over.capacity_applicable);
}

TEST_CASE("capacityGrid strict-no-op for a formula that already equals minimal", "[capacity]") {
    const auto snap = fakeProps();
    // The exact minimal grid is produced (no padding); the caller's
    // "current formula already equals this" case is a strict no-op downstream
    // because grid*block == work (no overshoot introduced). Pin that equality
    // property so a padded grid can never slip in here.
    const CapacityGrid g = capacityGrid(8192, 256, snap);
    REQUIRE(g.capacity_applicable);
    REQUIRE(static_cast<std::int64_t>(g.grid_blocks) * g.block_threads == 8192);
    REQUIRE(g.grid_blocks == 32);
}

TEST_CASE("capacityGrid edge: work == 0 -> minimal 1-block launch", "[capacity]") {
    const auto snap = fakeProps();
    const CapacityGrid g = capacityGrid(0, 256, snap);
    REQUIRE(g.capacity_applicable);
    REQUIRE(g.grid_blocks == 1);
}

TEST_CASE("capacityGrid grid-dimension ceiling triggers fallback", "[capacity]") {
    auto snap = fakeProps();
    snap.grid_dim_limit = 1024;  // artificially tiny ceiling
    const CapacityGrid g = capacityGrid(300000LL, 256, snap);  // ceil ~1172 blocks
    REQUIRE_FALSE(g.capacity_applicable);
}

TEST_CASE("evalBankCount is memory-bounded with N_MAX clamp, floored at 1", "[capacity]") {
    auto snap = fakeProps();
    const int n = evalBankCount(snap);  // floor(24GB/75MB)=327 < N_MAX=4 -> 4
    REQUIRE(n == 4);
    REQUIRE(n >= 1);

    // Footprint larger than free memory -> floor at 1.
    snap.per_bank_footprint_bytes = 40LL * 1024 * 1024 * 1024;  // 40 GB > 24 GB
    REQUIRE(evalBankCount(snap) == 1);

    // Unmeasured / zero inputs -> serial identity N=1.
    snap.free_device_bytes = 0;
    REQUIRE(evalBankCount(snap) == 1);
}

TEST_CASE("capacity unavailable flag makes consumers fall back", "[capacity]") {
    const DeviceCapacitySnapshot empty{};  // zeroed -> unavailable
    REQUIRE_FALSE(isCapacityAvailable(empty));

    auto snap = fakeProps();
    snap.sm_count = 0;
    REQUIRE_FALSE(isCapacityAvailable(snap));
}
