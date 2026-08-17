/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */
/* @file cost_capacity_oracle_test.cpp
 *
 * Plan 010 U9 (Phase B).  GPU-labeled oracle for the REAL device/occupancy
 * queries in CostCapacityService.  NEVER headless: requires a CUDA device.
 * Run explicitly with: ctest --test-dir .build -L oracle -R cost_capacity
 *
 * Pins (integration, from the plan's U9 test scenarios):
 *  - On the real device, the occupancy query returns > 0 blocks/SM for a real
 *    kernel function.
 *  - The service's SAFE_CAP equals maximum_stride_size * (threads_per_block - 1)
 *    = 2.55e9, the host overflow guard's exact bound in render_engine.cu:803.
 */
#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "compute/cost_capacity_service.cuh"
#include "compute/cuda_launch_parameters.h"

namespace {

__global__ void capacity_probe_kernel(int* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 && out != nullptr) {
        *out = 1;
    }
}

}  // namespace

TEST_CASE("U9 oracle: SAFE_CAP equals the render_engine overflow guard's bound", "[capacity][gpu]") {
    using gpu_cost_function::CostCapacityService;
    using gpu_cost_function::isCapacityAvailable;
    using gpu_cost_function::safeCapBound;

    CostCapacityService service;
    REQUIRE(service.refreshDeviceSnapshot(0));  // device 0

    // isCapacityAvailable reflects a real, resolved device.
    REQUIRE(isCapacityAvailable(service.snapshot()));

    // The SAFE_CAP constant is computed once per device and cached in the
    // instance; it must equal maximum_stride_size * (threads_per_block - 1).
    const std::int64_t expected =
        static_cast<std::int64_t>(maximum_stride_size) * (threads_per_block - 1);
    REQUIRE(expected == 2550000000LL);  // 2.55e9
    REQUIRE(safeCapBound(service.snapshot()) == expected);
}

TEST_CASE("U9 oracle: occupancy query returns > 0 blocks/SM for a real kernel", "[capacity][gpu]") {
    using gpu_cost_function::CostCapacityService;

    CostCapacityService service;
    REQUIRE(service.refreshDeviceSnapshot(0));

    const void* kernel = reinterpret_cast<const void*>(&capacity_probe_kernel);
    const int block = service.occupancyOptimalBlockSize(kernel);
    REQUIRE(block > 0);
    REQUIRE(block <= 1024);  // maxThreadsPerBlock (CC 3.0+, guide Table 5)

    // The snapshot's grid sizing is usable with the occupancy-derived block.
    const auto grid = service.gridFor(4096, block);
    REQUIRE(grid.capacity_applicable);
    REQUIRE(static_cast<std::int64_t>(grid.grid_blocks) * grid.block_threads >= 4096);
}

TEST_CASE("U12 oracle: extra bank owns a stream and completion event", "[capacity][gpu]") {
    using gpu_cost_function::BankFootprintInput;
    using gpu_cost_function::CostCapacityService;

    CostCapacityService service;
    service.setSnapshot(gpu_cost_function::DeviceCapacitySnapshot{
        .sm_count = 80,
        .max_threads_per_sm = 2048,
        .safe_cap = 2550000000LL,
        .grid_dim_limit = 2147483647LL,
        .free_device_bytes = 64LL * 1024 * 1024,
        .per_bank_footprint_bytes = 0,
        .n_max = 2,
    });
    BankFootprintInput layout;
    layout.width = 1;
    layout.height = 1;
    layout.triangle_count = 1;
    layout.maximum_stride_size = 1;
    layout.cub_storage_bytes = 1;
    REQUIRE(service.ConfigurePool(layout, 2));
    REQUIRE(service.poolSize() == 2);
    // Extra bank allocation is lazy: it appears when checked out.
    REQUIRE(service.bankState(1) == nullptr);
    const int bank = service.CheckoutBank();
    REQUIRE(bank == 1);
    const auto* state = service.bankState(1);
    REQUIRE(state != nullptr);
    REQUIRE(state->stream != nullptr);
    REQUIRE(state->completion_event != nullptr);

    REQUIRE(service.bankInFlight(1));
    REQUIRE_FALSE(service.RecycleBank(1, false));
    REQUIRE(service.bankInFlight(1));
    REQUIRE(service.RecycleBank(1, true));
    REQUIRE_FALSE(service.bankInFlight(1));
}
