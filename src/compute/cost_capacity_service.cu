/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */
/* @file cost_capacity_service.cu
 *
 * Plan 010 U9 (Phase B).  Implements the pure capacity sizing math declared in
 * cost_capacity_service.cuh (free functions) plus the GPU-only device/occupancy
 * queries that populate the snapshot in the real service.
 *
 * The pure math has zero CUDA dependency and is exercised by the headless unit
 * test (test/unit/test_cost_capacity_service.cpp).  The real queries call
 * cudaGetDeviceProperties / cudaMemGetInfo / cudaOccupancyMaxPotentialBlockSize
 * and are covered by the GPU-labeled oracle test (test/oracle/).
 */
#include "compute/cost_capacity_service.cuh"

#include <cuda_runtime.h>

namespace gpu_cost_function {


bool CostCapacityService::refreshDeviceSnapshot(int device) {
    cudaDeviceProp props{};
    cudaError_t err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) {
        snap_ = DeviceCapacitySnapshot{};
        return false;
    }

    size_t free_bytes = 0, total_bytes = 0;
    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        snap_ = DeviceCapacitySnapshot{};
        return false;
    }

    snap_.sm_count = props.multiProcessorCount;
    snap_.max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    snap_.grid_dim_limit = props.maxGridSize[0];  // per-axis; x is the launch axis
    snap_.free_device_bytes = static_cast<std::int64_t>(free_bytes);
    // SAFE_CAP: the host overflow guard's exact bound in render_engine.cu
    // (fragment_fill_[0] > maximum_stride_size * (threads_per_block - 1)) --
    // one source of truth, computed here so the service carries the ceiling.
    snap_.safe_cap =
        static_cast<std::int64_t>(maximum_stride_size) * (threads_per_block - 1);
    snap_.n_max = 0;  // N_MAX numeric value is implementation-time (deep-dive estimate ~4)
    return isCapacityAvailable(snap_);
}

int CostCapacityService::occupancyOptimalBlockSize(const void* kernel_func) {
    if (kernel_func == nullptr) {
        return threads_per_block;
    }
    int min_grid = 0;
    int block = 0;
    cudaError_t err =
        cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, kernel_func, 0, 0);
    if (err != cudaSuccess || block <= 0) {
        return threads_per_block;  // fallback (P2)
    }
    return block;
}

}  // namespace gpu_cost_function
