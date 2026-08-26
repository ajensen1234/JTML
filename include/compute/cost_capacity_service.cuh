/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */
/* @file cost_capacity_service.cuh
 *
 * Plan 010 U9 (Phase B).  The capacity knowledge the optimizer queries, owned
 * by compute: a device snapshot, per-kernel occupancy, SAFE_CAP arithmetic,
 * capacity-clamped grid sizing, and the runtime eval-bank count.
 *
 * Design posture (see the plan's U9 approach):
 *  - This header carries ONLY pure sizing math + the plain data snapshot.  It
 *    deliberately pulls NO CUDA runtime headers, so the headless unit test
 *    (test/unit/test_cost_capacity_service.cpp) compiles it as plain C++
 *    against injected fake props.  The real CUDA device/occupancy queries live
 *    in src/compute/cost_capacity_service.cu.
 *  - The service owns no streams, no buffers, no allocations -- it is pure
 *    knowledge + sizing math.  Buffer/stream ownership lands with U12.
 *  - SAFE_CAP is a single source of truth with the host overflow guard in
 *    render_engine.cu (fragment_fill_[0] > maximum_stride_size *
 * (threads_per_block-1)).
 */
#ifndef COST_CAPACITY_SERVICE_CUH
#define COST_CAPACITY_SERVICE_CUH
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "compute/bank_state.cuh"  // U12 Stage 1 ownership contract
#include "cuda_launch_parameters.h"  // threads_per_block, maximum_stride_size (plain consts)
#include "domain/data_structures_6D.h"  // Point6D for the U11 batch seam

namespace gpu_cost_function {

class BankStatePool;  // CUDA-owned implementation in cost_capacity_service.cu

/* Plain device snapshot, headless-injectable. */
struct DeviceCapacitySnapshot {
    int sm_count = 0;            // cudaDevAttrMultiProcessorCount
    int max_threads_per_sm = 0;  // cudaDevAttrMaxThreadsPerMultiProcessor
    std::int64_t safe_cap = 0;  // maximum_stride_size * (threads_per_block - 1)
    std::int64_t grid_dim_limit = 0;  // max grid lifetime per axis (2^31 - 1)
    std::int64_t free_device_bytes = 0;  // cudaMemGetInfo free (queried in .cu)
    std::int64_t per_bank_footprint_bytes =
        0;          // U12 write-set footprint per bank
    int n_max = 0;  // numeric N_MAX ceiling (implementation-time)
};

/* A launch config produced by the capacity solver. */
struct CapacityGrid {
    int block_threads =
        256;  // occupancy-optimal B_opt (or threads_per_block fallback)
    int grid_blocks =
        1;  // minimal covering grid for `work` items at block_threads
    bool capacity_applicable =
        false;  // false => work exceeds SAFE_CAP: caller keeps today's
                // sizing + the existing synchronous overflow fallback (P2)
};

/* Pure capacity math.  Headless-testable; no CUDA dependency.  Inline so the
 * headless unit test compiles standalone (no .cu link) -- the .cu only carries
 * the GPU-only device/occupancy queries. */
inline bool isCapacityAvailable(const DeviceCapacitySnapshot& snap) {
    return snap.sm_count > 0 && snap.max_threads_per_sm > 0;
}

/* The SAFE_CAP constant: exactly the host overflow guard's bound in
 * render_engine.cu. */
inline std::int64_t safeCapBound(const DeviceCapacitySnapshot& snap) {
    return snap.safe_cap;
}

/* Minimal covering grid for `work_items` at `block_threads`.
 * Invariants (P3, deepened):
 *  - never a grid larger than the minimal covering grid (grid*block >= work,
 * never padded up);
 *  - exact-grid-wins below resident capacity;
 *  - never above SAFE_CAP threads (grid*block <= SAFE_CAP) and never above
 * grid_dim_limit;
 *  - work == 0 => 1 block;
 *  - work == SAFE_CAP exactly => unchanged (grid*block == SAFE_CAP);
 *  - if work > SAFE_CAP, capacity_applicable=false (caller falls back per P2).
 * grid_dim_limit <= 0 means "no explicit limit" (caller supplies a real value
 * from the device). */
inline CapacityGrid capacityGrid(
    std::int64_t work_items,
    int block_threads,
    const DeviceCapacitySnapshot& snap) {
    CapacityGrid result;
    result.block_threads =
        block_threads > 0 ? block_threads : threads_per_block;

    if (work_items <= 0) {
        result.grid_blocks = 1;
        result.capacity_applicable = true;
        return result;
    }

    const std::int64_t block = result.block_threads;
    const std::int64_t safe =
        snap.safe_cap > 0 ? snap.safe_cap : safeCapBound(snap);

    // Work beyond SAFE_CAP: cannot stay under the overflow bound -> fall back
    // (P2).
    if (safe > 0 && work_items > safe) {
        result.capacity_applicable = false;
        result.grid_blocks = 1;  // sentinel; caller ignores on !applicable
        return result;
    }

    // Minimal covering grid: ceil(work / block); never padded up beyond this.
    std::int64_t grid = (work_items + block - 1) / block;
    if (grid < 1) {
        grid = 1;
    }

    // Grid-dimension ceiling (2^31-1 per axis, CC 3.0+): a grid in one axis
    // above this is not launchable -> fall back.
    if (snap.grid_dim_limit > 0 && grid > snap.grid_dim_limit) {
        result.capacity_applicable = false;
        result.grid_blocks = 1;
        return result;
    }

    result.grid_blocks = static_cast<int>(grid);
    result.capacity_applicable = true;
    return result;
}

/* Runtime eval-bank count (origin R11, memory-bounded):
 * min(floor(free/per_bank), N_MAX), floored at 1.  Occupancy is a grid-sizing
 * input only -- it bounds concurrent kernels, not buffer allocation. */
inline int evalBankCount(const DeviceCapacitySnapshot& snap) {
    if (snap.free_device_bytes <= 0 || snap.per_bank_footprint_bytes <= 0) {
        return 1;  // unmeasured -> serial (N=1 identity)
    }
    std::int64_t n = snap.free_device_bytes / snap.per_bank_footprint_bytes;
    if (snap.n_max > 0) {
        n = std::min(n, static_cast<std::int64_t>(snap.n_max));
    }
    return n >= 1 ? static_cast<int>(n) : 1;
}

/* The service itself.  Constructed by the manager's GPU object graph in
 * Initialize (deep-dive R2-4).  `refreshDeviceSnapshot` performs the real CUDA
 * queries and is GPU-only (defined in the .cu); the pure sizing entry points
 * are inlined below so the headless unit test never links CUDA. */
class CostCapacityService {
public:
    CostCapacityService();
    ~CostCapacityService();

    CostCapacityService(const CostCapacityService&) = delete;
    CostCapacityService& operator=(const CostCapacityService&) = delete;

    bool refreshDeviceSnapshot(int device);
    int occupancyOptimalBlockSize(const void* kernel_func);

    CapacityGrid gridFor(std::int64_t work_items, int block_threads) const {
        return capacityGrid(work_items, block_threads, snap_);
    }

    int bankCount() const {
        return evalBankCount(snap_);
    }
    bool available() const {
        return isCapacityAvailable(snap_);
    }
    const DeviceCapacitySnapshot& snapshot() const {
        return snap_;
    }

    template <typename SinglePointEval>
    std::vector<double> RunCostBatch(
        const std::vector<Point6D>& poses,
        SinglePointEval eval) const {
        std::vector<double> out;
        out.reserve(poses.size());
        for (const auto& p : poses) {
            out.push_back(static_cast<double>(eval(p)));
        }
        return out;
    }

    void setSnapshot(const DeviceCapacitySnapshot& snap) {
        snap_ = snap;
    }

    /* U12 lifecycle: configure admission and lazily create extra banks. The
     * compatibility bank 0 remains owned by RenderEngine/GPUMetrics. */
    bool ConfigurePool(const BankFootprintInput& layout, std::size_t n_max);
    std::size_t poolSize() const;
    int CheckoutBank();
    bool RecycleBank(std::size_t index, bool completion_ready);
    bool bankInFlight(std::size_t index) const;
    BankState* bankState(std::size_t index);
    const BankState* bankState(std::size_t index) const;

    using SerialCost = std::function<double(const Point6D&)>;
    using BankCost = std::function<double(const Point6D&, BankState&)>;
    // CUDA-free status: 0 means cudaSuccess; nonzero aborts the batch.
    using BankEnqueue = std::function<int(const Point6D&, BankState&)>;
    using BankComplete = std::function<double(BankState&)>;

    /* Greedy U12 feeder. N=1/unsupported paths remain exact serial. */
    std::vector<double> RunCostBatchGreedy(
        const std::vector<Point6D>& poses,
        const SerialCost& serial_cost,
        const BankCost& bank_cost);

    /* Enqueue/complete variant: enqueue may return before GPU completion;
     * complete is called only after the bank completion query succeeds. */
    std::vector<double> RunCostBatchGreedy(
        const std::vector<Point6D>& poses,
        const SerialCost& serial_cost,
        const BankEnqueue& enqueue,
        const BankComplete& complete);

private:
    DeviceCapacitySnapshot snap_;
    std::unique_ptr<BankStatePool> pool_;
};

}  // namespace gpu_cost_function

#endif  // COST_CAPACITY_SERVICE_CUH
