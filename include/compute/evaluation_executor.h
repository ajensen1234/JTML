/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*
 * U6: EvaluationExecutor — greedy batch wiring + ordered result assembly.
 * Owns EvaluationContextPool and GraphRecipeRegistry. RunBatch is the sole
 * domain-facing entry point: synchronous to DirectOptimizer, greedy internally.
 * Header is CUDA-free (opaque handles, no cuda_runtime).
 */

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <vector>

#include "compute/bank_state.cuh"
#include "compute/evaluation_context.h"
#include "compute/graph_recipe.h"
#include "compute/batch_outcome.h"
#include "domain/data_structures_6D.h"

namespace gpu_cost_function {

class EvaluationExecutor {
public:
    EvaluationExecutor() = default;
    ~EvaluationExecutor();

    EvaluationExecutor(const EvaluationExecutor&) = delete;
    EvaluationExecutor& operator=(const EvaluationExecutor&) = delete;

    // Initialize pool for given layout. free_device_bytes and n_max follow
    // bank_state_math::admit half-memory; graphOverhead is included in layout.
    bool Initialize(const BankFootprintInput& layout,
                    std::uint64_t free_device_bytes,
                    std::size_t n_max);
    void Shutdown();

    std::size_t poolSize() const;
    EvaluationContextPool& pool() { return pool_; }
    const EvaluationContextPool& pool() const { return pool_; }

    GraphRecipeRegistry& registry() { return registry_; }
    const GraphRecipeRegistry& registry() const { return registry_; }

    // Domain-facing greedy batch. Serial cost is the BuildGpuCostAdapter
    // closure (Point6D -> double). Returns typed BatchOutcome (plan 012 U1):
    // OrderedScores for success (including empty input), NotSubmitted for
    // pre-submission failure, PostLaunchAbort/WatchdogPoisoned for R7 after
    // firstSubmission. Throws std::invalid_argument only for true wrong-size
    // contract violation from an injected cost (never as abort sentinel).
    BatchOutcome RunBatch(const std::vector<Point6D>& poses,
                          const std::function<double(const Point6D&)>& serialCost);

    // Overload for testing: inject per-pose cost with index, to simulate
    // out-of-order completions while still preserving ordered store.
    BatchOutcome RunBatchWithCost(
        const std::vector<Point6D>& poses,
        const std::function<double(const Point6D&, std::size_t)>& costWithIndex);

    // Watchdog timeout for EventQuery polling. Default 5s.
    void setWatchdogTimeout(std::chrono::milliseconds t) { watchdogTimeout_ = t; }
    std::chrono::milliseconds watchdogTimeout() const { return watchdogTimeout_; }

    // For testing: force firstSubmission state.
    bool firstSubmission() const { return firstSubmission_.load(); }
    void resetFirstSubmission() { firstSubmission_.store(false); }

private:
    struct Lease {
        std::size_t ctxIdx = 0;
        std::size_t inputPos = 0;
    };

    // Polls for completion of one lease. Returns true on success (recycles),
    // false on real error or timeout (caller should abort batch).
    bool pollOneLease(const Lease& lease, std::vector<double>& result,
                      const std::function<double(const Point6D&, std::size_t)>& costWithIndex,
                      const std::vector<Point6D>& poses);

    EvaluationContextPool pool_{};
    GraphRecipeRegistry registry_{};
    std::atomic<bool> firstSubmission_{false};
    std::chrono::milliseconds watchdogTimeout_{5000};
};

}  // namespace gpu_cost_function
