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

enum class PollResult { Pending, Done, Error };

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

    // Plan 012 U3 C2/C5: executor-owned wrappers + CUDA-free hook seam
    using PrepareHookFn = std::function<void*(std::size_t idx, const GraphRecipeKey& key)>;
    using DestroyHookFn = std::function<void(std::size_t idx)>;
    // Plan 012 U4 (C5/C6): hook-driven greedy feeder + no-sync completion.
    // Header stays CUDA-free (no cuda_runtime.h). .cu installs real CUDA hooks.
    using EnqueueHookFn = std::function<bool(std::size_t ctxIdx, std::size_t inputPos, const Point6D& pose)>;
    using PollHookFn = std::function<PollResult(std::size_t ctxIdx)>;
    using CompleteFromPinsHookFn = std::function<double(std::size_t ctxIdx)>;
    using TeardownHookFn = std::function<void()>;
    void InstallEnqueueHook(EnqueueHookFn hook);
    void InstallPollHook(PollHookFn hook);
    void InstallCompleteFromPinsHook(CompleteFromPinsHookFn hook);
    void InstallTeardownHook(TeardownHookFn hook);
    void InstallPrepareHook(PrepareHookFn hook);
    void InstallDestroyHook(DestroyHookFn hook);
    std::size_t graphExecsSize() const;
    void* graphExecAt(std::size_t idx) const;
    std::size_t preparedContextCount() const;
    BatchOutcome Prepare(const GraphRecipeKey& key, std::size_t count);

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
    std::vector<void*> graphExecs_{};
    PrepareHookFn prepareHook_{};
    DestroyHookFn destroyHook_{};
    EnqueueHookFn enqueueHook_{};
    PollHookFn pollHook_{};
    CompleteFromPinsHookFn completeFromPinsHook_{};
    TeardownHookFn teardownHook_{};
};


// Plan 012 U4 (C5): CUDA feeder hooks installer defined in .cu.
// Headless tests link without .cu, so this symbol is only required
// when the .cu TU is linked (coordinator links full jtml_compute).
void InstallCudaFeederHooks(EvaluationExecutor& exec);

}  // namespace gpu_cost_function
