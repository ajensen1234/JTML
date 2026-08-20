/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*
 * U6: EvaluationExecutor — greedy batch wiring + ordered result assembly.
 * CUDA-free headless implementation. GPU path (cudaGraphLaunch / EventQuery)
 * is stubbed as immediate success for headless; real CUDA polling lives in
 * evaluation_executor.cu when GPU is available. This keeps ctest -L headless
 * green without linking CUDA, while preserving ordered store and watchdog.
 */

#include "compute/evaluation_executor.h"

#include <stdexcept>
#include <chrono>
#include <thread>

namespace gpu_cost_function {

EvaluationExecutor::~EvaluationExecutor() {
    Shutdown();
}

bool EvaluationExecutor::Initialize(const BankFootprintInput& layout,
                                    std::uint64_t free_device_bytes,
                                    std::size_t n_max) {
    return pool_.Initialize(layout, free_device_bytes, n_max);
}

void EvaluationExecutor::Shutdown() {
    pool_.Shutdown();
    firstSubmission_.store(false);
}

std::size_t EvaluationExecutor::poolSize() const {
    return pool_.size();
}

bool EvaluationExecutor::pollOneLease(const Lease& lease,
                                      std::vector<double>& result,
                                      const std::function<double(const Point6D&, std::size_t)>& costWithIndex,
                                      const std::vector<Point6D>& poses) {
    // Headless path: immediate success, no CUDA. In GPU build this would do
    // cudaEventQuery / cudaStreamQuery and discriminate NotReady vs real error,
    // with watchdog. We simulate success and ordered store.
    // Real CUDA error injection for testing can be done by making cost throw.
    try {
        double v = costWithIndex(poses[lease.inputPos], lease.inputPos);
        result[lease.inputPos] = v;
    } catch (...) {
        return false;
    }
    // Recycle the context — completion_ready = true
    if (!pool_.Recycle(lease.ctxIdx, true)) {
        return false;
    }
    return true;
}

BatchOutcome EvaluationExecutor::RunBatch(
    const std::vector<Point6D>& poses,
    const std::function<double(const Point6D&)>& serialCost) {
    if (!serialCost) {
        return BatchOutcome::NotSubmitted("null serialCost");
    }
    auto costWithIndex = [&](const Point6D& p, std::size_t) -> double {
        return serialCost(p);
    };
    return RunBatchWithCost(poses, costWithIndex);
}

BatchOutcome EvaluationExecutor::RunBatchWithCost(
    const std::vector<Point6D>& poses,
    const std::function<double(const Point6D&, std::size_t)>& costWithIndex) {
    if (!costWithIndex) {
        return BatchOutcome::NotSubmitted("null costWithIndex");
    }
    if (poses.empty()) {
        return BatchOutcome::Ordered({});
    }
    // Degenerate N<=1: serial fallback, still ordered, still respects firstSubmission
    // but does not exercise greedy. This keeps headless deterministic and
    // preserves the serial adapter semantics for trivial batches (R1,R2,R11).
    if (poolSize() <= 1) {
        std::vector<double> out;
        out.reserve(poses.size());
        for (std::size_t i = 0; i < poses.size(); ++i) {
            // First submission flag still set on first pose for R7/R8 semantics
            if (!firstSubmission_.exchange(true)) {
                // first time
            }
            double v = costWithIndex(poses[i], i);
            out.push_back(v);
        }
        if (out.size() != poses.size()) {
            throw std::invalid_argument("BatchCostFunction returned wrong-sized vector");
        }
        return BatchOutcome::Ordered(std::move(out));
    }

    // Greedy N>1 path
    std::vector<double> result(poses.size(), 0.0);
    std::vector<Lease> inFlight;
    inFlight.reserve(poses.size());
    std::size_t nextPos = 0;
    auto watchdogStart = std::chrono::steady_clock::now();

    // For headless determinism, we finish in input order but simulate
    // out-of-order completion by polling oldest first; result is still
    // stored at input index, so out-of-order does not affect ordering.
    while (nextPos < poses.size() || !inFlight.empty()) {
        // Try to checkout as many as possible up to pool size
        while (nextPos < poses.size()) {
            int idx = pool_.Checkout();
            if (idx < 0) break;
            EvaluationContext* ctx = pool_.context(static_cast<std::size_t>(idx));
            if (!ctx) {
                pool_.Recycle(static_cast<std::size_t>(idx), false);
                // Real CUDA error after firstSubmission should abort
                if (firstSubmission_.load()) {
                    // clear ordered result vector, wait for all streams/events (noop headless), abort
                    result.clear();
                    for (auto &l : inFlight) pool_.Recycle(l.ctxIdx, false);
                    return BatchOutcome::PostLaunchAbort("null context after checkout");
                }
                return BatchOutcome::NotSubmitted("null context before submission");
            }
            ctx->input_index = static_cast<int>(nextPos);
            ctx->status = EvaluationStatus::InFlight;
            ctx->in_flight = true;
            // Simulate graph param update + launch
            // In GPU build this would be: registry.FindEligible->updateParams + cudaGraphLaunch
            // Here we just mark firstSubmission
            firstSubmission_.store(true);
            inFlight.push_back(Lease{static_cast<std::size_t>(idx), nextPos});
            ++nextPos;
            if (inFlight.size() >= pool_.size()) break;
        }

        if (inFlight.empty()) {
            // No work in flight but still have pending poses — should not happen
            // Check watchdog
            if (std::chrono::steady_clock::now() - watchdogStart > watchdogTimeout_) {
                result.clear();
                return BatchOutcome::WatchdogPoisoned("watchdog expiry (no work in flight)");
            }
            std::this_thread::yield();
            continue;
        }

        // Poll oldest lease first (headless immediate success). Real GPU would
        // loop over inFlight with cudaEventQuery and handle NotReady vs error.
        Lease cur = inFlight.front();
        // Simulate EventQuery: always success in headless (no NotReady)
        bool ok = pollOneLease(cur, result, costWithIndex, poses);
        if (!ok) {
            // Real CUDA error after firstSubmission -> abort batch, clear vector
            result.clear();
            // Wait for all streams/events (headless noop) and recycle remaining
            for (auto &l : inFlight) {
                if (l.ctxIdx != cur.ctxIdx) pool_.Recycle(l.ctxIdx, false);
            }
            return BatchOutcome::PostLaunchAbort("pollOneLease failed");
        }
        inFlight.erase(inFlight.begin());
        watchdogStart = std::chrono::steady_clock::now();

        // Watchdog check for hangs
        if (std::chrono::steady_clock::now() - watchdogStart > watchdogTimeout_) {
            result.clear();
            for (auto &l : inFlight) pool_.Recycle(l.ctxIdx, false);
            return BatchOutcome::WatchdogPoisoned("watchdog expiry");
        }
    }

    if (result.size() != poses.size()) {
        throw std::invalid_argument("BatchCostFunction returned wrong-sized vector");
    }
    // Determinism stress: result must be input-ordered regardless of completion order
    return BatchOutcome::Ordered(std::move(result));
}

}  // namespace gpu_cost_function
