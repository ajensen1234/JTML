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

#include <cmath>
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
    // Destroy executor-owned wrappers via destroy hook before pool teardown (C2),
    // EXCEPT poisoned contexts (U5/C8): their graph may be hung, so destroying
    // would block. Leak poisoned wrappers until process exit — pool_.Shutdown()
    // also skips their buffers. The DestroyHook for a poisoned idx is never run.
    for (std::size_t i = 0; i < graphExecs_.size(); ++i) {
        if (graphExecs_[i] != nullptr) {
            if (destroyHook_ && !pool_.IsPoisoned(i)) {
                destroyHook_(i);
            }
            graphExecs_[i] = nullptr;
        }
    }
    graphExecs_.clear();
    pool_.Shutdown();
    firstSubmission_.store(false);
}

std::size_t EvaluationExecutor::poolSize() const {
    return pool_.size();
}

void EvaluationExecutor::InstallPrepareHook(PrepareHookFn hook) {
    prepareHook_ = std::move(hook);
}

void EvaluationExecutor::InstallDestroyHook(DestroyHookFn hook) {
    destroyHook_ = std::move(hook);
}

void EvaluationExecutor::InstallEnqueueHook(EnqueueHookFn hook) {
    enqueueHook_ = std::move(hook);
}

void EvaluationExecutor::InstallPollHook(PollHookFn hook) {
    pollHook_ = std::move(hook);
}

void EvaluationExecutor::InstallCompleteFromPinsHook(CompleteFromPinsHookFn hook) {
    completeFromPinsHook_ = std::move(hook);
}

void EvaluationExecutor::InstallTeardownHook(TeardownHookFn hook) {
    teardownHook_ = std::move(hook);
}

void EvaluationExecutor::InstallPacingHook(PacingHookFn hook) {
    pacingHook_ = std::move(hook);
    pacingInstalled_ = true;
}

std::size_t EvaluationExecutor::graphExecsSize() const {
    return graphExecs_.size();
}

void* EvaluationExecutor::graphExecAt(std::size_t idx) const {
    if (idx >= graphExecs_.size()) return nullptr;
    return graphExecs_[idx];
}
std::size_t EvaluationExecutor::preparedContextCount() const {
    std::size_t c = 0;
    for (auto* p : graphExecs_) if (p != nullptr) ++c;
    return c;
}

BatchOutcome EvaluationExecutor::Prepare(const GraphRecipeKey& key, std::size_t count) {
    if (poisoned_.load()) {
        return BatchOutcome::WatchdogPoisoned("executor poisoned");
    }
    if (count == 0) {
        return BatchOutcome::Ordered({});
    }
    // Ensure graphExecs_ can hold any pool index up to pool_.size() and count
    std::size_t needed = std::max(count, pool_.size());
    if (graphExecs_.size() < needed) {
        graphExecs_.resize(needed, nullptr);
    }
    std::vector<std::size_t> successIdxs;
    successIdxs.reserve(count);
    std::vector<int> checkedOutIndices;
    checkedOutIndices.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        int idx = pool_.Checkout();
        if (idx < 0) {
            for (auto sIdx : successIdxs) {
                if (sIdx < graphExecs_.size() && graphExecs_[sIdx] != nullptr) {
                    if (destroyHook_) destroyHook_(sIdx);
                    graphExecs_[sIdx] = nullptr;
                }
            }
            for (int cIdx : checkedOutIndices) {
                pool_.ForceRelease(static_cast<std::size_t>(cIdx));
            }
            return BatchOutcome::NotSubmitted("Prepare: no free context");
        }
        checkedOutIndices.push_back(idx);
        if (!prepareHook_) {
            for (auto sIdx : successIdxs) {
                if (sIdx < graphExecs_.size() && graphExecs_[sIdx] != nullptr) {
                    if (destroyHook_) destroyHook_(sIdx);
                    graphExecs_[sIdx] = nullptr;
                }
            }
            for (int cIdx : checkedOutIndices) {
                pool_.ForceRelease(static_cast<std::size_t>(cIdx));
            }
            return BatchOutcome::NotSubmitted("Prepare: no prepare hook");
        }
        void* w = prepareHook_(static_cast<std::size_t>(idx), key);
        if (!w) {
            for (auto sIdx : successIdxs) {
                if (sIdx < graphExecs_.size() && graphExecs_[sIdx] != nullptr) {
                    if (destroyHook_) destroyHook_(sIdx);
                    graphExecs_[sIdx] = nullptr;
                }
            }
            for (int cIdx : checkedOutIndices) {
                pool_.ForceRelease(static_cast<std::size_t>(cIdx));
            }
            return BatchOutcome::NotSubmitted("Prepare: createGraph failed");
        }
        if (static_cast<std::size_t>(idx) >= graphExecs_.size()) {
            graphExecs_.resize(static_cast<std::size_t>(idx) + 1, nullptr);
        }
        // C2/C4 re-prepare safety: if a wrapper from an earlier Prepare lives at
        // this index (e.g. a generation change or a second Prepare), destroy the
        // stale one BEFORE overwriting — never leak a graph exec.
        if (graphExecs_[static_cast<std::size_t>(idx)] != nullptr) {
            if (destroyHook_) destroyHook_(static_cast<std::size_t>(idx));
            graphExecs_[static_cast<std::size_t>(idx)] = nullptr;
        }
        graphExecs_[static_cast<std::size_t>(idx)] = w;
        successIdxs.push_back(static_cast<std::size_t>(idx));
    }
    // All count contexts prepared successfully — recycle so contexts are idle-but-graph-ready (C4)
    for (int cIdx : checkedOutIndices) {
        pool_.Recycle(static_cast<std::size_t>(cIdx), true);
    }
    // Do not set firstSubmission_
    return BatchOutcome::Ordered({});
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
    if (poisoned_.load()) {
        return BatchOutcome::WatchdogPoisoned("executor poisoned");
    }
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
    if (poisoned_.load()) {
        return BatchOutcome::WatchdogPoisoned("executor poisoned");
    }
    if (!costWithIndex) {
        return BatchOutcome::NotSubmitted("null costWithIndex");
    }
    if (poses.empty()) {
        return BatchOutcome::Ordered({});
    }
    // Degenerate N<=1: serial fallback ONLY when no hooks installed (headless stub).
    // When hooks are installed (real graph), even N=1 must go through the hook-driven
    // greedy path to actually launch graphs (U7 N=1 overhead arm).
    const bool useHooksEarly = static_cast<bool>(enqueueHook_) && static_cast<bool>(pollHook_) && static_cast<bool>(completeFromPinsHook_);
    if (poolSize() <= 1 && !useHooksEarly) {
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
    // Greedy N>1 path — hook-driven when all hooks set (U4), else headless stub
    const bool useHooks = useHooksEarly;
    if (useHooks) {
        std::vector<double> result(poses.size(), 0.0);
        std::vector<Lease> inFlight;
        inFlight.reserve(poses.size());
        std::size_t nextPos = 0;
        auto watchdogStart = std::chrono::steady_clock::now();
        while (nextPos < poses.size() || !inFlight.empty()) {
            while (nextPos < poses.size()) {
                int idx = pool_.Checkout();
                if (idx < 0) break;
                EvaluationContext* ctx = pool_.context(static_cast<std::size_t>(idx));
                if (!ctx) {
                    if (firstSubmission_.load()) {
                        result.clear();
                        if (teardownHook_) teardownHook_();
                        for (auto &l : inFlight) pool_.ForceRelease(l.ctxIdx);
                        pool_.ForceRelease(static_cast<std::size_t>(idx));
                        return BatchOutcome::PostLaunchAbort("null context after checkout");
                    }
                    pool_.ForceRelease(static_cast<std::size_t>(idx));
                    for (auto &l : inFlight) pool_.ForceRelease(l.ctxIdx);
                    return BatchOutcome::NotSubmitted("null context before submission");
                }
                ctx->input_index = static_cast<int>(nextPos);
                ctx->status = EvaluationStatus::InFlight;
                ctx->in_flight = true;
                bool enqOk = enqueueHook_(static_cast<std::size_t>(idx), nextPos, poses[nextPos]);
                if (!enqOk) {
                    result.clear();
                    if (teardownHook_) teardownHook_();
                    for (auto &l : inFlight) pool_.ForceRelease(l.ctxIdx);
                    pool_.ForceRelease(static_cast<std::size_t>(idx));
                    if (firstSubmission_.load()) {
                        return BatchOutcome::PostLaunchAbort("enqueue/launch failed");
                    }
                    return BatchOutcome::NotSubmitted("enqueue failed before first submission");
                }
                firstSubmission_.store(true);
                inFlight.push_back(Lease{static_cast<std::size_t>(idx), nextPos});
                ++nextPos;
                if (inFlight.size() >= pool_.size()) break;
            }
            if (inFlight.empty()) {
                if (std::chrono::steady_clock::now() - watchdogStart > watchdogTimeout_) {
                    result.clear();
                    poisoned_.store(true);
                    return BatchOutcome::WatchdogPoisoned("watchdog expiry (no work in flight)");
                }
                std::this_thread::yield();
                continue;
            }
            // Poll ALL inFlight to allow OOO completion.
            std::vector<Lease> survivors;
            survivors.reserve(inFlight.size());
            bool anyDone = false;
            bool abort = false;
            BatchOutcome abortOutcome = BatchOutcome::PostLaunchAbort("poll error");
            for (auto cur : inFlight) {
                PollResult pr = pollHook_(cur.ctxIdx);
                if (pr == PollResult::Done) {
                    double s = completeFromPinsHook_(cur.ctxIdx);
                    if (!std::isfinite(s)) {
                        abort = true;
                        abortOutcome = BatchOutcome::PostLaunchAbort("overflow/non-finite");
                        // fall through to abort handling after polling all
                    } else {
                        result[cur.inputPos] = s;
                        pool_.Recycle(cur.ctxIdx, true);
                        anyDone = true;
                        watchdogStart = std::chrono::steady_clock::now();
                        continue;
                    }
                } else if (pr == PollResult::Error) {
                    abort = true;
                    abortOutcome = BatchOutcome::PostLaunchAbort("poll error");
                } else {
                    survivors.push_back(cur);
                }
            }
            if (abort) {
                result.clear();
                if (teardownHook_) teardownHook_();
                for (auto &l : inFlight) {
                    pool_.ForceRelease(l.ctxIdx);
                }
                return abortOutcome;
            }
            inFlight = std::move(survivors);
            if (!anyDone) {
                if (std::chrono::steady_clock::now() - watchdogStart > watchdogTimeout_) {
                    result.clear();
                    if (teardownHook_) teardownHook_();
                    for (auto &l : inFlight) pool_.LeavePoisoned(l.ctxIdx);
                    poisoned_.store(true);
                    return BatchOutcome::WatchdogPoisoned("watchdog expiry");
                }
                pacingHook_();   // Plan 013 U1: injectable bounded pacing (default yield, CUDA adaptive)
            }
        }
        if (result.size() != poses.size()) {
            throw std::invalid_argument("BatchCostFunction returned wrong-sized vector");
        }
        return BatchOutcome::Ordered(std::move(result));
    }
    // Legacy headless stub (null hooks) — preserve exact U1/U3 behavior
    std::vector<double> result(poses.size(), 0.0);
    std::vector<Lease> inFlight;
    inFlight.reserve(poses.size());
    std::size_t nextPos = 0;
    auto watchdogStart = std::chrono::steady_clock::now();
    while (nextPos < poses.size() || !inFlight.empty()) {
        while (nextPos < poses.size()) {
            int idx = pool_.Checkout();
            if (idx < 0) break;
            EvaluationContext* ctx = pool_.context(static_cast<std::size_t>(idx));
            if (!ctx) {
                pool_.Recycle(static_cast<std::size_t>(idx), false);
                if (firstSubmission_.load()) {
                    result.clear();
                    for (auto &l : inFlight) pool_.Recycle(l.ctxIdx, false);
                    return BatchOutcome::PostLaunchAbort("null context after checkout");
                }
                return BatchOutcome::NotSubmitted("null context before submission");
            }
            ctx->input_index = static_cast<int>(nextPos);
            ctx->status = EvaluationStatus::InFlight;
            ctx->in_flight = true;
            firstSubmission_.store(true);
            inFlight.push_back(Lease{static_cast<std::size_t>(idx), nextPos});
            ++nextPos;
            if (inFlight.size() >= pool_.size()) break;
        }
        if (inFlight.empty()) {
            if (std::chrono::steady_clock::now() - watchdogStart > watchdogTimeout_) {
                result.clear();
                return BatchOutcome::WatchdogPoisoned("watchdog expiry (no work in flight)");
            }
            std::this_thread::yield();
            continue;
        }
        Lease cur = inFlight.front();
        bool ok = pollOneLease(cur, result, costWithIndex, poses);
        if (!ok) {
            result.clear();
            for (auto &l : inFlight) {
                if (l.ctxIdx != cur.ctxIdx) pool_.Recycle(l.ctxIdx, false);
            }
            return BatchOutcome::PostLaunchAbort("pollOneLease failed");
        }
        inFlight.erase(inFlight.begin());
        watchdogStart = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() - watchdogStart > watchdogTimeout_) {
            result.clear();
            for (auto &l : inFlight) pool_.Recycle(l.ctxIdx, false);
            return BatchOutcome::WatchdogPoisoned("watchdog expiry");
        }
    }
    if (result.size() != poses.size()) {
        throw std::invalid_argument("BatchCostFunction returned wrong-sized vector");
    }
    return BatchOutcome::Ordered(std::move(result));

}

}  // namespace gpu_cost_function
