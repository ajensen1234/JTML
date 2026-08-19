/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*
 * U1: EvaluationContextPool — CUDA-aware implementation.
 * Null-init, destruction waits for stream/event, then frees.
 * This TU is the only place that includes cuda_runtime.
 */

#include "compute/evaluation_context.h"

#include <cuda_runtime.h>

namespace gpu_cost_function {

EvaluationContextPool::~EvaluationContextPool() {
    Shutdown();
}

bool EvaluationContextPool::Initialize(const BankFootprintInput& layout,
                                       std::uint64_t free_device_bytes,
                                       std::size_t n_max) {
    Shutdown();
    // Compute footprint (includes RenderBuffers + MetricBuffers)
    // Graph VRAM is probed separately in U3/U5; U1 admission uses the
    // Stage-1 math as the budget and floors at 1 (serial fallback).
    BankFootprint fp = bank_state_math::footprint(layout);
    if (!fp.valid || fp.total_bytes == 0) {
        return false;
    }
    BankAdmission adm = bank_state_math::admit(free_device_bytes, fp, n_max);
    std::size_t n = adm.bank_count;
    if (n == 0) n = 1;

    contexts_.resize(n);
    checked_out_.assign(n, false);
    for (std::size_t i = 0; i < n; ++i) {
        EvaluationContext& ctx = contexts_[i];
        ctx.index = i;
        ctx.width = static_cast<int>(layout.width);
        ctx.height = static_cast<int>(layout.height);
        ctx.status = EvaluationStatus::Idle;
        ctx.in_flight = false;
        ctx.stream = nullptr;
        ctx.completion_event = nullptr;
        ctx.graph_exec = nullptr;
        ctx.dev_nextCandidate = nullptr;
        ctx.dev_nextChunk = nullptr;
        ctx.dev_overflowFlag = nullptr;
        ctx.host_overflowFlag = nullptr;
        ctx.input_index = -1;
        ctx.initialized_correctly = false;

        // U1 does not yet allocate CUDA resources for worker counters;
        // allocation lands in the executor/graph recipe. We keep the
        // context correctly null-initted so a zero-work construction
        // is safely destructible (cudaFree(nullptr) is a no-op) and
        // initialized_correctly remains false until a later stage
        // successfully allocates.
        if (n == 1 && layout.triangle_count == 0) {
            ctx.initialized_correctly = true;
        }
    }
    // For U1 the pool is considered initialized if we created at least
    // one context, even without CUDA allocations (null-init correctness).
    // Real CUDA streams/events/graphExecs are created in U5/U6.
    if (!contexts_.empty()) {
        contexts_[0].initialized_correctly = true;
    }
    return true;
}

void EvaluationContextPool::Shutdown() {
    // Destruction waits for each context's stream/event before freeing,
    // mirroring jtml-heatmap-guard preconditions. U1 pool has no
    // allocations yet, but we implement the wait correctly for future
    // stages.
    for (auto& ctx : contexts_) {
        if (ctx.stream) {
            auto stream = reinterpret_cast<cudaStream_t>(ctx.stream);
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            ctx.stream = nullptr;
        }
        if (ctx.completion_event) {
            auto event = reinterpret_cast<cudaEvent_t>(ctx.completion_event);
            // Wait for event before destroying (if in-flight, synchronize)
            cudaEventSynchronize(event);
            cudaEventDestroy(event);
            ctx.completion_event = nullptr;
        }
        if (ctx.graph_exec) {
            auto exec = reinterpret_cast<cudaGraphExec_t>(ctx.graph_exec);
            cudaGraphExecDestroy(exec);
            ctx.graph_exec = nullptr;
        }
        if (ctx.dev_nextCandidate) {
            cudaFree(ctx.dev_nextCandidate);
            ctx.dev_nextCandidate = nullptr;
        }
        if (ctx.dev_nextChunk) {
            cudaFree(ctx.dev_nextChunk);
            ctx.dev_nextChunk = nullptr;
        }
        if (ctx.dev_overflowFlag) {
            cudaFree(ctx.dev_overflowFlag);
            ctx.dev_overflowFlag = nullptr;
        }
        if (ctx.host_overflowFlag) {
            cudaFreeHost(ctx.host_overflowFlag);
            ctx.host_overflowFlag = nullptr;
        }
        ctx.in_flight = false;
        ctx.status = EvaluationStatus::Idle;
    }
    contexts_.clear();
    checked_out_.clear();
}

std::size_t EvaluationContextPool::size() const {
    return contexts_.size();
}

EvaluationContext* EvaluationContextPool::context(std::size_t idx) {
    if (idx >= contexts_.size()) return nullptr;
    return &contexts_[idx];
}

const EvaluationContext* EvaluationContextPool::context(std::size_t idx) const {
    if (idx >= contexts_.size()) return nullptr;
    return &contexts_[idx];
}

int EvaluationContextPool::Checkout() {
    for (std::size_t i = 0; i < checked_out_.size(); ++i) {
        if (!checked_out_[i]) {
            checked_out_[i] = true;
            contexts_[i].in_flight = true;
            contexts_[i].status = EvaluationStatus::InFlight;
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool EvaluationContextPool::Recycle(std::size_t idx, bool completion_ready) {
    if (idx >= checked_out_.size() || !checked_out_[idx] || !completion_ready) return false;
    checked_out_[idx] = false;
    contexts_[idx].in_flight = false;
    contexts_[idx].status = EvaluationStatus::Idle;
    contexts_[idx].input_index = -1;
    return true;
}

bool EvaluationContextPool::IsInFlight(std::size_t idx) const {
    if (idx >= checked_out_.size()) return false;
    return checked_out_[idx];
}

}  // namespace gpu_cost_function
