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

#include <cstdint>

namespace gpu_cost_function {

EvaluationContextPool::~EvaluationContextPool() {
    Shutdown();
}

bool EvaluationContextPool::Initialize(
    const BankFootprintInput& layout,
    std::uint64_t free_device_bytes,
    std::size_t n_max) {
    Shutdown();

    // Graph object memory is added by U3/U5 through graph_overhead_bytes.
    // U1 uses the Stage-1 render/metric footprint plus its worker counters
    // and always keeps the compatibility floor of one context.
    const BankFootprint footprint = bank_state_math::footprint(layout);
    if (!footprint.valid || footprint.total_bytes == 0) {
        return false;
    }
    const BankAdmission admission =
        bank_state_math::admit(free_device_bytes, footprint, n_max);
    const std::size_t count =
        admission.bank_count == 0 ? 1 : admission.bank_count;

    contexts_.resize(count);
    checked_out_.assign(count, false);

    for (std::size_t i = 0; i < count; ++i) {
        EvaluationContext& ctx = contexts_[i];
        ctx.index = i;
        ctx.width = static_cast<int>(layout.width);
        ctx.height = static_cast<int>(layout.height);
        ctx.status = EvaluationStatus::Idle;
        ctx.in_flight = false;
        ctx.input_index = -1;
        ctx.initialized_correctly = false;

        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
            cudaSuccess) {
            Shutdown();
            return false;
        }
        ctx.stream = reinterpret_cast<void*>(stream);

        cudaEvent_t event = nullptr;
        if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) !=
            cudaSuccess) {
            Shutdown();
            return false;
        }
        ctx.completion_event = reinterpret_cast<void*>(event);

        if (cudaMalloc(&ctx.dev_nextCandidate, sizeof(std::int32_t)) !=
                cudaSuccess ||
            cudaMalloc(&ctx.dev_nextChunk, sizeof(std::int32_t)) !=
                cudaSuccess ||
            cudaMalloc(&ctx.dev_overflowFlag, sizeof(std::int32_t)) !=
                cudaSuccess ||
            cudaHostAlloc(
                &ctx.host_overflowFlag,
                sizeof(std::int32_t),
                cudaHostAllocDefault) != cudaSuccess) {
            Shutdown();
            return false;
        }

        *static_cast<std::int32_t*>(ctx.host_overflowFlag) = 0;
        if (cudaMemsetAsync(
                ctx.dev_nextCandidate, 0, sizeof(std::int32_t), stream) !=
                cudaSuccess ||
            cudaMemsetAsync(
                ctx.dev_nextChunk, 0, sizeof(std::int32_t), stream) !=
                cudaSuccess ||
            cudaMemsetAsync(
                ctx.dev_overflowFlag, 0, sizeof(std::int32_t), stream) !=
                cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            Shutdown();
            return false;
        }

        ctx.initialized_correctly = true;
    }

    return true;
}

void EvaluationContextPool::Shutdown() {
    for (auto& ctx : contexts_) {
        auto stream = reinterpret_cast<cudaStream_t>(ctx.stream);
        if (stream != nullptr) {
            // Every per-context allocation may still be referenced by queued
            // graph work. Drain the owning stream before destroying the graph
            // or releasing any of its mutable write set.
            cudaStreamSynchronize(stream);
        } else if (ctx.in_flight && ctx.completion_event != nullptr) {
            cudaEventSynchronize(
                reinterpret_cast<cudaEvent_t>(ctx.completion_event));
        }

        if (ctx.graph_exec != nullptr) {
            cudaGraphExecDestroy(
                reinterpret_cast<cudaGraphExec_t>(ctx.graph_exec));
            ctx.graph_exec = nullptr;
        }
        if (ctx.completion_event != nullptr) {
            cudaEventDestroy(
                reinterpret_cast<cudaEvent_t>(ctx.completion_event));
            ctx.completion_event = nullptr;
        }
        if (ctx.dev_nextCandidate != nullptr) {
            cudaFree(ctx.dev_nextCandidate);
            ctx.dev_nextCandidate = nullptr;
        }
        if (ctx.dev_nextChunk != nullptr) {
            cudaFree(ctx.dev_nextChunk);
            ctx.dev_nextChunk = nullptr;
        }
        if (ctx.dev_overflowFlag != nullptr) {
            cudaFree(ctx.dev_overflowFlag);
            ctx.dev_overflowFlag = nullptr;
        }
        if (ctx.host_overflowFlag != nullptr) {
            cudaFreeHost(ctx.host_overflowFlag);
            ctx.host_overflowFlag = nullptr;
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            ctx.stream = nullptr;
        }

        ctx.initialized_correctly = false;
        ctx.in_flight = false;
        ctx.status = EvaluationStatus::Idle;
        ctx.input_index = -1;
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
