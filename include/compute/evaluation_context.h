/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*
 * U1: EvaluationContext — private mutable state per in-flight evaluation.
 * Each context owns its stream, event, graph Exec, and write sets.
 * BankState is retained as Stage-1 math/alias compatibility; EvaluationContext
 * is the primary executed type for the graph-backed greedy executor.
 * Header is CUDA-free (opaque void* handles) so headless unit tests can
 * include it without linking CUDA.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "compute/bank_state.cuh"

namespace gpu_cost_function {

enum class EvaluationStatus { Idle = 0, InFlight, Ready, Failed };

struct EvaluationContext {
    std::size_t index = 0;
    int width = 0;
    int height = 0;
    RenderBuffers primary{};
    RenderBuffers secondary{};  // empty for monoplane
    MetricBuffers metrics{};

    // Opaque CUDA handles — CUDA pool casts to
    // cudaStream_t/cudaEvent_t/cudaGraphExec_t
    void* stream = nullptr;
    void* completion_event = nullptr;
    void* graph_exec = nullptr;

    // Device-driven persistent worker counters (device int32, host pinned
    // twins)
    void* dev_nextCandidate = nullptr;
    void* dev_nextChunk = nullptr;
    void* dev_overflowFlag = nullptr;
    void* host_overflowFlag = nullptr;

    // Pose for this evaluation (6 DOF, host side)
    float x_location = 0;
    float y_location = 0;
    float z_location = 0;
    float x_angle = 0;
    float y_angle = 0;
    float z_angle = 0;

    int input_index = -1;
    // DIRECT_DILATION monoplane: frozen per-frame baseline used by the graph
    // recipe's complete() (white-pixel sum of the dilated comparison image A).
    // Pose-independent, set once per frame-index by the executor; the serial
    // path adds the same constant via the DIRECT_DILATION custom variable.
    int comparison_image_white_sum = 0;
    EvaluationStatus status = EvaluationStatus::Idle;
    bool initialized_correctly = false;
    bool in_flight = false;
};

// Pool owns EvaluationContexts and their CUDA resources.
// Construction is explicit: caller provides frame/model layout and free-device
// bytes; pool decides N via bank_state_math::admit half-memory + graph VRAM
// probe. Destruction waits for each context's stream/event before freeing.
class EvaluationContextPool {
public:
    EvaluationContextPool() = default;
    ~EvaluationContextPool();

    EvaluationContextPool(const EvaluationContextPool&) = delete;
    EvaluationContextPool& operator=(const EvaluationContextPool&) = delete;

    bool Initialize(
        const BankFootprintInput& layout,
        std::uint64_t free_device_bytes,
        std::size_t n_max);
    void Shutdown();

    std::size_t size() const;
    EvaluationContext* context(std::size_t idx);
    const EvaluationContext* context(std::size_t idx) const;

    // Checkout/Recycle mirror BankCheckoutTracker but operate on
    // EvaluationContext
    int Checkout();
    bool Recycle(std::size_t idx, bool completion_ready);
    bool IsInFlight(std::size_t idx) const;

    // Plan 012 U3 C4/C8: preparation leases + poisoned handling (headless
    // testable)
    void InitForTest(std::size_t count);
    bool ForceRelease(std::size_t idx);
    bool LeavePoisoned(std::size_t idx);
    bool IsPoisoned(std::size_t idx) const;

private:
    std::vector<EvaluationContext> contexts_{};
    std::vector<bool> checked_out_{};
    std::vector<bool> poisoned_{};
};

}  // namespace gpu_cost_function
