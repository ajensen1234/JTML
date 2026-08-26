/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*
 * U1/U4: EvaluationContextPool — CUDA-aware implementation.
 * Allocates full RenderBuffers + MetricBuffers per context (U4 requirement).
 * Null-init, destruction waits for stream/event, then frees everything.
 * This TU is the only place that includes cuda_runtime.
 */

#include "compute/evaluation_context.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <new>

namespace gpu_cost_function {

namespace {

// ── Free helpers ────────────────────────────────────────────────────
void FreeDevice(void*& p) {
    if (p != nullptr) {
        cudaFree(p);
        p = nullptr;
    }
}
void FreeHost(void*& p) {
    if (p != nullptr) {
        cudaFreeHost(p);
        p = nullptr;
    }
}

void Release(RenderBuffers& render) {
    FreeDevice(render.output);
    FreeHost(render.host_bounding_box);
    FreeDevice(render.dev_backface);
    FreeDevice(render.dev_transformed_vertex_zs);
    FreeDevice(render.dev_tangent_triangle);
    FreeDevice(render.dev_projected_triangles);
    FreeDevice(render.dev_projected_triangles_snapped);
    FreeDevice(render.dev_bounding_box_triangles);
    FreeDevice(render.dev_bounding_box_triangles_sizes);
    FreeDevice(render.dev_bounding_box_triangles_sizes_prefix);
    FreeDevice(render.dev_bounding_box);
    FreeDevice(render.dev_fragment_fill);
    FreeHost(render.host_fragment_fill);
    FreeDevice(render.dev_stride_prefixes);
    FreeDevice(render.dev_cub_storage);
    FreeDevice(render.dev_metric_crop);
}

void Release(MetricBuffers& metrics) {
    FreeHost(metrics.host_pixel_score);
    FreeDevice(metrics.dev_pixel_score);
    FreeHost(metrics.host_intersection);
    FreeHost(metrics.host_union);
    FreeDevice(metrics.dev_intersection);
    FreeDevice(metrics.dev_union);
    FreeHost(metrics.host_white_count);
    FreeDevice(metrics.dev_white_count);
    FreeHost(metrics.host_distance_score);
    FreeDevice(metrics.dev_distance_score);
    FreeHost(metrics.host_edge_count);
    FreeDevice(metrics.dev_edge_count);
    FreeHost(metrics.host_curvature);
    FreeDevice(metrics.dev_curvature);
}

// Mirror CostCapacityService helpers: zero-byte allocations succeed with
// null pointers (documented 0-keypoint/0-work guard).
bool HostAlloc(void** ptr, std::size_t bytes) {
    if (bytes == 0) {
        *ptr = nullptr;
        return true;
    }
    return cudaHostAlloc(ptr, bytes, cudaHostAllocDefault) == cudaSuccess;
}
bool DeviceAlloc(void** ptr, std::size_t bytes) {
    if (bytes == 0) {
        *ptr = nullptr;
        return true;
    }
    return cudaMalloc(ptr, bytes) == cudaSuccess;
}

// ── U4: Full RenderBuffers allocation (mirrors BankAllocation) ──────
bool AllocateRender(RenderBuffers& render, const BankFootprintInput& in) {
    const std::size_t pixels = in.width * in.height;
    const std::size_t triangles = in.triangle_count;
    const std::size_t stride = in.maximum_stride_size;
    if (!DeviceAlloc(&render.output, pixels * sizeof(std::uint8_t)) ||
        !HostAlloc(&render.host_bounding_box, 4 * sizeof(std::int32_t)) ||
        !DeviceAlloc(&render.dev_backface, triangles * sizeof(std::uint8_t)) ||
        !DeviceAlloc(
            &render.dev_transformed_vertex_zs, 3 * triangles * sizeof(float)) ||
        !DeviceAlloc(
            &render.dev_tangent_triangle,
            3 * triangles * sizeof(std::uint8_t)) ||
        !DeviceAlloc(
            &render.dev_projected_triangles, 6 * triangles * sizeof(float)) ||
        !DeviceAlloc(
            &render.dev_projected_triangles_snapped,
            6 * triangles * sizeof(std::int32_t)) ||
        !DeviceAlloc(
            &render.dev_bounding_box_triangles,
            4 * triangles * sizeof(std::int32_t)) ||
        !DeviceAlloc(
            &render.dev_bounding_box_triangles_sizes,
            triangles * sizeof(std::int32_t)) ||
        !DeviceAlloc(
            &render.dev_bounding_box_triangles_sizes_prefix,
            triangles * sizeof(std::int32_t)) ||
        !DeviceAlloc(&render.dev_bounding_box, 4 * sizeof(std::int32_t)) ||
        !DeviceAlloc(&render.dev_fragment_fill, sizeof(std::int32_t)) ||
        !HostAlloc(&render.host_fragment_fill, sizeof(std::int32_t)) ||
        !DeviceAlloc(
            &render.dev_stride_prefixes, stride * sizeof(std::int32_t)) ||
        !DeviceAlloc(&render.dev_metric_crop, sizeof(MetricCropParams))) {
        return false;
    }
    std::size_t cub_bytes = in.cub_storage_bytes;
    if (cub_bytes == 0) {
        // Conservative reserve: legacy engine may report zero from in-place
        // probe.
        cub_bytes = std::max<std::size_t>(64, triangles * 64);
    }
    render.cub_storage_bytes = cub_bytes;
    return DeviceAlloc(&render.dev_cub_storage, cub_bytes);
}

// ── U4: Full MetricBuffers allocation (mirrors BankAllocation) ──────
bool AllocateMetrics(MetricBuffers& metrics, const BankFootprintInput& in) {
    const std::size_t curvature = in.curvature_capacity;
    return HostAlloc(&metrics.host_pixel_score, sizeof(std::int32_t)) &&
        DeviceAlloc(&metrics.dev_pixel_score, sizeof(std::int32_t)) &&
        HostAlloc(&metrics.host_intersection, sizeof(std::int32_t)) &&
        HostAlloc(&metrics.host_union, sizeof(std::int32_t)) &&
        DeviceAlloc(&metrics.dev_intersection, sizeof(std::int32_t)) &&
        DeviceAlloc(&metrics.dev_union, sizeof(std::int32_t)) &&
        HostAlloc(&metrics.host_white_count, sizeof(std::int32_t)) &&
        DeviceAlloc(&metrics.dev_white_count, sizeof(std::int32_t)) &&
        HostAlloc(&metrics.host_distance_score, sizeof(std::int32_t)) &&
        DeviceAlloc(&metrics.dev_distance_score, sizeof(std::int32_t)) &&
        HostAlloc(&metrics.host_edge_count, sizeof(std::int32_t)) &&
        DeviceAlloc(&metrics.dev_edge_count, sizeof(std::int32_t)) &&
        HostAlloc(&metrics.host_curvature, curvature * sizeof(std::int32_t)) &&
        DeviceAlloc(&metrics.dev_curvature, curvature * sizeof(std::int32_t));
}

}  // anonymous namespace

// ── Pool lifecycle ──────────────────────────────────────────────────

EvaluationContextPool::~EvaluationContextPool() {
    Shutdown();
}

bool EvaluationContextPool::Initialize(
    const BankFootprintInput& layout,
    std::uint64_t free_device_bytes,
    std::size_t n_max) {
    Shutdown();

    // Graph object memory is added by U3/U5 through graph_overhead_bytes.
    // U1/U4 uses the full render/metric footprint plus worker counters
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
    poisoned_.assign(count, false);

    for (std::size_t i = 0; i < count; ++i) {
        EvaluationContext& ctx = contexts_[i];
        ctx.index = i;
        ctx.width = static_cast<int>(layout.width);
        ctx.height = static_cast<int>(layout.height);
        ctx.status = EvaluationStatus::Idle;
        ctx.in_flight = false;
        ctx.input_index = -1;
        ctx.initialized_correctly = false;

        // ── Stream + Event ──
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

        // ── U4: Full RenderBuffers + MetricBuffers ──
        if (!AllocateRender(ctx.primary, layout) ||
            !AllocateMetrics(ctx.metrics, layout)) {
            Shutdown();
            return false;
        }

        // ── Persistent worker counters ──
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
    for (std::size_t idx = 0; idx < contexts_.size(); ++idx) {
        auto& ctx = contexts_[idx];
        bool isPoisoned = (idx < poisoned_.size() && poisoned_[idx]);
        if (isPoisoned) {
            // C8: poisoned/hung contexts must not be synchronized or freed;
            // leak intentionally until process exit; just reset bookkeeping.
            ctx.initialized_correctly = false;
            ctx.in_flight = false;
            ctx.status = EvaluationStatus::Idle;
            ctx.input_index = -1;
            continue;
        }
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

        // U4: release full RenderBuffers + MetricBuffers
        Release(ctx.primary);
        Release(ctx.metrics);

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
    poisoned_.clear();
}

std::size_t EvaluationContextPool::size() const {
    return contexts_.size();
}

EvaluationContext* EvaluationContextPool::context(std::size_t idx) {
    if (idx >= contexts_.size()) {
        return nullptr;
    }
    return &contexts_[idx];
}

const EvaluationContext* EvaluationContextPool::context(std::size_t idx) const {
    if (idx >= contexts_.size()) {
        return nullptr;
    }
    return &contexts_[idx];
}

int EvaluationContextPool::Checkout() {
    for (std::size_t i = 0; i < checked_out_.size(); ++i) {
        bool isPoisoned = (i < poisoned_.size() && poisoned_[i]);
        if (!checked_out_[i] && !isPoisoned) {
            checked_out_[i] = true;
            contexts_[i].in_flight = true;
            contexts_[i].status = EvaluationStatus::InFlight;
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool EvaluationContextPool::Recycle(std::size_t idx, bool completion_ready) {
    if (idx >= checked_out_.size() || !checked_out_[idx] || !completion_ready) {
        return false;
    }
    checked_out_[idx] = false;
    contexts_[idx].in_flight = false;
    contexts_[idx].status = EvaluationStatus::Idle;
    contexts_[idx].input_index = -1;
    return true;
}

bool EvaluationContextPool::IsInFlight(std::size_t idx) const {
    if (idx >= checked_out_.size()) {
        return false;
    }
    return checked_out_[idx];
}

void EvaluationContextPool::InitForTest(std::size_t count) {
    contexts_.resize(count);
    checked_out_.assign(count, false);
    poisoned_.assign(count, false);
    for (std::size_t i = 0; i < count; ++i) {
        EvaluationContext& ctx = contexts_[i];
        ctx = EvaluationContext{};
        ctx.index = i;
        ctx.status = EvaluationStatus::Idle;
        ctx.in_flight = false;
        ctx.initialized_correctly = false;
        ctx.input_index = -1;
    }
}

bool EvaluationContextPool::ForceRelease(std::size_t idx) {
    if (idx >= checked_out_.size() || !checked_out_[idx]) {
        return false;
    }
    if (idx < poisoned_.size() && poisoned_[idx]) {
        return false;
    }
    checked_out_[idx] = false;
    contexts_[idx].in_flight = false;
    contexts_[idx].status = EvaluationStatus::Idle;
    contexts_[idx].input_index = -1;
    return true;
}

bool EvaluationContextPool::LeavePoisoned(std::size_t idx) {
    if (idx >= checked_out_.size() || !checked_out_[idx]) {
        return false;
    }
    if (idx >= poisoned_.size()) {
        poisoned_.resize(checked_out_.size(), false);
    }
    poisoned_[idx] = true;
    contexts_[idx].status = EvaluationStatus::Failed;
    return true;
}

bool EvaluationContextPool::IsPoisoned(std::size_t idx) const {
    if (idx >= poisoned_.size()) {
        return false;
    }
    return poisoned_[idx];
}

}  // namespace gpu_cost_function
