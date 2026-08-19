/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */
/* Plan 010 U12 Stage 1: bank-state ownership contract.
 * U1 Note: BankState remains as Stage-1 math/alias compatibility; EvaluationContext
 * (evaluation_context.h) is the primary executed type for the graph-backed greedy executor.
 *
 * This is deliberately allocation-free and CUDA-header-free.  It describes the
 * complete per-evaluation write set and provides checked footprint/admission
 * arithmetic.  Runtime CUDA ownership, streams, leases, and bank execution are
 * later U12 stages; no pointer in this contract is allocated or freed here.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace gpu_cost_function {

/* Shared model data.  These are read-only during an evaluation and therefore
 * are intentionally absent from BankState and from per-bank footprint bytes. */
struct SharedReadOnlyGeometry {
    const void* triangles = nullptr;
    const void* normals = nullptr;
};

/* Non-owning names for every render-side write target identified by the U12
 * ownership audit.  Pointers are opaque until the stream/buffer migration;
 * Stage 1 does not allocate or bind any of them. */
struct RenderBuffers {
    void* output = nullptr;
    void* host_bounding_box = nullptr;
    void* dev_backface = nullptr;
    void* dev_transformed_vertex_zs = nullptr;
    void* dev_tangent_triangle = nullptr;
    void* dev_projected_triangles = nullptr;
    void* dev_projected_triangles_snapped = nullptr;
    void* dev_bounding_box_triangles = nullptr;
    void* dev_bounding_box_triangles_sizes = nullptr;
    void* dev_bounding_box_triangles_sizes_prefix = nullptr;
    void* dev_bounding_box = nullptr;
    void* dev_fragment_fill = nullptr;
    void* host_fragment_fill = nullptr;
    void* dev_stride_prefixes = nullptr;
    void* dev_cub_storage = nullptr;
    std::size_t cub_storage_bytes = 0;
};

/* Non-owning names for every metric reduction target and pinned host twin in
 * the audit.  Curvature is capacity-sized because its allocation is dynamic. */
struct MetricBuffers {
    void* host_pixel_score = nullptr;
    void* dev_pixel_score = nullptr;
    void* host_intersection = nullptr;
    void* host_union = nullptr;
    void* dev_intersection = nullptr;
    void* dev_union = nullptr;
    void* host_white_count = nullptr;
    void* dev_white_count = nullptr;
    void* host_distance_score = nullptr;
    void* dev_distance_score = nullptr;
    void* host_edge_count = nullptr;
    void* dev_edge_count = nullptr;
    void* host_curvature = nullptr;
    void* dev_curvature = nullptr;
    std::size_t curvature_capacity = 0;
};

/* A future bank view.  Stage 1 intentionally has no cudaStream_t: stream and
 * lease ownership are introduced only after this write-set contract is locked. */
struct BankState {
    std::size_t index = 0;
    int width = 0;
    int height = 0;
    RenderBuffers primary;
    RenderBuffers secondary;  // empty for monoplane
    MetricBuffers metrics;
    // Opaque handles keep this contract CUDA-header-free. U12's CUDA pool casts
    // them to cudaStream_t/cudaEvent_t; bank 0 remains nullptr for compatibility.
    void* stream = nullptr;
    void* completion_event = nullptr;
    bool in_flight = false;
};

/* Headless lifecycle seam. The CUDA pool uses the same state transitions while
 * its GPU oracle supplies the real event readiness. */
class BankCheckoutTracker {
public:
    explicit BankCheckoutTracker(std::size_t count) : checked_out_(count, false) {}

    int checkout() {
        for (std::size_t i = 0; i < checked_out_.size(); ++i) {
            if (!checked_out_[i]) {
                checked_out_[i] = true;
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    bool recycle(std::size_t index, bool ready) {
        if (index >= checked_out_.size() || !checked_out_[index] || !ready) return false;
        checked_out_[index] = false;
        return true;
    }

    bool checkedOut(std::size_t index) const {
        return index < checked_out_.size() && checked_out_[index];
    }

    std::size_t size() const { return checked_out_.size(); }

private:
    std::vector<bool> checked_out_;
};

struct BankFootprintInput {
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    std::uint64_t triangle_count = 0;
    std::uint64_t maximum_stride_size = 0;
    std::uint64_t cub_storage_bytes = 0;
    std::uint64_t curvature_capacity = 0;
    std::uint64_t graph_overhead_bytes = 0; // U1: per-context cudaGraphExec + events + counters
    bool biplane = false;
};

struct BankFootprint {
    bool valid = false;
    std::uint64_t render_bytes_per_camera = 0;
    std::uint64_t render_bytes = 0;
    std::uint64_t metric_bytes = 0;
    std::uint64_t total_bytes = 0;
};

struct BankAdmission {
    bool admitted = false;
    std::uint64_t budget_bytes = 0;
    std::uint64_t fitting_banks = 0;
    std::uint64_t bank_count = 1;  // 1 means compatibility/serial fallback
};

namespace bank_state_math {

inline bool add(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    if (b > max - a) return false;
    out = a + b;
    return true;
}

inline bool multiply(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    if (a != 0 && b > max / a) return false;
    out = a * b;
    return true;
}

inline bool addProduct(std::uint64_t base, std::uint64_t a, std::uint64_t b,
                       std::uint64_t& out) {
    std::uint64_t product = 0;
    return multiply(a, b, product) && add(base, product, out);
}

/* Bytes written by one camera's render pipeline.  The fields mirror the
 * RenderBuffers names above and the audited CUDA allocations exactly. */
inline std::uint64_t renderBytesPerCamera(const BankFootprintInput& in,
                                          bool& valid) {
    valid = true;
    std::uint64_t bytes = 0;
    const auto add = [&](std::uint64_t count, std::uint64_t element_bytes) {
        std::uint64_t next = 0;
        if (!multiply(count, element_bytes, next) ||
            !bank_state_math::add(bytes, next, bytes)) {
            valid = false;
        }
    };
    const auto addScaled = [&](std::uint64_t count, std::uint64_t groups,
                               std::uint64_t element_bytes) {
        std::uint64_t elements = 0;
        if (!multiply(count, element_bytes, elements) ||
            !addProduct(bytes, groups, elements, bytes)) {
            valid = false;
        }
    };
    std::uint64_t pixels = 0;
    if (!multiply(in.width, in.height, pixels)) valid = false;

    add(pixels, sizeof(std::uint8_t));                    // output image
    add(4, sizeof(std::int32_t));                         // device bbox
    add(4, sizeof(std::int32_t));                         // pinned host bbox
    add(in.triangle_count, sizeof(std::uint8_t));         // backface
    addScaled(in.triangle_count, 3, sizeof(float));       // transformed vertex z
    addScaled(in.triangle_count, 3, sizeof(std::uint8_t)); // tangent triangle
    addScaled(in.triangle_count, 6, sizeof(float));      // projected triangles
    addScaled(in.triangle_count, 6, sizeof(std::int32_t)); // snapped triangles
    addScaled(in.triangle_count, 4, sizeof(std::int32_t)); // triangle bboxes
    add(in.triangle_count, sizeof(std::int32_t));         // bbox sizes
    add(in.triangle_count, sizeof(std::int32_t));         // bbox prefix
    add(1, sizeof(std::int32_t));                         // fragment fill device
    add(1, sizeof(std::int32_t));                         // fragment fill host
    add(in.maximum_stride_size, sizeof(std::int32_t));   // stride prefixes
    addScaled(in.width, in.height, sizeof(float));        // z-line values
    add(in.cub_storage_bytes, 1);                         // CUB scratch
    return bytes;
}

/* Bytes written by the metrics path, including every reduction target and its
 * pinned host twin.  The metric state is shared by the two camera render sets. */
inline std::uint64_t metricBytes(const BankFootprintInput& in, bool& valid) {
    valid = true;
    std::uint64_t bytes = 0;
    const auto add = [&](std::uint64_t count, std::uint64_t element_bytes) {
        std::uint64_t next = 0;
        if (!multiply(count, element_bytes, next) ||
            !bank_state_math::add(bytes, next, bytes)) {
            valid = false;
        }
    };
    add(2, sizeof(std::int32_t));  // pixel score device + host
    add(4, sizeof(std::int32_t));  // intersection/union device + host
    add(2, sizeof(std::int32_t));  // white count device + host
    add(2, sizeof(std::int32_t));  // distance score device + host
    add(2, sizeof(std::int32_t));  // edge count device + host
    add(in.curvature_capacity, sizeof(std::int32_t));  // curvature device
    add(in.curvature_capacity, sizeof(std::int32_t));  // curvature host
    return bytes;
}

inline BankFootprint footprint(const BankFootprintInput& in) {
    BankFootprint result;
    bool render_valid = false;
    bool metric_valid = false;
    result.render_bytes_per_camera = renderBytesPerCamera(in, render_valid);
    result.metric_bytes = metricBytes(in, metric_valid);
    result.valid = render_valid && metric_valid;
    if (!result.valid) return result;

    const std::uint64_t cameras = in.biplane ? 2 : 1;
    if (!multiply(result.render_bytes_per_camera, cameras, result.render_bytes)) {
        result.valid = false;
        return result;
    }
    if (!add(result.render_bytes, result.metric_bytes, result.total_bytes)) {
        result.valid = false;
        return result;
    }
    // U1: per-context device counters (nextCandidate/nextChunk/overflowFlag) + graph overhead
    std::uint64_t extra = 0;
    if (!add(extra, 3 * sizeof(std::int32_t), extra)) { result.valid = false; return result; } // device counters
    if (!add(extra, 1 * sizeof(std::int32_t), extra)) { result.valid = false; return result; } // host overflow pinned
    if (!add(extra, in.graph_overhead_bytes, extra)) { result.valid = false; return result; }
    if (!add(result.total_bytes, extra, result.total_bytes)) { result.valid = false; return result; }
    return result;
}

/* Admission is deliberately conservative: at most half of currently free
 * memory may be committed to extra banks.  If the measurement or footprint is
 * unavailable, or even one bank cannot fit in that budget, return bank_count=1
 * and admitted=false so the compatibility bank remains the only usable bank. */
inline BankAdmission admit(std::uint64_t free_bytes,
                           const BankFootprint& footprint_value,
                           std::uint64_t n_max) {
    BankAdmission result;
    result.budget_bytes = free_bytes / 2;
    if (free_bytes == 0 || !footprint_value.valid ||
        footprint_value.total_bytes == 0 ||
        result.budget_bytes < footprint_value.total_bytes || n_max == 0) {
        return result;
    }
    result.fitting_banks = result.budget_bytes / footprint_value.total_bytes;
    if (result.fitting_banks == 0) return result;
    result.bank_count = result.fitting_banks < n_max ? result.fitting_banks : n_max;
    if (result.bank_count == 0) result.bank_count = 1;
    result.admitted = result.bank_count > 1;
    return result;
}

}  // namespace bank_state_math
}  // namespace gpu_cost_function
