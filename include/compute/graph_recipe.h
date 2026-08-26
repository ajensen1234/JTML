/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*
 * U1: GraphRecipe — generic cost-function graph-recipe interface.
 * First admitted recipe is monoplane DIRECT_DILATION; unsupported
 * recipes remain serial. Reads dilation via
 * getActiveCostFunctionClass()->get*ParameterValue, never via
 * updateCostFunctionParameterValues by-value trap.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "compute/bank_state.cuh"

namespace gpu_cost_function {

struct GraphRecipeKey {
    std::string recipeId;  // e.g. "direct_dilation_monoplane"
    bool biplane = false;
    int width = 0;
    int height = 0;
    std::uint64_t triangle_count = 0;
    int dilation = 6;
    std::uint64_t camera_calib_hash = 0;
    std::uint64_t cub_storage_bytes = 0;
    std::uint64_t curvature_capacity = 0;
    std::uint64_t maximum_stride_size = 0;
    std::uint64_t graph_overhead_bytes = 0;
    std::string version = "1";

    bool operator==(const GraphRecipeKey& o) const {
        return recipeId == o.recipeId && biplane == o.biplane &&
            width == o.width && height == o.height &&
            triangle_count == o.triangle_count && dilation == o.dilation &&
            camera_calib_hash == o.camera_calib_hash &&
            cub_storage_bytes == o.cub_storage_bytes &&
            curvature_capacity == o.curvature_capacity &&
            maximum_stride_size == o.maximum_stride_size &&
            graph_overhead_bytes == o.graph_overhead_bytes &&
            version == o.version;
    }
};

// Plan 012 U2 (C7): generation identity = input identity, NOT computed
// white-sum. White-sum is a capture OUTPUT cached on the context. Pointer
// equality alone is insufficient — the upload epoch changes when buffers are
// rewritten in place.
struct CaptureGeneration {
    int frame_index = -1;
    int stage_id = -1;
    int dilation = 6;
    std::uint64_t upload_epoch = 0;
    const void* rendered_image = nullptr;
    const void* comparison_frame = nullptr;
    const void* distance_map = nullptr;
    bool operator==(const CaptureGeneration& o) const {
        return frame_index == o.frame_index && stage_id == o.stage_id &&
            dilation == o.dilation && upload_epoch == o.upload_epoch &&
            rendered_image == o.rendered_image &&
            comparison_frame == o.comparison_frame &&
            distance_map == o.distance_map;
    }
};

struct GraphPreflightResult {
    bool capturable = false;
    int reasonCode =
        0;  // 0 = ok, non-zero maps to cudaError / CUB alias / overflow
    std::string failingNodeHint;
    std::string reasonString;
};

class EvaluationContext;  // forward
class RenderEngine;
class GPUMetrics;
class GPUImage;
class GPUDilatedFrame;
class GPUFrame;

// Production objects used while capturing the U4 render + metric topology.
// The recipe never owns these objects; they must outlive createGraph().
struct GraphRecipeCaptureInputs {
    EvaluationContext* context = nullptr;
    RenderEngine* render = nullptr;
    GPUMetrics* metrics = nullptr;
    GPUImage* rendered_image = nullptr;
    GPUDilatedFrame* comparison_frame = nullptr;
    GPUFrame* distance_map = nullptr;
    int dilation = 6;
};

class GraphRecipe {
public:
    virtual ~GraphRecipe() = default;

    virtual std::string recipeId() const = 0;
    virtual bool isEligible(const std::string& costName, bool biplane)
        const = 0;
    virtual GraphPreflightResult preflight(const GraphRecipeKey& key) const = 0;
    virtual GraphPreflightResult preflight(
        const GraphRecipeKey& key,
        const GraphRecipeCaptureInputs& inputs) const {
        (void)inputs;
        return preflight(key);
    }
    virtual GraphRecipeKey keyForContext(const GraphRecipeKey& base) const = 0;

    // Graph lifecycle — CUDA-owned in .cu, headless in this header.
    // createGraph builds and instantiates a private cudaGraphExec_t per
    // context. updateParams patches pose/buffer addresses for a new pose
    // without re-capturing topology.
    virtual bool createGraph(
        const GraphRecipeKey& key,
        void* stream,
        const GraphRecipeCaptureInputs& inputs,
        void** out_graphExec) const = 0;
    virtual bool updateParams(void* graphExec, EvaluationContext& ctx)
        const = 0;
    virtual bool launch(void* graphExec, void* stream) const = 0;
    virtual double complete(EvaluationContext& ctx) const = 0;
    virtual void destroyGraph(void* graphExec) const = 0;

    // Plan 012 U4 (C6): no-sync twin of complete() — reads pinned scores
    // without stream sync. Caller guarantees D2H landed via event query.
    virtual double completeFromPins(EvaluationContext& ctx) const {
        return complete(ctx);
    }
};

// Plan 012 U4 (C6): shared composition helper — the syncing complete()
// and the no-sync completeFromPins() must produce identical scores.
inline double ComposeDirectDilationScore(
    int white_sum,
    int pixel_score,
    int distance_score,
    int edge_count) {
    return static_cast<double>(white_sum) +
        (-1.0 * static_cast<double>(pixel_score)) +
        (static_cast<double>(distance_score) /
         (static_cast<double>(edge_count) + 0.1));
}

// Registry enumerates recipes; admits only DIRECT_DILATION monoplane initially.
class GraphRecipeRegistry {
public:
    GraphRecipeRegistry() = default;
    inline void Register(std::unique_ptr<GraphRecipe> recipe) {
        recipes_.push_back(std::move(recipe));
    }

    // Returns nullptr if no eligible recipe.
    inline const GraphRecipe* FindEligible(
        const std::string& costName,
        bool biplane) const {
        for (const auto& r : recipes_) {
            if (r->isEligible(costName, biplane)) {
                return r.get();
            }
        }
        return nullptr;
    }
    inline bool IsAdmitted(const std::string& costName, bool biplane) const {
        return FindEligible(costName, biplane) != nullptr;
    }

    // Preflight for the eligible recipe; if no eligible, capturable=false.
    inline GraphPreflightResult Preflight(
        const std::string& costName,
        bool biplane,
        const GraphRecipeKey& key) const {
        const GraphRecipe* r = FindEligible(costName, biplane);
        if (!r) {
            return GraphPreflightResult{
                false, 1, "no eligible recipe", "no recipe"};
        }
        return r->preflight(key);
    }

    inline std::size_t size() const {
        return recipes_.size();
    }

    inline void AddDirectDilationMonoplaneForTesting() {}

private:
    std::vector<std::unique_ptr<GraphRecipe>> recipes_{};
};

// Monoplane DIRECT_DILATION recipe — defined in graph_recipe_direct_dilation.cu
// (U5) Forward declaration for registry wiring.
std::unique_ptr<GraphRecipe> CreateDirectDilationMonoplaneRecipe();

}  // namespace gpu_cost_function
