/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/* U5: the monoplane DIRECT_DILATION recipe captures the production U4
 * render-to-metric enqueue chain.  There is intentionally no replacement
 * kernel in this translation unit: RenderEngine and GPUMetrics own the
 * operation set, and this recipe only owns graph lifetime and node updates.
 */
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "compute/evaluation_context.h"
#include "compute/fast_implant_dilation_metric.cuh"
#include "compute/gpu_dilated_frame.cuh"
#include "compute/gpu_frame.cuh"
#include "compute/gpu_image.cuh"
#include "compute/gpu_metrics.cuh"
#include "compute/graph_preflight.h"
#include "compute/graph_recipe_direct_dilation.h"
#include "compute/render_engine.cuh"

namespace gpu_cost_function {

// WorldToPixelKernel is defined by the production render engine.  The graph
// recipe uses its symbol only to identify the captured node for SetParams;
// the kernel is never launched from this file.
extern __global__ void WorldToPixelKernel(
    float*,
    float*,
    int*,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    RotationMatrix,
    float*,
    bool*,
    bool,
    float,
    float,
    float,
    float);

namespace {

struct ProbeState {
    const GraphRecipeCaptureInputs* inputs = nullptr;
};

int EnqueueProductionSet(void* stream, void* opaque) {
    auto* state = static_cast<ProbeState*>(opaque);
    if (state == nullptr || state->inputs == nullptr ||
        state->inputs->context == nullptr || state->inputs->render == nullptr ||
        state->inputs->metrics == nullptr ||
        state->inputs->comparison_frame == nullptr ||
        state->inputs->distance_map == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    EvaluationContext& ctx = *state->inputs->context;
    void* original_stream = ctx.stream;
    ctx.stream = stream;

    cudaError_t err = state->inputs->render->EnqueueRenderPhase(ctx);
    if (err == cudaSuccess) {
        err = state->inputs->metrics->EnqueueFastImplantDilationMetric(
            state->inputs->rendered_image,
            state->inputs->comparison_frame,
            state->inputs->dilation,
            ctx);
    }
    if (err == cudaSuccess) {
        err = state->inputs->metrics->EnqueueDistanceMapMetric(
            state->inputs->rendered_image,
            state->inputs->distance_map,
            state->inputs->dilation,
            ctx);
    }

    ctx.stream = original_stream;
    return static_cast<int>(err);
}

struct WorldNodeState {
    cudaGraphNode_t node = nullptr;
    cudaKernelNodeParams params{};
    std::array<void*, 18> args{};

    float* triangles = nullptr;
    float* projected = nullptr;
    int* snapped = nullptr;
    int vertex_count = 0;
    float dist_over_pix_pitch = 0.0f;
    float pix_conversion_x = 0.0f;
    float pix_conversion_y = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    RotationMatrix rotation{};
    float* normals = nullptr;
    bool* backface = nullptr;
    bool use_backface_culling = false;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
};

struct GraphExecWrapper {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    EvaluationContext* captured_context = nullptr;
    GraphRecipeCaptureInputs inputs{};
    WorldNodeState world{};
};

template <typename T>
void ReadKernelArg(T& destination, void* const* args, std::size_t index) {
    std::memcpy(&destination, args[index], sizeof(T));
}

RotationMatrix RotationFor(const EvaluationContext& ctx) {
    const float pi = 3.14159265358979323846f;
    const float cz = std::cos(ctx.z_angle * pi / 180.0f);
    const float sz = std::sin(ctx.z_angle * pi / 180.0f);
    const float cx = std::cos(ctx.x_angle * pi / 180.0f);
    const float sx = std::sin(ctx.x_angle * pi / 180.0f);
    const float cy = std::cos(ctx.y_angle * pi / 180.0f);
    const float sy = std::sin(ctx.y_angle * pi / 180.0f);
    return RotationMatrix(
        cz * cy - sz * sx * sy,
        -sz * cx,
        cz * sy + sz * cy * sx,
        sz * cy + cz * sx * sy,
        cz * cx,
        sz * sy - cz * cy * sx,
        -cx * sy,
        sx,
        cx * cy);
}

bool CaptureWorldNode(cudaGraph_t graph, GraphExecWrapper& wrapper) {
    std::size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess) {
        return false;
    }
    std::vector<cudaGraphNode_t> nodes(count);
    if (count != 0 &&
        cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) {
        return false;
    }

    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess ||
            type != cudaGraphNodeTypeKernel) {
            continue;
        }
        cudaKernelNodeParams params{};
        if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess ||
            params.func != reinterpret_cast<void*>(WorldToPixelKernel) ||
            params.kernelParams == nullptr) {
            continue;
        }

        wrapper.world.node = node;
        wrapper.world.params = params;
        ReadKernelArg(wrapper.world.triangles, params.kernelParams, 0);
        ReadKernelArg(wrapper.world.projected, params.kernelParams, 1);
        ReadKernelArg(wrapper.world.snapped, params.kernelParams, 2);
        ReadKernelArg(wrapper.world.vertex_count, params.kernelParams, 3);
        ReadKernelArg(
            wrapper.world.dist_over_pix_pitch, params.kernelParams, 4);
        ReadKernelArg(wrapper.world.pix_conversion_x, params.kernelParams, 5);
        ReadKernelArg(wrapper.world.pix_conversion_y, params.kernelParams, 6);
        ReadKernelArg(wrapper.world.x, params.kernelParams, 7);
        ReadKernelArg(wrapper.world.y, params.kernelParams, 8);
        ReadKernelArg(wrapper.world.z, params.kernelParams, 9);
        ReadKernelArg(wrapper.world.rotation, params.kernelParams, 10);
        ReadKernelArg(wrapper.world.normals, params.kernelParams, 11);
        ReadKernelArg(wrapper.world.backface, params.kernelParams, 12);
        ReadKernelArg(
            wrapper.world.use_backface_culling, params.kernelParams, 13);
        ReadKernelArg(wrapper.world.fx, params.kernelParams, 14);
        ReadKernelArg(wrapper.world.fy, params.kernelParams, 15);
        ReadKernelArg(wrapper.world.cx, params.kernelParams, 16);
        ReadKernelArg(wrapper.world.cy, params.kernelParams, 17);
        return true;
    }
    return false;
}

void RebuildWorldArgs(WorldNodeState& world, const EvaluationContext& ctx) {
    world.projected = static_cast<float*>(ctx.primary.dev_projected_triangles);
    world.snapped =
        static_cast<int*>(ctx.primary.dev_projected_triangles_snapped);
    world.backface = static_cast<bool*>(ctx.primary.dev_backface);
    world.x = ctx.x_location;
    world.y = ctx.y_location;
    world.z = ctx.z_location;
    world.rotation = RotationFor(ctx);

    world.args = {
        &world.triangles,
        &world.projected,
        &world.snapped,
        &world.vertex_count,
        &world.dist_over_pix_pitch,
        &world.pix_conversion_x,
        &world.pix_conversion_y,
        &world.x,
        &world.y,
        &world.z,
        &world.rotation,
        &world.normals,
        &world.backface,
        &world.use_backface_culling,
        &world.fx,
        &world.fy,
        &world.cx,
        &world.cy};
}

GraphPreflightResult FootprintPreflight(const GraphRecipeKey& key) {
    BankFootprintInput input{};
    input.width = static_cast<std::uint64_t>(key.width);
    input.height = static_cast<std::uint64_t>(key.height);
    input.triangle_count = key.triangle_count;
    input.maximum_stride_size = key.maximum_stride_size;
    input.cub_storage_bytes = key.cub_storage_bytes;
    input.curvature_capacity = key.curvature_capacity;
    input.graph_overhead_bytes = key.graph_overhead_bytes;
    input.biplane = key.biplane;
    const BankFootprint value = bank_state_math::footprint(input);
    if (!value.valid || value.total_bytes == 0) {
        return GraphPreflightResult{
            false,
            kOverflow,
            "BankFootprint",
            "invalid graph write-set footprint"};
    }

    // Run the same admission arithmetic used by the real pool.  A graph recipe
    // is one context, so fitting one context is the relevant preflight check;
    // callers still decide how many contexts to admit from measured free VRAM.
    const auto admission = bank_state_math::admit(
        std::numeric_limits<std::uint64_t>::max(), value, 1);
    if (admission.fitting_banks == 0) {
        return GraphPreflightResult{
            false,
            kOverflow,
            "BankAdmission",
            "graph write set exceeds budget"};
    }
    return GraphPreflightResult{true, kPreflightOk, "", "footprint admitted"};
}

}  // namespace

std::string DirectDilationMonoplaneRecipe::recipeId() const {
    return "direct_dilation_monoplane";
}

bool DirectDilationMonoplaneRecipe::isEligible(
    const std::string& costName,
    bool biplane) const {
    return costName == "DIRECT_DILATION" && !biplane;
}

GraphPreflightResult DirectDilationMonoplaneRecipe::preflight(
    const GraphRecipeKey& key) const {
    if (key.biplane) {
        return GraphPreflightResult{
            false, kNoEligibleRecipe, "biplane", "biplane not admitted"};
    }
    if (!key.recipeId.empty() && key.recipeId != recipeId()) {
        return GraphPreflightResult{
            false, kNoEligibleRecipe, "recipeId", "recipeId mismatch"};
    }
    if (key.width <= 0 || key.height <= 0 || key.triangle_count == 0 ||
        key.maximum_stride_size == 0) {
        return GraphPreflightResult{
            false, kZeroTriangle, "zero work", "zero dimensions or work"};
    }
    return FootprintPreflight(key);
}

GraphPreflightResult DirectDilationMonoplaneRecipe::preflight(
    const GraphRecipeKey& key,
    const GraphRecipeCaptureInputs& inputs) const {
    const auto basic = preflight(key);
    if (!basic.capturable) {
        return basic;
    }
    if (inputs.context == nullptr || inputs.render == nullptr ||
        inputs.metrics == nullptr || inputs.comparison_frame == nullptr ||
        inputs.distance_map == nullptr) {
        return GraphPreflightResult{
            false,
            kNoEligibleRecipe,
            "production capture inputs",
            "render, metrics, context and comparison frames are required"};
    }
    if (!inputs.context->initialized_correctly || !inputs.context->in_flight ||
        inputs.context->stream == nullptr) {
        return GraphPreflightResult{
            false,
            kGraphCaptureError,
            "EvaluationContext",
            "context is not checked out"};
    }
    if (inputs.context->width != key.width ||
        inputs.context->height != key.height) {
        return GraphPreflightResult{
            false,
            kGraphCaptureError,
            "frame dimensions",
            "key/context mismatch"};
    }

    ProbeState state{&inputs};
    return ProbeCapturableOpSet(EnqueueProductionSet, &state);
}

GraphRecipeKey DirectDilationMonoplaneRecipe::keyForContext(
    const GraphRecipeKey& base) const {
    GraphRecipeKey key = base;
    key.recipeId = recipeId();
    key.biplane = false;
    key.version = "1";
    return key;
}

bool DirectDilationMonoplaneRecipe::createGraph(
    const GraphRecipeKey& key,
    void* stream,
    const GraphRecipeCaptureInputs& inputs,
    void** out_graphExec) const {
    if (out_graphExec == nullptr) {
        return false;
    }
    *out_graphExec = nullptr;
    if (!inputs.context || !inputs.render || !inputs.metrics ||
        !inputs.comparison_frame || !inputs.distance_map || !stream ||
        inputs.context->stream != stream ||
        !inputs.context->initialized_correctly || !inputs.context->in_flight) {
        return false;
    }
    const auto preflight_result = preflight(key, inputs);
    if (!preflight_result.capturable) {
        return false;
    }

    auto wrapper = std::make_unique<GraphExecWrapper>();
    wrapper->captured_context = inputs.context;
    wrapper->inputs = inputs;
    const auto capture_stream = reinterpret_cast<cudaStream_t>(stream);

    cudaError_t err =
        cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        return false;
    }

    err = inputs.render->EnqueueRenderPhase(*inputs.context);
    if (err == cudaSuccess) {
        err = inputs.metrics->EnqueueFastImplantDilationMetric(
            inputs.rendered_image,
            inputs.comparison_frame,
            inputs.dilation,
            *inputs.context);
    }
    if (err == cudaSuccess) {
        err = inputs.metrics->EnqueueDistanceMapMetric(
            inputs.rendered_image,
            inputs.distance_map,
            inputs.dilation,
            *inputs.context);
    }

    cudaGraph_t graph = nullptr;
    const cudaError_t end_err = cudaStreamEndCapture(capture_stream, &graph);
    if (err != cudaSuccess || end_err != cudaSuccess || graph == nullptr) {
        if (graph) {
            cudaGraphDestroy(graph);
        }
        cudaGetLastError();
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess || exec == nullptr) {
        cudaGraphDestroy(graph);
        return false;
    }
    wrapper->graph = graph;
    wrapper->exec = exec;

    // Producer for the per-frame DIRECT_DILATION white-sum baseline: the
    // serial path computes this once per frame (ComputeSumWhitePixels over
    // the DILATED comparison frame A) and adds it to every score.  Capture is
    // a one-time host event, so computing it here (and caching on the
    // context) keeps complete() free of the per-frame constant without adding
    // a per-relaunch sync.  Guaranteed non-null by preflight().
    if (inputs.metrics != nullptr && inputs.comparison_frame != nullptr &&
        inputs.comparison_frame->GetGPUImage() != nullptr) {
        cudaError_t ws_err = cudaSuccess;
        const int white_sum = inputs.metrics->ComputeSumWhitePixels(
            inputs.comparison_frame->GetGPUImage(), &ws_err);
        if (ws_err != cudaSuccess || white_sum < 0) {
            destroyGraph(reinterpret_cast<void*>(wrapper.release()));
            return false;
        }
        inputs.context->comparison_image_white_sum = white_sum;
    } else {
        inputs.context->comparison_image_white_sum = 0;
    }

    // The WorldToPixel node is the only node whose pose values are copied from
    // host launch arguments.  All remaining pointers are private to this
    // context and therefore remain topology-stable for its Exec.
    if (!CaptureWorldNode(graph, *wrapper)) {
        destroyGraph(reinterpret_cast<void*>(wrapper.release()));
        return false;
    }
    *out_graphExec = reinterpret_cast<void*>(wrapper.release());
    return true;
}

bool DirectDilationMonoplaneRecipe::updateParams(
    void* graphExec,
    EvaluationContext& ctx) const {
    if (!graphExec || !ctx.initialized_correctly || !ctx.in_flight ||
        ctx.stream == nullptr) {
        return false;
    }
    auto* wrapper = static_cast<GraphExecWrapper*>(graphExec);
    if (!wrapper->exec || wrapper->captured_context != &ctx ||
        wrapper->world.node == nullptr) {
        return false;
    }
    if (ctx.primary.dev_projected_triangles == nullptr ||
        ctx.primary.dev_projected_triangles_snapped == nullptr ||
        ctx.primary.dev_backface == nullptr) {
        return false;
    }

    RebuildWorldArgs(wrapper->world, ctx);
    cudaKernelNodeParams params = wrapper->world.params;
    params.kernelParams = wrapper->world.args.data();
    return cudaGraphExecKernelNodeSetParams(
               wrapper->exec, wrapper->world.node, &params) == cudaSuccess;
}

bool DirectDilationMonoplaneRecipe::launch(void* graphExec, void* stream)
    const {
    if (!graphExec || !stream) {
        return false;
    }
    auto* wrapper = static_cast<GraphExecWrapper*>(graphExec);
    if (!wrapper->exec || wrapper->captured_context == nullptr) {
        return false;
    }
    // complete() synchronizes the captured context's stream, so every launch
    // must go to that same stream; launching on a different stream would make
    // complete() read host pins before the graph finished (read-after-launch
    // race).  Reject a mismatched stream instead of racing.
    if (wrapper->captured_context->stream != stream) {
        return false;
    }
    return cudaGraphLaunch(
               wrapper->exec, reinterpret_cast<cudaStream_t>(stream)) ==
        cudaSuccess;
}

double DirectDilationMonoplaneRecipe::complete(EvaluationContext& ctx) const {
    if (!ctx.initialized_correctly || !ctx.in_flight || ctx.stream == nullptr ||
        ctx.host_overflowFlag == nullptr) {
        ctx.status = EvaluationStatus::Failed;
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto stream = reinterpret_cast<cudaStream_t>(ctx.stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess ||
        *static_cast<int*>(ctx.host_overflowFlag) != 0) {
        ctx.status = EvaluationStatus::Failed;
        return std::numeric_limits<double>::quiet_NaN();
    }
    return completeFromPins(ctx);
}

double DirectDilationMonoplaneRecipe::completeFromPins(
    EvaluationContext& ctx) const {
    if (!ctx.initialized_correctly || !ctx.in_flight ||
        ctx.host_overflowFlag == nullptr) {
        ctx.status = EvaluationStatus::Failed;
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (*static_cast<int*>(ctx.host_overflowFlag) != 0) {
        ctx.status = EvaluationStatus::Failed;
        return std::numeric_limits<double>::quiet_NaN();
    }
    ctx.status = EvaluationStatus::Ready;
    if (ctx.metrics.host_pixel_score == nullptr ||
        ctx.metrics.host_distance_score == nullptr ||
        ctx.metrics.host_edge_count == nullptr) {
        ctx.status = EvaluationStatus::Failed;
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double score = ComposeDirectDilationScore(
        ctx.comparison_image_white_sum,
        *static_cast<int*>(ctx.metrics.host_pixel_score),
        *static_cast<int*>(ctx.metrics.host_distance_score),
        *static_cast<int*>(ctx.metrics.host_edge_count));
    if (!std::isfinite(score)) {
        ctx.status = EvaluationStatus::Failed;
        return std::numeric_limits<double>::quiet_NaN();
    }
    return score;
}

void DirectDilationMonoplaneRecipe::destroyGraph(void* graphExec) const {
    if (!graphExec) {
        return;
    }
    auto* wrapper = static_cast<GraphExecWrapper*>(graphExec);
    if (wrapper->exec) {
        cudaGraphExecDestroy(wrapper->exec);
    }
    if (wrapper->graph) {
        cudaGraphDestroy(wrapper->graph);
    }
    delete wrapper;
}

std::unique_ptr<GraphRecipe> CreateDirectDilationMonoplaneRecipe() {
    return std::make_unique<DirectDilationMonoplaneRecipe>();
}

}  // namespace gpu_cost_function
