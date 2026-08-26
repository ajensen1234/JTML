#pragma once
#include <cstdint>
#include <cstring>
#include <string>

#include "compute/graph_recipe.h"

namespace gpu_cost_function {

struct GraphKeyAssemblerInputs {
    std::string recipeId;
    int width = 0;
    int height = 0;
    std::uint64_t triangle_count = 0;
    int dilation = 6;
    std::uint64_t camera_calib_hash = 0;
    std::uint64_t cub_storage_bytes = 0;
    std::uint64_t curvature_capacity = 0;
    std::uint64_t maximum_stride_size = 0;
    std::uint64_t graph_overhead_bytes = 0;
    bool biplane = false;
    std::string version = "1";
};

// Fills every GraphRecipeKey field from inputs. Deterministic.
inline GraphRecipeKey AssembleGraphRecipeKey(
    const GraphKeyAssemblerInputs& in) {
    GraphRecipeKey k;
    k.recipeId = in.recipeId;
    k.biplane = in.biplane;
    k.width = in.width;
    k.height = in.height;
    k.triangle_count = in.triangle_count;
    k.dilation = in.dilation;
    k.camera_calib_hash = in.camera_calib_hash;
    k.cub_storage_bytes = in.cub_storage_bytes;
    k.curvature_capacity = in.curvature_capacity;
    k.maximum_stride_size = in.maximum_stride_size;
    k.graph_overhead_bytes = in.graph_overhead_bytes;
    k.version = in.version;
    return k;
}

// FNV-1a 64-bit over the four CameraCalibration floats + biplane flag.
inline std::uint64_t HashCameraCalibrationParams(
    float principal_distance,
    float principal_x,
    float principal_y,
    float pixel_pitch,
    bool biplane) {
    const std::uint64_t offset = 14695981039346656037ULL;
    const std::uint64_t prime = 1099511628211ULL;
    auto feed = [&](std::uint64_t& h, std::uint64_t byte) {
        h ^= byte;
        h *= prime;
    };
    std::uint64_t h = offset;
    for (const float f :
         {principal_distance, principal_x, principal_y, pixel_pitch}) {
        std::uint32_t bits;
        static_assert(sizeof(bits) == sizeof(f));
        std::memcpy(&bits, &f, sizeof(bits));
        feed(h, bits & 0xFF);
        feed(h, (bits >> 8) & 0xFF);
        feed(h, (bits >> 16) & 0xFF);
        feed(h, (bits >> 24) & 0xFF);
    }
    feed(h, biplane ? 1 : 0);
    return h;
}

struct CaptureGenerationAssemblerInputs {
    int frame_index = -1;
    int stage_id = -1;
    int dilation = 6;
    std::uint64_t upload_epoch = 0;
    const void* rendered_image = nullptr;
    const void* comparison_frame = nullptr;
    const void* distance_map = nullptr;
};

inline CaptureGeneration AssembleCaptureGeneration(
    const CaptureGenerationAssemblerInputs& in) {
    CaptureGeneration g;
    g.frame_index = in.frame_index;
    g.stage_id = in.stage_id;
    g.dilation = in.dilation;
    g.upload_epoch = in.upload_epoch;
    g.rendered_image = in.rendered_image;
    g.comparison_frame = in.comparison_frame;
    g.distance_map = in.distance_map;
    return g;
}

// U2 validation gate: key.dilation == inputs.dilation && rendered_image !=
// nullptr.
inline bool ValidateGraphKeyVsInputs(
    const GraphRecipeKey& key,
    const GraphRecipeCaptureInputs& inputs) {
    return key.dilation == inputs.dilation && inputs.rendered_image != nullptr;
}

}  // namespace gpu_cost_function
