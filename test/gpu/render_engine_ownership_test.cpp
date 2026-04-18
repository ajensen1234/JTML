#define private public
#include "gpu/render_engine.cuh"
#undef private

#include <cstddef>
#include <memory>
#include <type_traits>

using gpu_cost_function::GPUImage;
using gpu_cost_function::RenderEngine;

static_assert(
    std::is_same_v<decltype(RenderEngine::fragment_fill_), unique_host_ptr<int>>,
    "RenderEngine should own fragment_fill_ with unique_host_ptr<int>");

static_assert(
    std::is_same_v<decltype(RenderEngine::renderer_output_), std::unique_ptr<GPUImage>>,
    "RenderEngine should own renderer_output_ with std::unique_ptr<GPUImage>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_z_line_values_), unique_device_ptr<float>>,
    "RenderEngine should own dev_z_line_values_ with unique_device_ptr<float>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_transf_vertex_zs_), unique_device_ptr<float>>,
    "RenderEngine should own dev_transf_vertex_zs_ with unique_device_ptr<float>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_tangent_triangle_), unique_device_ptr<bool>>,
    "RenderEngine should own dev_tangent_triangle_ with unique_device_ptr<bool>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_triangles_), unique_device_ptr<float>>,
    "RenderEngine should own dev_triangles_ with unique_device_ptr<float>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_normals_), unique_device_ptr<float>>,
    "RenderEngine should own dev_normals_ with unique_device_ptr<float>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_backface_), unique_device_ptr<bool>>,
    "RenderEngine should own dev_backface_ with unique_device_ptr<bool>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_projected_triangles_), unique_device_ptr<float>>,
    "RenderEngine should own dev_projected_triangles_ with unique_device_ptr<float>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_projected_triangles_snapped_), unique_device_ptr<int>>,
    "RenderEngine should own dev_projected_triangles_snapped_ with unique_device_ptr<int>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_bounding_box_triangles_), unique_device_ptr<int>>,
    "RenderEngine should own dev_bounding_box_triangles_ with unique_device_ptr<int>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_bounding_box_triangles_sizes_), unique_device_ptr<int>>,
    "RenderEngine should own dev_bounding_box_triangles_sizes_ with unique_device_ptr<int>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_bounding_box_triangles_sizes_prefix_), unique_device_ptr<int>>,
    "RenderEngine should own dev_bounding_box_triangles_sizes_prefix_ with unique_device_ptr<int>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_bounding_box_), unique_device_ptr<int>>,
    "RenderEngine should own dev_bounding_box_ with unique_device_ptr<int>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_fragment_fill_), unique_device_ptr<int>>,
    "RenderEngine should own dev_fragment_fill_ with unique_device_ptr<int>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_stride_prefixes_), unique_device_ptr<int>>,
    "RenderEngine should own dev_stride_prefixes_ with unique_device_ptr<int>");

static_assert(
    std::is_same_v<decltype(RenderEngine::dev_cub_storage_), unique_device_ptr<std::byte>>,
    "RenderEngine should own dev_cub_storage_ with unique_device_ptr<std::byte>");

int main() {
    RenderEngine engine;
    return !engine.IsInitializedCorrectly() && engine.GetRenderOutput() == nullptr ? 0 : 1;
}
