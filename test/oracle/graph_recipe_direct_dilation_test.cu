/* Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/* U5 oracle: the real monoplane DIRECT_DILATION graph recipe over the frozen
 * Kneel_1 workload.  The graph path is compared with the production serial
 * enqueue path at image, raw-reduction, and composed-score layers. */
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "compute/camera_calibration.h"
#include "compute/evaluation_context.h"
#include "compute/gpu_dilated_frame.cuh"
#include "compute/gpu_frame.cuh"
#include "compute/gpu_image.cuh"
#include "compute/gpu_metrics.cuh"
#include "compute/graph_recipe_direct_dilation.h"
#include "compute/render_engine.cuh"

namespace {
constexpr int kWidth = 1024;
constexpr int kHeight = 1024;
constexpr int kTriangleCount = 12412;
constexpr std::uint64_t kMaximumStrideSize = 10000000;
constexpr int kDilation = 6;
constexpr const char* kImplant = "example_studies/Kneel_1/KR_right_7_fem.stl";

// Frozen in test/golden/graph_pre_registration.json (layer_c_tolerance).
constexpr double kLayerCAbsTolerance = 1e-12;
constexpr double kLayerCRelTolerance = 1e-9;

struct PoseValues {
    float x = 0.0f;
    float y = 0.0f;
    float z = -900.0f;
    float x_angle = 0.0f;
    float y_angle = 0.0f;
    float z_angle = 0.0f;
};

struct EvaluationResult {
    double score = 0.0;
    std::vector<unsigned char> image;
    int pixel_score = 0;
    int distance_score = 0;
    int edge_count = 0;
};

struct KneelFixture {
    std::vector<float> triangles;
    std::vector<float> normals;
    std::unique_ptr<gpu_cost_function::RenderEngine> engine;
    std::unique_ptr<gpu_cost_function::GPUMetrics> metrics;
    std::unique_ptr<gpu_cost_function::GPUImage> comparison_image;
    std::unique_ptr<gpu_cost_function::GPUDilatedFrame> comparison_frame;
    std::unique_ptr<gpu_cost_function::GPUFrame> distance_map;
    gpu_cost_function::EvaluationContextPool pool;

    bool loadStl() {
        std::ifstream file(kImplant);
        if (!file) return false;
        triangles.reserve(static_cast<std::size_t>(kTriangleCount) * 9);
        normals.reserve(static_cast<std::size_t>(kTriangleCount) * 3);

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream facet(line);
            std::string keyword;
            std::string normal_keyword;
            float nx = 0.0f;
            float ny = 0.0f;
            float nz = 0.0f;
            facet >> keyword >> normal_keyword >> nx >> ny >> nz;
            if (keyword != "facet" || normal_keyword != "normal") continue;

            std::vector<float> vertices;
            while (vertices.size() < 9 && std::getline(file, line)) {
                std::istringstream vertex_line(line);
                vertex_line >> keyword;
                if (keyword != "vertex") continue;
                float x = 0.0f;
                float y = 0.0f;
                float z = 0.0f;
                vertex_line >> x >> y >> z;
                vertices.insert(vertices.end(), {x, y, z});
            }
            if (vertices.size() != 9) return false;
            triangles.insert(triangles.end(), vertices.begin(), vertices.end());
            normals.insert(normals.end(), {nx, ny, nz});
        }
        return triangles.size() ==
                   static_cast<std::size_t>(kTriangleCount) * 9 &&
               normals.size() == static_cast<std::size_t>(kTriangleCount) * 3;
    }

    bool setup() {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
            device_count == 0) {
            return false;
        }
        if (!loadStl()) return false;

        const CameraCalibration calibration(1198.0f, 0.0f, 0.0f, 0.373f);
        engine = std::make_unique<gpu_cost_function::RenderEngine>(
            kWidth,
            kHeight,
            0,
            false,
            triangles.data(),
            normals.data(),
            kTriangleCount,
            calibration);
        if (!engine->IsInitializedCorrectly()) return false;

        metrics = std::make_unique<gpu_cost_function::GPUMetrics>();
        if (!metrics->IsInitializedCorrectly()) return false;

        // Match the established U4 fixture: a real non-empty comparison frame
        // with a white center patch, so the dilated white-sum baseline and raw
        // metric reductions are both exercised.
        std::vector<unsigned char> comparison_host(kWidth * kHeight, 0);
        for (int y = kHeight / 4; y < 3 * kHeight / 4; ++y) {
            for (int x = kWidth / 4; x < 3 * kWidth / 4; ++x) {
                comparison_host[y * kWidth + x] = 255;
            }
        }
        comparison_image = std::make_unique<gpu_cost_function::GPUImage>(
            kWidth, kHeight, 0, comparison_host.data());
        comparison_frame = std::make_unique<gpu_cost_function::GPUDilatedFrame>(
            kWidth, kHeight, 0, comparison_host.data(), kDilation);
        distance_map = std::make_unique<gpu_cost_function::GPUFrame>(
            kWidth, kHeight, 0, comparison_host.data());
        if (!comparison_image->IsInitializedCorrectly() ||
            !comparison_frame->IsInitializedCorrectly() ||
            !distance_map->IsInitializedCorrectly()) {
            return false;
        }

        gpu_cost_function::BankFootprintInput layout{};
        layout.width = kWidth;
        layout.height = kHeight;
        layout.triangle_count = kTriangleCount;
        layout.maximum_stride_size = kMaximumStrideSize;
        layout.cub_storage_bytes = engine->GetCubStorageBytes();
        layout.curvature_capacity = 0;
        layout.graph_overhead_bytes = 0;
        layout.biplane = false;
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
            !pool.Initialize(layout, free_bytes, 2) || pool.size() < 2) {
            return false;
        }
        engine->SetPose(
            gpu_cost_function::Pose(0.0f, 0.0f, -900.0f, 0.0f, 0.0f, 0.0f));
        return true;
    }

    gpu_cost_function::GraphRecipeKey key() const {
        gpu_cost_function::GraphRecipeKey value;
        value.recipeId = "direct_dilation_monoplane";
        value.biplane = false;
        value.width = kWidth;
        value.height = kHeight;
        value.triangle_count = kTriangleCount;
        value.dilation = kDilation;
        value.camera_calib_hash = 0x1198000000000175ULL;
        value.cub_storage_bytes = engine->GetCubStorageBytes();
        value.curvature_capacity = 0;
        value.maximum_stride_size = kMaximumStrideSize;
        value.graph_overhead_bytes = 0;
        value.version = "1";
        return value;
    }

    gpu_cost_function::GraphRecipeCaptureInputs
    inputs(gpu_cost_function::EvaluationContext& context) {
        return gpu_cost_function::GraphRecipeCaptureInputs{
            &context,
            engine.get(),
            metrics.get(),
            comparison_image.get(),
            comparison_frame.get(),
            distance_map.get(),
            kDilation};
    }
};

void SetPose(
    gpu_cost_function::EvaluationContext& ctx, const PoseValues& pose) {
    ctx.x_location = pose.x;
    ctx.y_location = pose.y;
    ctx.z_location = pose.z;
    ctx.x_angle = pose.x_angle;
    ctx.y_angle = pose.y_angle;
    ctx.z_angle = pose.z_angle;
    ctx.status = gpu_cost_function::EvaluationStatus::InFlight;
}

void SetEnginePose(
    gpu_cost_function::RenderEngine& engine, const PoseValues& pose) {
    engine.SetPose(gpu_cost_function::Pose(
        pose.x, pose.y, pose.z, pose.x_angle, pose.y_angle, pose.z_angle));
}

std::vector<unsigned char>
CopyImage(const gpu_cost_function::EvaluationContext& ctx) {
    std::vector<unsigned char> image(kWidth * kHeight);
    REQUIRE(
        cudaMemcpy(
            image.data(),
            ctx.primary.output,
            image.size(),
            cudaMemcpyDeviceToHost) == cudaSuccess);
    return image;
}

EvaluationResult
ReadResult(const gpu_cost_function::EvaluationContext& ctx, double score) {
    REQUIRE(ctx.metrics.host_pixel_score != nullptr);
    REQUIRE(ctx.metrics.host_distance_score != nullptr);
    REQUIRE(ctx.metrics.host_edge_count != nullptr);
    EvaluationResult result;
    result.score = score;
    result.image = CopyImage(ctx);
    result.pixel_score = *static_cast<const int*>(ctx.metrics.host_pixel_score);
    result.distance_score =
        *static_cast<const int*>(ctx.metrics.host_distance_score);
    result.edge_count = *static_cast<const int*>(ctx.metrics.host_edge_count);
    return result;
}

void RequireLayerCEqual(double actual, double expected) {
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    REQUIRE(
        std::abs(actual - expected) <=
        std::max(kLayerCAbsTolerance, kLayerCRelTolerance * scale));
}

EvaluationResult RunGraph(
    gpu_cost_function::DirectDilationMonoplaneRecipe& recipe,
    void* graph,
    gpu_cost_function::EvaluationContext& ctx,
    const PoseValues& pose) {
    SetPose(ctx, pose);
    REQUIRE(recipe.updateParams(graph, ctx));
    REQUIRE(recipe.launch(graph, ctx.stream));
    const double score = recipe.complete(ctx);
    REQUIRE(std::isfinite(score));
    REQUIRE(ctx.status == gpu_cost_function::EvaluationStatus::Ready);
    return ReadResult(ctx, score);
}

EvaluationResult RunSerial(
    KneelFixture& fixture,
    gpu_cost_function::EvaluationContext& ctx,
    const PoseValues& pose) {
    SetPose(ctx, pose);
    SetEnginePose(*fixture.engine, pose);
    REQUIRE(fixture.engine->EnqueueRenderPhase(ctx) == cudaSuccess);
    REQUIRE(fixture.engine->CompleteRenderPhase(ctx) == cudaSuccess);

    REQUIRE(
        fixture.metrics->EnqueueFastImplantDilationMetric(
            fixture.comparison_image.get(),
            fixture.comparison_frame.get(),
            kDilation,
            ctx) == cudaSuccess);
    REQUIRE(
        fixture.metrics->EnqueueDistanceMapMetric(
            fixture.comparison_image.get(),
            fixture.distance_map.get(),
            kDilation,
            ctx) == cudaSuccess);
    REQUIRE(
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.stream)) ==
        cudaSuccess);

    REQUIRE(ctx.comparison_image_white_sum > 0);
    const double score =
        static_cast<double>(ctx.comparison_image_white_sum) -
        static_cast<double>(*static_cast<int*>(ctx.metrics.host_pixel_score)) +
        static_cast<double>(
            *static_cast<int*>(ctx.metrics.host_distance_score)) /
            (static_cast<double>(
                 *static_cast<int*>(ctx.metrics.host_edge_count)) +
             0.1);
    REQUIRE(std::isfinite(score));
    ctx.status = gpu_cost_function::EvaluationStatus::Ready;
    return ReadResult(ctx, score);
}

void RequireParity(
    const EvaluationResult& graph, const EvaluationResult& serial) {
    REQUIRE(graph.image == serial.image);
    REQUIRE(graph.pixel_score == serial.pixel_score);
    REQUIRE(graph.distance_score == serial.distance_score);
    REQUIRE(graph.edge_count == serial.edge_count);
    RequireLayerCEqual(graph.score, serial.score);
}

} // namespace

TEST_CASE(
    "oracle: U5 captures and validates the real Kneel_1 U4 chain",
    "[graph_recipe][oracle][U5]") {
    KneelFixture fixture;
    int device_count = 0;
    REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
    if (device_count == 0) {
        WARN("No CUDA device — skipping");
        return;
    }
    REQUIRE(fixture.setup());

    gpu_cost_function::DirectDilationMonoplaneRecipe recipe;
    const auto key = fixture.key();
    const int context_index = fixture.pool.Checkout();
    REQUIRE(context_index >= 0);
    auto* context =
        fixture.pool.context(static_cast<std::size_t>(context_index));
    REQUIRE(context != nullptr);

    const auto inputs = fixture.inputs(*context);
    REQUIRE(recipe.preflight(key, inputs).capturable);

    void* graph = nullptr;
    REQUIRE(recipe.createGraph(key, context->stream, inputs, &graph));
    REQUIRE(graph != nullptr);
    REQUIRE(context->comparison_image_white_sum > 0);

    const PoseValues first_pose{};
    const auto first = RunGraph(recipe, graph, *context, first_pose);
    const auto repeated = RunGraph(recipe, graph, *context, first_pose);
    REQUIRE(repeated.image == first.image);
    REQUIRE(repeated.pixel_score == first.pixel_score);
    REQUIRE(repeated.distance_score == first.distance_score);
    REQUIRE(repeated.edge_count == first.edge_count);
    REQUIRE(repeated.score == first.score);

    const PoseValues second_pose{2.0f, 1.0f, -900.0f, 0.0f, 0.0f, 0.0f};
    const auto second = RunGraph(recipe, graph, *context, second_pose);
    REQUIRE((second.image != first.image || second.score != first.score));

    const auto serial_first = RunSerial(fixture, *context, first_pose);
    RequireParity(first, serial_first);

    recipe.destroyGraph(graph);
    REQUIRE(
        fixture.pool.Recycle(static_cast<std::size_t>(context_index), true));
}

TEST_CASE(
    "oracle: U5 rejects unsupported recipe families",
    "[graph_recipe][oracle][U5]") {
    gpu_cost_function::DirectDilationMonoplaneRecipe recipe;
    REQUIRE(recipe.isEligible("DIRECT_DILATION", false));
    REQUIRE_FALSE(recipe.isEligible("DIRECT_DILATION", true));
    REQUIRE_FALSE(recipe.isEligible("DIRECT_MAHFOUZ", false));

    auto biplane = gpu_cost_function::GraphRecipeKey{};
    biplane.recipeId = "direct_dilation_monoplane";
    biplane.biplane = true;
    biplane.width = kWidth;
    biplane.height = kHeight;
    biplane.triangle_count = kTriangleCount;
    biplane.maximum_stride_size = kMaximumStrideSize;
    REQUIRE_FALSE(recipe.preflight(biplane).capturable);
}

TEST_CASE(
    "oracle: U5 private Execs preserve two-context pose-to-score mapping",
    "[graph_recipe][oracle][U5]") {
    KneelFixture fixture;
    int device_count = 0;
    REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
    if (device_count == 0) {
        WARN("No CUDA device — skipping");
        return;
    }
    REQUIRE(fixture.setup());

    gpu_cost_function::DirectDilationMonoplaneRecipe recipe;
    const auto key = fixture.key();
    const int first_index = fixture.pool.Checkout();
    const int second_index = fixture.pool.Checkout();
    REQUIRE(first_index >= 0);
    REQUIRE(second_index >= 0);
    auto* first = fixture.pool.context(static_cast<std::size_t>(first_index));
    auto* second = fixture.pool.context(static_cast<std::size_t>(second_index));
    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);

    void* first_graph = nullptr;
    void* second_graph = nullptr;
    REQUIRE(recipe.createGraph(
        key, first->stream, fixture.inputs(*first), &first_graph));
    REQUIRE(recipe.createGraph(
        key, second->stream, fixture.inputs(*second), &second_graph));
    REQUIRE(first_graph != nullptr);
    REQUIRE(second_graph != nullptr);
    REQUIRE(first_graph != second_graph);
    REQUIRE(first->comparison_image_white_sum > 0);
    REQUIRE(second->comparison_image_white_sum > 0);

    const PoseValues first_pose{0.0f, 0.0f, -900.0f, 0.0f, 0.0f, 0.0f};
    const PoseValues second_pose{2.0f, 1.0f, -900.0f, 0.0f, 0.0f, 0.0f};
    SetPose(*first, first_pose);
    SetPose(*second, second_pose);
    REQUIRE(recipe.updateParams(first_graph, *first));
    REQUIRE(recipe.updateParams(second_graph, *second));
    REQUIRE(recipe.launch(first_graph, first->stream));
    REQUIRE(recipe.launch(second_graph, second->stream));

    const double first_score = recipe.complete(*first);
    REQUIRE(std::isfinite(first_score));
    const auto first_graph_result = ReadResult(*first, first_score);
    const double second_score = recipe.complete(*second);
    REQUIRE(std::isfinite(second_score));
    const auto second_graph_result = ReadResult(*second, second_score);

    const auto first_serial = RunSerial(fixture, *first, first_pose);
    const auto second_serial = RunSerial(fixture, *second, second_pose);
    RequireParity(first_graph_result, first_serial);
    RequireParity(second_graph_result, second_serial);

    recipe.destroyGraph(first_graph);
    recipe.destroyGraph(second_graph);
    REQUIRE(fixture.pool.Recycle(static_cast<std::size_t>(first_index), true));
    REQUIRE(fixture.pool.Recycle(static_cast<std::size_t>(second_index), true));
}
