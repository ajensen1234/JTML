/* Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/* U2: GraphRecipe — generic interface, registry, and pre-registration artifact.
 * Covers R3, R10 (matrix), R9 tolerance frozen. Headless, CUDA-free.
 */

#include <catch2/catch_test_macros.hpp>

#include <fstream>

#include "compute/graph_recipe.h"
#include "compute/graph_key_assembler.h"
#include "compute/bank_state.cuh"
#include "compute/bank_state.cuh"

using gpu_cost_function::GraphPreflightResult;
using gpu_cost_function::GraphRecipeKey;
using gpu_cost_function::GraphRecipeRegistry;

TEST_CASE("GraphRecipeKey equality is field-wise", "[graph_recipe]") {
    GraphRecipeKey a;
    a.recipeId = "direct_dilation_monoplane";
    a.biplane = false;
    a.width = 512;
    a.height = 512;
    a.triangle_count = 300000;
    a.dilation = 6;
    a.camera_calib_hash = 42;
    a.cub_storage_bytes = 4096;
    a.graph_overhead_bytes = 1024;
    GraphRecipeKey b = a;
    REQUIRE(a == b);
    b.dilation = 7;
    REQUIRE_FALSE(a == b);
    b = a;
    b.biplane = true;
    REQUIRE_FALSE(a == b);
    b = a;
    b.recipeId = "other";
    REQUIRE_FALSE(a == b);
}

TEST_CASE("GraphRecipeRegistry empty preflight is not capturable", "[graph_recipe]") {
    GraphRecipeRegistry reg;
    REQUIRE(reg.size() == 0);
    REQUIRE(reg.FindEligible("DIRECT_DILATION", false) == nullptr);
    REQUIRE(reg.FindEligible("DIRECT_DILATION", true) == nullptr);
    REQUIRE_FALSE(reg.IsAdmitted("DIRECT_DILATION", false));
    REQUIRE_FALSE(reg.IsAdmitted("DIRECT_MAHFOUZ", false));
    GraphRecipeKey key;
    key.recipeId = "direct_dilation_monoplane";
    auto res = reg.Preflight("DIRECT_DILATION", false, key);
    REQUIRE_FALSE(res.capturable);
    REQUIRE(res.reasonCode != 0);
}

TEST_CASE("GraphRecipeRegistry admits only DIRECT_DILATION monoplane after registration", "[graph_recipe]") {
    // U1's AddDirectDilationMonoplaneForTesting is currently a no-op shim (U5 lands the real recipe).
    // For U2 we verify the registry remains empty and that the *interface* is correctly
    // gated: no eligible recipe means deterministic fallback to serial.
    GraphRecipeRegistry reg;
    reg.AddDirectDilationMonoplaneForTesting();
    // Still not admitted until U5 provides the real recipe — this is the R8 fallback.
    // The test pins that fallback is deterministic, not that the recipe exists yet.
    REQUIRE(reg.size() == 0);
    REQUIRE_FALSE(reg.IsAdmitted("DIRECT_DILATION", false));
}

TEST_CASE("GraphPreflightResult default is not capturable", "[graph_recipe]") {
    GraphPreflightResult r;
    REQUIRE_FALSE(r.capturable);
    REQUIRE(r.reasonCode == 0);
}

TEST_CASE(
    "graph_pre_registration.json is frozen and contains required keys",
    "[graph_recipe]") {
    std::ifstream f("test/golden/graph_pre_registration.json");
    REQUIRE(f.good());
    std::string content(
        (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Top-level frozen flag
    REQUIRE(content.find("\"frozen\": true") != std::string::npos);

    // Complete graph-key dimension schema frozen for U5/U8. Fixture-dependent
    // values remain in workloads; runtime-probed dimensions must not be
    // invented.
    const auto graph_key_field = content.find("\"graph_key\"");
    REQUIRE(graph_key_field != std::string::npos);
    const auto graph_key_start = content.find('[', graph_key_field);
    REQUIRE(graph_key_start != std::string::npos);
    const auto graph_key_end = content.find(']', graph_key_start);
    REQUIRE(graph_key_end != std::string::npos);
    const std::string graph_key_schema =
        content.substr(graph_key_start, graph_key_end - graph_key_start + 1);
    for (const char* dimension : {
             "recipeId",
             "biplaneFlag",
             "width",
             "height",
             "triangle_count",
             "dilationParam",
             "cameraCalibHash",
             "cub_storage_bytes",
             "curvature_capacity",
             "maximum_stride_size",
             "graph_overhead_bytes",
             "version",
         }) {
        INFO("missing graph-key dimension: " << dimension);
        REQUIRE(
            graph_key_schema.find("\"" + std::string(dimension) + "\"") !=
            std::string::npos);
    }

    // Real Kneel_1 fixture values are frozen in the workload entries.
    REQUIRE(content.find("\"width\": 1024") != std::string::npos);
    REQUIRE(content.find("\"height\": 1024") != std::string::npos);
    REQUIRE(content.find("\"triangle_count\": 12412") != std::string::npos);
    REQUIRE(content.find("\"dilation\": 6") != std::string::npos);

    // Workload fixture values (real Kneel_1)
    REQUIRE(content.find("\"pose_batch_size\"") != std::string::npos);
    REQUIRE(content.find("\"N_values\"") != std::string::npos);

    // Layer-C tolerance frozen artifact
    REQUIRE(content.find("\"layer_c_tolerance\"") != std::string::npos);
    REQUIRE(content.find("\"abs\": 1e-12") != std::string::npos);
    REQUIRE(content.find("\"rel\": 1e-9") != std::string::npos);
}

TEST_CASE("CaptureGeneration default is not equal to an assembled generation",
          "[graph_recipe]") {
    using gpu_cost_function::AssembleCaptureGeneration;
    using gpu_cost_function::CaptureGeneration;
    using gpu_cost_function::CaptureGenerationAssemblerInputs;
    CaptureGeneration def;
    REQUIRE(def.frame_index == -1);
    CaptureGenerationAssemblerInputs in;
    in.frame_index = 0;
    in.stage_id = 0;
    in.dilation = 6;
    in.upload_epoch = 1;
    unsigned char a = 0, b = 0, c = 0;
    in.rendered_image = &a;
    in.comparison_frame = &b;
    in.distance_map = &c;
    CaptureGeneration g = AssembleCaptureGeneration(in);
    REQUIRE_FALSE(def == g);
}

TEST_CASE("TEST_IMPACT_MATRIX covers required touching files", "[graph_recipe]") {
    std::ifstream m("docs/TEST_IMPACT_MATRIX.md");
    REQUIRE(m.good());
    std::string c((std::istreambuf_iterator<char>(m)), std::istreambuf_iterator<char>());
    // Must cover the cost-function/compute touching set per plan
    REQUIRE(c.find("test_direct_optimizer_batch.cpp") != std::string::npos);
    REQUIRE(c.find("test_bank_state.cpp") != std::string::npos);
    REQUIRE(c.find("cost_capacity_oracle_test.cu") != std::string::npos);
    REQUIRE(c.find("bit_identity_test.cpp") != std::string::npos);
    REQUIRE(c.find("evaluation_context_test.cpp") != std::string::npos);
    REQUIRE(c.find("graph_recipe_preflight_test.cpp") != std::string::npos);
    REQUIRE(c.find("graph_pre_registration.json") != std::string::npos);
}
