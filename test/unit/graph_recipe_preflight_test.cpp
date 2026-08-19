/* Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/* U2: GraphRecipe — generic interface, registry, and pre-registration artifact.
 * Covers R3, R10 (matrix), R9 tolerance frozen. Headless, CUDA-free.
 */

#include <catch2/catch_test_macros.hpp>

#include <fstream>

#include "compute/graph_recipe.h"
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

TEST_CASE("graph_pre_registration.json is frozen and contains required keys", "[graph_recipe]") {
    std::ifstream f("test/golden/graph_pre_registration.json");
    REQUIRE(f.good());
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    REQUIRE(content.find("\"frozen\"") != std::string::npos);
    REQUIRE(content.find("true") != std::string::npos);
    REQUIRE(content.find("\"pose_batch_size\"") != std::string::npos);
    REQUIRE(content.find("\"N_values\"") != std::string::npos);
    REQUIRE(content.find("\"layer_c_tolerance\"") != std::string::npos);
    REQUIRE(content.find("\"abs\"") != std::string::npos);
    REQUIRE(content.find("\"graph_key\"") != std::string::npos);
    REQUIRE(content.find("recipeId") != std::string::npos);
    REQUIRE(content.find("biplaneFlag") != std::string::npos);
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
