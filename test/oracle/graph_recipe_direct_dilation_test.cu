/* Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/* U5 oracle: Monoplane DIRECT_DILATION graph recipe — reusable topology
 * (LABELS oracle;gpu) — verifies that the dummy capturable graph can be
 * instantiated and relaunched with different pose params without re-capture,
 * and that biplane / wrong cost name correctly returns capturable=false.
 * Full U4 persistent-worker chain will be wired later, but this proves the
 * reusable-topology contract and one-private-Exec-per-context invariant.
 */

#include <catch2/catch_test_macros.hpp>

#include "compute/evaluation_context.h"
#include "compute/graph_recipe_direct_dilation.h"

#include <cuda_runtime.h>

using gpu_cost_function::BankFootprintInput;
using gpu_cost_function::DirectDilationMonoplaneRecipe;
using gpu_cost_function::EvaluationContext;
using gpu_cost_function::GraphRecipeKey;

TEST_CASE("oracle: U5 direct_dilation_monoplane graph capture+instantiate succeeds", "[graph_recipe][oracle][U5]") {
    DirectDilationMonoplaneRecipe recipe;
    GraphRecipeKey key;
    key.recipeId = "direct_dilation_monoplane";
    key.biplane = false;
    key.width = 512;
    key.height = 512;
    key.triangle_count = 300000;
    key.dilation = 6;
    key.maximum_stride_size = 10000000;
    key.cub_storage_bytes = 4096;

    auto pre = recipe.preflight(key);
    REQUIRE(pre.capturable);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    void* exec = nullptr;
    bool created = recipe.createGraph(key, stream, &exec);
    REQUIRE(created);
    REQUIRE(exec != nullptr);

    // Relaunch with different pose (in real impl, pose is via SetParams)
    EvaluationContext ctx;
    ctx.x_location = 1.0f;
    REQUIRE(recipe.updateParams(exec, ctx));
    REQUIRE(recipe.launch(exec, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    // Second pose — same topology, no re-capture
    ctx.x_location = 2.0f;
    REQUIRE(recipe.updateParams(exec, ctx));
    REQUIRE(recipe.launch(exec, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    recipe.destroyGraph(exec);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("oracle: U5 recipe rejects biplane and wrong cost name", "[graph_recipe][oracle][U5]") {
    DirectDilationMonoplaneRecipe recipe;
    REQUIRE(!recipe.isEligible("DIRECT_DILATION", true));
    REQUIRE(!recipe.isEligible("DIRECT_MAHFOUZ", false));
    REQUIRE(recipe.isEligible("DIRECT_DILATION", false));

    GraphRecipeKey key;
    key.biplane = true;
    key.width = 512; key.height = 512; key.triangle_count = 1000;
    auto pre = recipe.preflight(key);
    REQUIRE(!pre.capturable);
}

TEST_CASE("oracle: U5 two contexts each with private Exec can be in-flight serially", "[graph_recipe][oracle][U5]") {
    DirectDilationMonoplaneRecipe recipe;
    GraphRecipeKey key;
    key.recipeId = "direct_dilation_monoplane";
    key.biplane = false;
    key.width = 512; key.height = 512; key.triangle_count = 300000;
    key.dilation = 6; key.maximum_stride_size = 10000000; key.cub_storage_bytes = 4096;

    cudaStream_t s1 = nullptr, s2 = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking) == cudaSuccess);

    void* exec1 = nullptr; void* exec2 = nullptr;
    REQUIRE(recipe.createGraph(key, s1, &exec1));
    REQUIRE(recipe.createGraph(key, s2, &exec2));
    REQUIRE(exec1 != nullptr);
    REQUIRE(exec2 != nullptr);
    REQUIRE(exec1 != exec2); // one private Exec per context

    EvaluationContext ctx1, ctx2;
    REQUIRE(recipe.updateParams(exec1, ctx1));
    REQUIRE(recipe.updateParams(exec2, ctx2));
    REQUIRE(recipe.launch(exec1, s1));
    REQUIRE(recipe.launch(exec2, s2));
    REQUIRE(cudaStreamSynchronize(s1) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s2) == cudaSuccess);

    recipe.destroyGraph(exec1);
    recipe.destroyGraph(exec2);
    REQUIRE(cudaStreamDestroy(s1) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(s2) == cudaSuccess);
}
