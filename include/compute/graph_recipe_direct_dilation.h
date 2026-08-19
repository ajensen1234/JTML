/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*
 * U5: Monoplane DIRECT_DILATION graph recipe — reusable topology.
 * One private cudaGraphExec_t per EvaluationContext, per-eval SetParams update.
 */

#pragma once

#include "compute/graph_recipe.h"

namespace gpu_cost_function {

class DirectDilationMonoplaneRecipe : public GraphRecipe {
public:
    std::string recipeId() const override;
    bool isEligible(const std::string& costName, bool biplane) const override;
    GraphPreflightResult preflight(const GraphRecipeKey& key) const override;
    GraphRecipeKey keyForContext(const GraphRecipeKey& base) const override;

    bool createGraph(const GraphRecipeKey& key, void* stream, void** out_graphExec) const override;
    bool updateParams(void* graphExec, EvaluationContext& ctx) const override;
    bool launch(void* graphExec, void* stream) const override;
    double complete(EvaluationContext& ctx) const override;
    void destroyGraph(void* graphExec) const override;
};

}  // namespace gpu_cost_function
