/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*Plan 008 U7 (Cut A) — the optimizer-container pure surface (see
 * include/coordinator/optimizer_stage_script.h for the schema, the
 * convergence-tradeoff design note, and the R1/R2/R4/R5/R6 trace). Everything
 * here is a verbatim transcription of the manager's running code:
 *  - BuildStageScript: the Optimize() loop's stage shape + enabled-flag gating
 *    (src/coordinator/optimizer_manager.cpp, the trunk/branch/leaf blocks);
 *  - DeriveStageCostParams: the manager's per-manager parameter scan
 *    (dilation / dark-silhouette);
 *  - the registry: the jtml-production graph built from the same
 *    settings_constants.h constants the loop consumes (the lineage tibia
 *    transcription, angle 04 R3-1).*/

#include "coordinator/optimizer_stage_script.h"

#include <stdexcept>
#include <string>

#include "domain/settings_constants.h"

namespace jta {

namespace {

/*Validate the settings' numeric fields (error path: negative budgets fail
 * fast with a clear error — a negative budget is a settings corruption, not a
 * run shape; the engine never validates).*/
void ValidateSettings(const OptimizerSettings& settings) {
    if (settings.trunk_budget < 0) {
        throw std::invalid_argument(
            "BuildStageScript: negative trunk_budget (" +
            std::to_string(settings.trunk_budget) +
            "); budgets must be non-negative");
    }
    if (settings.branch_budget < 0) {
        throw std::invalid_argument(
            "BuildStageScript: negative branch_budget (" +
            std::to_string(settings.branch_budget) +
            "); budgets must be non-negative");
    }
    if (settings.leaf_budget < 0) {
        throw std::invalid_argument(
            "BuildStageScript: negative leaf_budget (" +
            std::to_string(settings.leaf_budget) +
            "); budgets must be non-negative");
    }
    if (settings.number_branches < 0) {
        throw std::invalid_argument(
            "BuildStageScript: negative number_branches (" +
            std::to_string(settings.number_branches) +
            "); the branch count must be non-negative");
    }
}

/*The five normal directives — frame-selection directives, stage-shape
 * neutral (the shape is identical across them; only img_indices_ differs).*/
bool IsNormalDirective(const std::string& directive) {
    return directive == "Single" || directive == "All" ||
           directive == "Each" || directive == "From" ||
           directive == "Backward";
}

/*The v1 graph — jtml-production — the run shape today's Optimize() executes:
 * trunk (35)^6 / 20000 / dil 6 (cfm 0) → 2× branch (15,15,25,25,25,25) / 5000
 * / dil 4 (cfm 1) → leaf (3,3,15,3,3,3) / 5000 / dil 1 (cfm 2). Built from
 * the same settings_constants.h constants the loop consumes; cfm indices
 * 0/1/2 select the trunk/branch/leaf CostFunctionManagers. Engine runtime
 * dilation 6/4/1 (baseline.json's dilation_px {6,3,1} is the known-stale
 * docs-claim, reconciled by the U5 probe data in the hygiene pass).*/
StageGraph JtmlProductionGraph() {
    return StageGraph{
        "jtml-production",
        {{StageKind::Trunk, TRUNK_RANGE,
          static_cast<unsigned int>(TRUNK_BUDGET), 1u, 0u},
         {StageKind::Branch, BRANCH_RANGE,
          static_cast<unsigned int>(BRANCH_BUDGET),
          static_cast<unsigned int>(NUMBER_BRANCHES), 1u},
         {StageKind::Leaf, Z_SEARCH_RANGE,
          static_cast<unsigned int>(Z_SEARCH_BUDGET), 1u, 2u}}};
}

}  // namespace

StageScript BuildStageScript(
    const OptimizerSettings& settings, const std::string& directive) {
    ValidateSettings(settings);

    if (directive == "Sym_Trap") {
        /*U6-corrected: the Sym_Trap script is the leaf-only, no-search spec —
         * today's engine skips trunk AND branches (the `if (!sym_trap_call)`
         * guard wraps both; costCalls lands on 0; stageText stays Idle; only
         * CalculateSymTrap's 60 uncounted analysis evals run; the early return
         * skips the final UpdateDisplay). repeat=0 = init + dilate + emit +
         * CalculateSymTrap, NO RunDirectStage. The leaf spec mirrors the
         * engine's leaf-init gate (enable_leaf_); note the engine calls
         * CalculateSymTrap itself regardless of enable_leaf_ (a latent hazard
         * on an uninitialized leaf CFM — never hit with the default settings,
         * enable_leaf_ = true).*/
        StageScript script;
        if (settings.enable_leaf_) {
            script.push_back(
                StageSpec{StageKind::Leaf, settings.leaf_range,
                          static_cast<unsigned int>(settings.leaf_budget),
                          /*repeat=*/0u, /*cfm_index=*/2u});
        }
        return script;
    }

    if (!IsNormalDirective(directive)) {
        throw std::invalid_argument(
            "BuildStageScript: unrecognized optimization directive: '" +
            directive +
            "' (expected Single/All/Each/From/Backward/Sym_Trap)");
    }

    /*Verbatim transcription of the Optimize() loop's enabled-flag gating:
     * trunk always present; the branch group iff enable_branch_ &&
     * number_branches > 0 (the loop's init gate) with repeat =
     * number_branches (the loop's `enable_branch_ * number_branches` count,
     * materialized — a false flag zeroes the count exactly like the engine);
     * the leaf iff enable_leaf_. The five normal directives differ only in
     * frame selection (img_indices_), never in stage shape — one script for
     * all of them.*/
    StageScript script;
    script.push_back(
        StageSpec{StageKind::Trunk, settings.trunk_range,
                  static_cast<unsigned int>(settings.trunk_budget),
                  /*repeat=*/1u, /*cfm_index=*/0u});
    if (settings.enable_branch_ && settings.number_branches > 0) {
        script.push_back(
            StageSpec{StageKind::Branch, settings.branch_range,
                      static_cast<unsigned int>(settings.branch_budget),
                      static_cast<unsigned int>(settings.number_branches),
                      /*cfm_index=*/1u});
    }
    if (settings.enable_leaf_) {
        script.push_back(
            StageSpec{StageKind::Leaf, settings.leaf_range,
                      static_cast<unsigned int>(settings.leaf_budget),
                      /*repeat=*/1u, /*cfm_index=*/2u});
    }
    return script;
}

std::vector<unsigned int> CumulativeStageCaps(const StageScript& script) {
    /*budget_ semantics transcribed from the loop: budget_ = trunk_budget at
     * the trunk, then += branch_budget per branch repeat, then += leaf_budget
     * — one cap entry per RunDirectStage invocation (search run). repeat=0
     * (the Sym_Trap no-search leaf) contributes nothing — costCalls stays 0
     * (the U6 pin).*/
    std::vector<unsigned int> caps;
    unsigned int running = 0;
    for (const StageSpec& spec : script) {
        for (unsigned int i = 0; i < spec.repeat; ++i) {
            running += spec.budget;
            caps.push_back(running);
        }
    }
    return caps;
}

StageCostParams DeriveStageCostParams(
    const std::string& cost_function_name,
    std::vector<jta_cost_function::Parameter<int>> int_params,
    std::vector<jta_cost_function::Parameter<bool>> bool_params) {
    /*Verbatim relocation of the manager's per-manager parameter scan: the
     * "Dilation"/"DILATION"/"dilation" int-name match (last match wins — the
     * manager's loop overwrites), the ≤0 → 0 clamp, the DIRECT_MAHFOUZ → 3
     * special case (applied after the clamp, like the manager), and the six
     * dark-silhouette bool name variants (default false). The vectors are
     * copied by value exactly like the manager's inline scan copies them
     * (active_int_params / active_bool_params locals) — the Parameter
     * accessors are non-const, so the scan operates on the copies. NEVER the
     * CFM internals — the wizard regions of CostFunctionManager.* are
     * untouched; dilation stays cost-owned (the CFM parameter remains the
     * single source of truth, angle 04 R2-2).*/
    StageCostParams out;

    for (auto& p : int_params) {
        const std::string& name = p.getParameterName();
        if (name == "Dilation" || name == "DILATION" || name == "dilation") {
            out.dilation = p.getParameterValue();
        }
    }
    if (out.dilation <= 0) {
        out.dilation = 0;
    }
    if (cost_function_name == "DIRECT_MAHFOUZ") {
        out.dilation = 3;
    }

    for (auto& p : bool_params) {
        const std::string& name = p.getParameterName();
        if (name == "Black_Silhouette" || name == "Dark_Silhouette" ||
            name == "BLACK_SILHOUETTE" || name == "DARK_SILHOUETTE" ||
            name == "black_silhouette" || name == "dark_silhouette") {
            out.dark_silhouette = p.getParameterValue();
        }
    }
    return out;
}

const std::vector<std::string>& ReservedStubGraphNames() {
    /*Future kinds recorded as data-only stubs (R6), NOT exercised: the polish
     * stage (R5's future StageKind), the ML initializer prefix (origin
     * R15/R16), and the flood-direct-jta shape (the lineage 3-branch / 50k +
     * 15k×3 + 50k config with the (5,5,20,5,5,5) leaf, angle 04 R3-1).
     * Requesting one fails fast with the stub error below until a follow-up
     * plan registers it.*/
    static const std::vector<std::string> names = {
        "jtml-polish-stage",
        "jtml-initializer-prefix",
        "jtml-flood-direct-jta",
    };
    return names;
}

std::vector<StageGraph> ListStageGraphs() { return {JtmlProductionGraph()}; }

const StageGraph& StageGraphByName(const std::string& name) {
    static const StageGraph production = JtmlProductionGraph();
    if (name == production.name) {
        return production;
    }
    for (const std::string& stub : ReservedStubGraphNames()) {
        if (name == stub) {
            throw std::invalid_argument(
                "StageGraphByName: '" + name +
                "' is a reserved stub graph (future kind; data-only, not "
                "implemented)");
        }
    }
    throw std::invalid_argument(
        "StageGraphByName: unknown stage graph: '" + name + "'");
}

}  // namespace jta
