/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*Plan 008 U7 (Cut A): the optimizer-container pure surface — the StageScript
 * types, the named builders (BuildStageScript / DeriveStageCostParams), and
 * the named graph registry (ListStageGraphs / StageGraphByName) mirroring
 * CostFunctionManager::listCostFunctions (src/compute/CostFunctionManager.cpp)
 * and the golden-pinned mapping pattern of
 * jta::BuildCostFunctionRegistryEntries
 * (src/services/cost_function_registry.cpp). Zero production behavior change at
 * U7; since U9 (Cut B) the manager's Optimize() loop CONSUMES BuildStageScript
 * (stage_script_, built once in Initialize) — the named-graph REGISTRY below
 * is the validated configuration surface; wiring the manager to run a NAMED
 * graph (StageGraphByName instead of the builder) is a small pinned
 * follow-up. Requirements: R1 (stages-as-data), R2 (named registry), R4 (the
 * existing seams stay the execution surface), R5 (the schema must not preclude
 * biplane / tiered-dilation / polish), R6 (v1 ships jtml-production; stubs
 * allowed).
 *
 * The dedicated TU exists because the direct-compile test pattern
 * (docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md)
 * must compile these pure functions WITHOUT pulling optimizer_manager.cpp's
 * Qt/CUDA surface (placement per the run's angle 04 R2-2). Qt/GPU-free except
 * for the services OptimizerSettings parameter (Qt meta-type header only).*/

#ifndef OPTIMIZER_STAGE_SCRIPT_H
#define OPTIMIZER_STAGE_SCRIPT_H

#include <string>
#include <vector>

#include "compute/Parameter.h"
#include "domain/data_structures_6D.h"
#include "services/optimizer_settings.h"

namespace jta {

/*Stage kind — the lineage paper's stage taxonomy is exactly Trunk/Branch/Leaf
 * (Flood & Banks 2018, IEEE TMI 37(1):326-335; angle 04 R3-1); a future polish
 * kind is a schema note only (R5 — the registry's reserved stubs).*/
enum class StageKind : unsigned char { Trunk = 0, Branch = 1, Leaf = 2 };

/*One stage of the run script (angle 04 R2-2 schema, verified 1:1 against the
 * Optimize() loop blocks):
 *  - kind: the stage's search flavor;
 *  - range: the SetSearchRange input, applied per repeat iteration;
 *  - budget: per-repeat accumulator (budget_ += budget — cumulative semantics
 *    preserved; the running cost_function_calls_ offset resets ONLY at trunk);
 *  - repeat: RunDirectStage invocations. trunk/leaf = 1; branch =
 *    number_branches (materialized); repeat=0 expresses the Sym_Trap
 *    no-search leaf — init + dilate + emit (+ CalculateSymTrap) with NO
 *    search (U6-corrected: under Sym_Trap the engine's outer
 *    `if (!sym_trap_call)` guard at optimizer_manager.cpp wraps trunk AND
 *    branches, costCalls lands on 0, stageText stays Idle, only the 60
 *    uncounted CalculateSymTrap analysis evals run, and the early return
 *    skips the final UpdateDisplay);
 *  - cfm_index: 0/1/2 -> trunk_manager_ / branch_manager_ / leaf_manager_.*/
struct StageSpec {
    StageKind kind = StageKind::Trunk;
    Point6D range;
    unsigned int budget = 0;
    unsigned int repeat = 0;
    unsigned int cfm_index = 0;
};

using StageScript = std::vector<StageSpec>;

/*--- Convergence-tradeoff design note (angle 04 R3-2 deliverable) ----------
 * The stage sequence intentionally drops the DIRECT cover property at the
 * STAGE LOOP's cross-stage boundary: each stage re-seeds from the current
 * optimum and restarts on a hyper-rectangle that does NOT cover the previous
 * stage's domain (branch (15,15,25,25,25,25) ⊂ trunk (35)^6; leaf
 * (3,3,15,3,3,3) ⊂ branch — include/domain/settings_constants.h). This is
 * inherited lineage doctrine, not a future decision: "Therefore, by removing
 * unpromising regions of the domain, DIRECT-JTA sacrifices a notion of global
 * convergence for improved asymptotic performance" (Flood & Banks 2018, IEEE
 * TMI 37(1):326-335, the cover/restart passages).
 * A single DirectOptimizer run over its given range keeps DIRECT's per-box
 * cover guarantee; the guarantee fails only across stage boundaries. Restoring
 * cover (a "global" script whose branch/leaf ranges equal the trunk range)
 * "would negate the explore and exploit structure" of the lineage and is a
 * behavioral change requiring its own pin — never a silent fix.
 * DirectOptimizer::Options carries NO cover knob: cover is a stage-loop
 * property, intentionally dropped across stages, not a per-stage option.*/

/*Per-stage cost parameters derived from the CFM parameter registry — a
 * literal relocation of the manager's per-manager parameter scan
 * (optimizer_manager.cpp, the dilation/dark-silhouette block). NEVER the CFM
 * internals: the wizard regions of CostFunctionManager.* are untouched, and
 * dilation stays cost-owned (the CFM parameter remains the single source of
 * truth).*/
struct StageCostParams {
    /*The active CFM's Dilation value (the "Dilation"/"DILATION"/"dilation"
     * name match; ≤0 clamps to 0; DIRECT_MAHFOUZ forces 3 — exactly the
     * manager's scan).*/
    int dilation = 0;
    /*The active CFM's black/dark silhouette flag (the six bool name
     * variants), default false.*/
    bool dark_silhouette = false;
};

/*Reproduce the manager's parameter scan for one stage's CFM parameter
 * registry: last matching parameter wins, ≤0 clamp, the DIRECT_MAHFOUZ → 3
 * special case, and the six dark-silhouette bool name variants. The vectors
 * are taken by value exactly like the manager's inline scan copies them
 * (optimizer_manager.cpp: active_int_params / active_bool_params locals);
 * pass the getActiveCostFunctionClass()->getIntParameters()/getBoolParameters()
 * vectors verbatim.*/
StageCostParams DeriveStageCostParams(
    const std::string& cost_function_name,
    std::vector<jta_cost_function::Parameter<int>> int_params,
    std::vector<jta_cost_function::Parameter<bool>> bool_params);

/*Build the run script for `settings` under `directive`, transcribing the
 * Optimize() loop's enabled-flag gating verbatim:
 *  - "Single" / "All" / "Each" / "From" / "Backward" (the manager's directive
 *    strings; the typed jta::OptimizerRunControllerCore::Directive maps onto
 *    them) all share ONE stage shape — directives select frames
 *    (img_indices_), never stages: [Trunk always] + [Branch iff
 *    enable_branch_ && number_branches > 0, repeat = number_branches] +
 *    [Leaf iff enable_leaf_, repeat = 1];
 *  - "Sym_Trap" yields the U6-corrected leaf-only script [{Leaf, repeat=0}]
 *    (gated on enable_leaf_ like the engine's leaf-init block) — NOT
 *    {Trunk, Branch, Leaf repeat=0}.
 * Error path: an unrecognized directive or a negative budget / negative
 * number_branches fails fast with a clear std::invalid_argument (the engine
 * silently accepts corrupted settings; the pure builder does not).*/
StageScript BuildStageScript(
    const OptimizerSettings& settings,
    const std::string& directive);

/*The cumulative budget caps the run lands on, one entry per search run
 * (RunDirectStage invocation), transcribing budget_ = trunk_budget at the
 * trunk and budget_ += stage budget per repeat thereafter. This is the U6
 * stage-bookkeeping gate (costCalls on 20/25/30/35k for jtml-production).
 * repeat=0 (the Sym_Trap no-search leaf) contributes nothing — the caps are
 * empty and costCalls stays 0 (the U6 sym-trap pin).*/
std::vector<unsigned int> CumulativeStageCaps(const StageScript& script);

/*A named run graph: registered C++ data (the listCostFunctions pattern).*/
struct StageGraph {
    std::string name;
    StageScript stages;
};

/*All registered graphs. v1 = "jtml-production":
 *   [{Trunk,  (35)^6,              20000, 1, cfm 0},
 *    {Branch, (15,15,25,25,25,25), 5000,  2, cfm 1},
 *    {Leaf,   (3,3,15,3,3,3),      5000,  1, cfm 2}]
 * — the run shape today's Optimize() executes, built from the same
 * settings_constants.h constants the loop consumes. Engine runtime dilation is
 * 6/4/1 (baseline.json's dilation_px {6,3,1} is the known-stale docs-claim,
 * reconciled by the U5 probe data in the hygiene pass).*/
std::vector<StageGraph> ListStageGraphs();

/*Lookup by name — fails fast with a clear std::invalid_argument for unknown
 * names and for the reserved stub names (data-only future kinds, R6).*/
const StageGraph& StageGraphByName(const std::string& name);

/*Reserved stub graph names — future kinds recorded as data-only, NOT
 * exercised: the polish stage (R5), the ML initializer prefix (origin
 * R15/R16), and the flood-direct-jta shape (the lineage 3-branch / 50k +
 * 15k×3 + 50k config with the (5,5,20,5,5,5) leaf). Requesting one fails fast
 * until a follow-up plan registers it.*/
const std::vector<std::string>& ReservedStubGraphNames();

}  // namespace jta

#endif /* OPTIMIZER_STAGE_SCRIPT_H */
