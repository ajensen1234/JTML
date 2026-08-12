// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 008 U7 (Cut A): deterministic pins for the pure stage-script surface
// (include/coordinator/optimizer_stage_script.h). The builders ARE the spec —
// they transcribe the manager's Optimize() loop shape and its per-manager
// parameter scan verbatim, so the unit pins ARE the schema contract:
//   - happy path: default settings -> the exact three-spec sequence
//     [{Trunk,20000,1,cfm0},{Branch,5000,2,cfm1},{Leaf,5000,1,cfm2}] with the
//     canonical ranges, for all five normal directives;
//   - caps: CumulativeStageCaps lands on the cumulative 20/25/30/35k (the U6
//     stage-bookkeeping gate);
//   - edges: number_branches=0 / enable_branch_=false / enable_leaf_=false ->
//     stage absent, caps recompute;
//   - edge: repeat=0 (Sym_Trap) -> the U6-corrected leaf-only script
//     [{Leaf, repeat=0}] (NOT {Trunk, Branch, Leaf repeat=0}) with empty caps;
//   - error path: an invalid directive or a negative budget fails fast with a
//     clear error;
//   - integration: DeriveStageCostParams consumes the REAL manager parameter
//     vectors (getActiveCostFunctionClass()->getIntParameters()/... ) and
//     reproduces the manager's runtime scan for the production settings
//     (dilation 6/4/1, the DIRECT_MAHFOUZ 3 special case, the clamps, and the
//     dark-silhouette name variants);
//   - registry: jtml-production is registered data with the golden-pinned
//     shape; unknown / reserved-stub names fail fast.
//
// Fixture recipe: the deterministic target direct-compiles the builder +
// optimizer_settings + data_structures_6D sources and links the real
// CostFunctionManager/CostFunction from jtml_compute (constructor + parameter
// surface only — CPU-only, no CUDA calls at runtime), the same recipe as
// jtml.cost_function_registry. The hegel PBT twin
// (test_stage_script_properties.cpp) owns the randomized invariants.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#include "compute/CostFunctionManager.h"
#include "coordinator/optimizer_stage_script.h"

namespace {

/*The five normal directives — frame-selection directives, one stage shape.*/
const char* const kNormalDirectives[] = {
    "Single", "All", "Each", "From", "Backward"};

void RequirePointEq(
    const Point6D& p, double x, double y, double z, double xa, double ya,
    double za) {
    REQUIRE(p.x == Catch::Approx(x));
    REQUIRE(p.y == Catch::Approx(y));
    REQUIRE(p.z == Catch::Approx(z));
    REQUIRE(p.xa == Catch::Approx(xa));
    REQUIRE(p.ya == Catch::Approx(ya));
    REQUIRE(p.za == Catch::Approx(za));
}

void RequireSpec(
    const jta::StageSpec& spec, jta::StageKind kind, double r0, double r1,
    double r2, double r3, double r4, double r5, unsigned int budget,
    unsigned int repeat, unsigned int cfm_index) {
    REQUIRE(spec.kind == kind);
    RequirePointEq(spec.range, r0, r1, r2, r3, r4, r5);
    REQUIRE(spec.budget == budget);
    REQUIRE(spec.repeat == repeat);
    REQUIRE(spec.cfm_index == cfm_index);
}

/*Require that `call` throws std::invalid_argument whose message contains
 * `expected_fragment` (the error path contract — fail fast, clear error).*/
template <typename Callable>
void RequireInvalidArgument(Callable&& call, const std::string& expected_fragment) {
    try {
        call();
        FAIL("expected std::invalid_argument");
    } catch (const std::invalid_argument& e) {
        REQUIRE(
            std::string(e.what()).find(expected_fragment) !=
            std::string::npos);
    }
}

/*The production manager configuration (fresh managers + the branch/leaf
 * DIRECT_DILATION Dilation=4/1 overrides — exactly what the widgets first-run
 * path applies, see jtml.cost_function_registry's FirstRunManagers).*/
struct ProductionManagers {
    ProductionManagers() :
        trunk(Stage::Trunk), branch(Stage::Branch), leaf(Stage::Leaf) {
        branch.getCostFunctionClass("DIRECT_DILATION")
            ->setIntParameterValue("Dilation", 4);
        leaf.getCostFunctionClass("DIRECT_DILATION")
            ->setIntParameterValue("Dilation", 1);
    }
    jta_cost_function::CostFunctionManager trunk;
    jta_cost_function::CostFunctionManager branch;
    jta_cost_function::CostFunctionManager leaf;
};

/*The manager's runtime scan call site (optimizer_manager.cpp:237-345):
 * DeriveStageCostParams over the REAL getActiveCostFunctionClass() vectors.*/
jta::StageCostParams DeriveFor(jta_cost_function::CostFunctionManager& m) {
    return jta::DeriveStageCostParams(
        m.getActiveCostFunction(),
        m.getActiveCostFunctionClass()->getIntParameters(),
        m.getActiveCostFunctionClass()->getBoolParameters());
}

}  // namespace

/*---------------------------------------------------------------------------*/

TEST_CASE(
    "stage_script: default settings produce the canonical three-spec script "
    "for every normal directive",
    "[stage_script]") {
    /*(a) Happy path: OptimizerSettings() -> the exact three-spec sequence with
     * the canonical ranges, identically for all five normal directives (they
     * select frames, not stages).*/
    for (const char* directive : kNormalDirectives) {
        INFO("directive: " << directive);
        const jta::StageScript script =
            jta::BuildStageScript(OptimizerSettings(), directive);
        REQUIRE(script.size() == 3);

        RequireSpec(
            script[0], jta::StageKind::Trunk, 35, 35, 35, 35, 35, 35, 20000,
            1, 0);
        RequireSpec(
            script[1], jta::StageKind::Branch, 15, 15, 25, 25, 25, 25, 5000,
            2, 1);
        RequireSpec(
            script[2], jta::StageKind::Leaf, 3, 3, 15, 3, 3, 3, 5000, 1, 2);
    }

    /*Determinism + directive invariance: all five scripts are identical.*/
    const jta::StageScript reference =
        jta::BuildStageScript(OptimizerSettings(), "Single");
    for (const char* directive : kNormalDirectives) {
        const jta::StageScript again =
            jta::BuildStageScript(OptimizerSettings(), directive);
        REQUIRE(again.size() == reference.size());
        for (size_t i = 0; i < reference.size(); ++i) {
            REQUIRE(again[i].kind == reference[i].kind);
            REQUIRE(again[i].budget == reference[i].budget);
            REQUIRE(again[i].repeat == reference[i].repeat);
            REQUIRE(again[i].cfm_index == reference[i].cfm_index);
            RequirePointEq(
                again[i].range, reference[i].range.x, reference[i].range.y,
                reference[i].range.z, reference[i].range.xa,
                reference[i].range.ya, reference[i].range.za);
        }
    }
}

TEST_CASE(
    "stage_script: cumulative caps land on the 20/25/30/35k stage-bookkeeping "
    "gate",
    "[stage_script]") {
    /*(a) The default script's caps ARE the U6 oracle's cumulative caps: one
     * entry per search run (trunk, branch x2, leaf).*/
    const jta::StageScript script =
        jta::BuildStageScript(OptimizerSettings(), "Single");
    const std::vector<unsigned int> caps = jta::CumulativeStageCaps(script);
    REQUIRE(caps.size() == 4);
    REQUIRE(caps[0] == 20000);
    REQUIRE(caps[1] == 25000);
    REQUIRE(caps[2] == 30000);
    REQUIRE(caps[3] == 35000);
}

TEST_CASE(
    "stage_script: number_branches=0 / branch or leaf disabled drops the "
    "stage and recomputes the caps",
    "[stage_script]") {
    OptimizerSettings settings;
    settings.number_branches = 0;
    jta::StageScript script = jta::BuildStageScript(settings, "Single");
    REQUIRE(script.size() == 2);
    RequireSpec(
        script[0], jta::StageKind::Trunk, 35, 35, 35, 35, 35, 35, 20000, 1, 0);
    RequireSpec(
        script[1], jta::StageKind::Leaf, 3, 3, 15, 3, 3, 3, 5000, 1, 2);
    REQUIRE(jta::CumulativeStageCaps(script).size() == 2);
    REQUIRE(jta::CumulativeStageCaps(script).back() == 25000);

    /*The engine's branch gate is `enable_branch_ && number_branches > 0` and
     * its repeat count is `enable_branch_ * number_branches` — a false flag
     * zeroes the count exactly like number_branches=0.*/
    settings = OptimizerSettings();
    settings.enable_branch_ = false;
    script = jta::BuildStageScript(settings, "Single");
    REQUIRE(script.size() == 2);
    REQUIRE(script[0].kind == jta::StageKind::Trunk);
    REQUIRE(script[1].kind == jta::StageKind::Leaf);
    REQUIRE(jta::CumulativeStageCaps(script).back() == 25000);

    /*Leaf disabled -> trunk + branch only; both disabled -> trunk only.*/
    settings = OptimizerSettings();
    settings.enable_leaf_ = false;
    script = jta::BuildStageScript(settings, "Single");
    REQUIRE(script.size() == 2);
    RequireSpec(
        script[0], jta::StageKind::Trunk, 35, 35, 35, 35, 35, 35, 20000, 1, 0);
    RequireSpec(
        script[1], jta::StageKind::Branch, 15, 15, 25, 25, 25, 25, 5000, 2, 1);
    const std::vector<unsigned int> caps = jta::CumulativeStageCaps(script);
    REQUIRE(caps.size() == 3);
    REQUIRE(caps[0] == 20000);
    REQUIRE(caps[1] == 25000);
    REQUIRE(caps[2] == 30000);

    settings.enable_branch_ = false;
    script = jta::BuildStageScript(settings, "Single");
    REQUIRE(script.size() == 1);
    REQUIRE(script[0].kind == jta::StageKind::Trunk);
    REQUIRE(jta::CumulativeStageCaps(script).size() == 1);
    REQUIRE(jta::CumulativeStageCaps(script)[0] == 20000);
}

TEST_CASE(
    "stage_script: Sym_Trap is the U6-corrected leaf-only no-search script "
    "[{Leaf, repeat=0}]",
    "[stage_script]") {
    /*(b) Edge: the Sym_Trap directive -> [{Leaf, leaf_range, leaf_budget,
     * repeat=0, cfm 2}] — NOT {Trunk, Branch, Leaf repeat=0}: the engine's
     * outer `if (!sym_trap_call)` guard wraps trunk AND branches (costCalls
     * lands on 0, only CalculateSymTrap's 60 uncounted analysis evals run).*/
    const jta::StageScript script =
        jta::BuildStageScript(OptimizerSettings(), "Sym_Trap");
    REQUIRE(script.size() == 1);
    RequireSpec(
        script[0], jta::StageKind::Leaf, 3, 3, 15, 3, 3, 3, 5000, 0, 2);

    /*repeat=0 contributes no cap entry — costCalls stays 0 (the U6 pin).*/
    REQUIRE(jta::CumulativeStageCaps(script).empty());

    /*The leaf spec mirrors the engine's leaf-init gate: enable_leaf_ false ->
     * no leaf init/dilate/emit -> empty script.*/
    OptimizerSettings settings;
    settings.enable_leaf_ = false;
    REQUIRE(jta::BuildStageScript(settings, "Sym_Trap").empty());
}

TEST_CASE(
    "stage_script: an invalid directive or a negative budget fails fast with "
    "a clear error",
    "[stage_script]") {
    /*(e) Error path.*/
    const OptimizerSettings settings;
    RequireInvalidArgument(
        [&]() { jta::BuildStageScript(settings, "Bogus"); },
        "unrecognized optimization directive");
    RequireInvalidArgument(
        [&]() { jta::BuildStageScript(settings, ""); },
        "unrecognized optimization directive");

    OptimizerSettings negative = settings;
    negative.trunk_budget = -1;
    RequireInvalidArgument(
        [&]() { jta::BuildStageScript(negative, "Single"); },
        "negative trunk_budget");

    negative = settings;
    negative.branch_budget = -5000;
    RequireInvalidArgument(
        [&]() { jta::BuildStageScript(negative, "Single"); },
        "negative branch_budget");

    negative = settings;
    negative.leaf_budget = -1;
    RequireInvalidArgument(
        [&]() { jta::BuildStageScript(negative, "Single"); },
        "negative leaf_budget");

    negative = settings;
    negative.number_branches = -2;
    RequireInvalidArgument(
        [&]() { jta::BuildStageScript(negative, "Single"); },
        "negative number_branches");

    /*The validation applies under Sym_Trap too.*/
    RequireInvalidArgument(
        [&]() { jta::BuildStageScript(negative, "Sym_Trap"); },
        "negative number_branches");
}

TEST_CASE(
    "stage_script: DeriveStageCostParams transcribes the manager's parameter "
    "scan (pure cases)",
    "[stage_script]") {
    /*(d) The scan's name-match / clamp / special-case behavior, pinned on raw
     * parameter data.*/

    /*Production DIRECT_DILATION dilation values 6/4/1.*/
    REQUIRE(
        jta::DeriveStageCostParams(
            "DIRECT_DILATION",
            {jta_cost_function::Parameter<int>("Dilation", 6)}, {})
            .dilation == 6);
    REQUIRE(
        jta::DeriveStageCostParams(
            "DIRECT_DILATION",
            {jta_cost_function::Parameter<int>("Dilation", 4)}, {})
            .dilation == 4);
    REQUIRE(
        jta::DeriveStageCostParams(
            "DIRECT_DILATION",
            {jta_cost_function::Parameter<int>("Dilation", 1)}, {})
            .dilation == 1);

    /*The three int-name variants match; the ≤0 clamp forces 0.*/
    for (const char* name : {"Dilation", "DILATION", "dilation"}) {
        REQUIRE(
            jta::DeriveStageCostParams(
                "DIRECT_DILATION",
                {jta_cost_function::Parameter<int>(name, 6)}, {})
                .dilation == 6);
    }
    REQUIRE(
        jta::DeriveStageCostParams(
            "DIRECT_DILATION",
            {jta_cost_function::Parameter<int>("Dilation", 0)}, {})
            .dilation == 0);
    REQUIRE(
        jta::DeriveStageCostParams(
            "DIRECT_DILATION",
            {jta_cost_function::Parameter<int>("Dilation", -3)}, {})
            .dilation == 0);

    /*Last matching parameter wins (the manager's loop overwrites).*/
    REQUIRE(
        jta::DeriveStageCostParams(
            "DIRECT_DILATION",
            {jta_cost_function::Parameter<int>("Dilation", 6),
             jta_cost_function::Parameter<int>("Dilation", 2)},
            {})
            .dilation == 2);

    /*No params -> 0 / false (the manager's defaults).*/
    const jta::StageCostParams empty =
        jta::DeriveStageCostParams("DIRECT_DILATION", {}, {});
    REQUIRE(empty.dilation == 0);
    REQUIRE(empty.dark_silhouette == false);

    /*The DIRECT_MAHFOUZ special case: no Dilation param, Black_Silhouette
     * true -> dilation 3, dark true (the manager forces 3 after the clamp).*/
    const jta::StageCostParams mahfouz = jta::DeriveStageCostParams(
        "DIRECT_MAHFOUZ", {},
        {jta_cost_function::Parameter<bool>("Black_Silhouette", true)});
    REQUIRE(mahfouz.dilation == 3);
    REQUIRE(mahfouz.dark_silhouette == true);

    /*All six dark-silhouette bool name variants set the flag.*/
    for (const char* name :
         {"Black_Silhouette", "Dark_Silhouette", "BLACK_SILHOUETTE",
          "DARK_SILHOUETTE", "black_silhouette", "dark_silhouette"}) {
        REQUIRE(
            jta::DeriveStageCostParams(
                "DIRECT_DILATION", {},
                {jta_cost_function::Parameter<bool>(name, true)})
                .dark_silhouette == true);
    }
    /*Unrelated bool params leave the flag at the default false.*/
    REQUIRE(
        jta::DeriveStageCostParams(
            "DIRECT_DILATION", {},
            {jta_cost_function::Parameter<bool>("X_TRANS", true)})
            .dark_silhouette == false);
}

TEST_CASE(
    "stage_script: DeriveStageCostParams matches the manager's runtime "
    "parameter scan for the production settings (dilation 6/4/1)",
    "[stage_script]") {
    /*(d) Integration: the derivation consumes the REAL manager parameter
     * vectors (getActiveCostFunctionClass()->getIntParameters()/... — the
     * exact call sites the Optimize() loop's inline scan reads) and returns
     * the engine runtime dilation 6/4/1 for the production configuration.*/
    ProductionManagers managers;

    const jta::StageCostParams trunk = DeriveFor(managers.trunk);
    const jta::StageCostParams branch = DeriveFor(managers.branch);
    const jta::StageCostParams leaf = DeriveFor(managers.leaf);

    REQUIRE(trunk.dilation == 6);
    REQUIRE(branch.dilation == 4);
    REQUIRE(leaf.dilation == 1);
    REQUIRE(trunk.dark_silhouette == false);
    REQUIRE(branch.dark_silhouette == false);
    REQUIRE(leaf.dark_silhouette == false);

    /*The DIRECT_MAHFOUZ special case through the real manager: dilation 3 +
     * the Black_Silhouette=true bool.*/
    jta_cost_function::CostFunctionManager mahfouz(Stage::Trunk);
    mahfouz.setActiveCostFunction("DIRECT_MAHFOUZ");
    const jta::StageCostParams mahfouz_params = DeriveFor(mahfouz);
    REQUIRE(mahfouz_params.dilation == 3);
    REQUIRE(mahfouz_params.dark_silhouette == true);
}

TEST_CASE(
    "stage_script: the graph registry registers jtml-production as golden-"
    "pinned data",
    "[stage_script]") {
    /*(c) Registry: v1 = jtml-production, the exact three-spec shape (the
     * golden-pinned mapping pattern of BuildCostFunctionRegistryEntries).*/
    const std::vector<jta::StageGraph> graphs = jta::ListStageGraphs();
    REQUIRE(graphs.size() == 1);
    REQUIRE(graphs[0].name == "jtml-production");
    REQUIRE(graphs[0].stages.size() == 3);
    RequireSpec(
        graphs[0].stages[0], jta::StageKind::Trunk, 35, 35, 35, 35, 35, 35,
        20000, 1, 0);
    RequireSpec(
        graphs[0].stages[1], jta::StageKind::Branch, 15, 15, 25, 25, 25, 25,
        5000, 2, 1);
    RequireSpec(
        graphs[0].stages[2], jta::StageKind::Leaf, 3, 3, 15, 3, 3, 3, 5000, 1,
        2);

    /*Lookup by name + its caps land on the cumulative 20/25/30/35k.*/
    const jta::StageGraph& production = jta::StageGraphByName("jtml-production");
    REQUIRE(production.stages.size() == 3);
    const std::vector<unsigned int> caps =
        jta::CumulativeStageCaps(production.stages);
    REQUIRE(caps.size() == 4);
    REQUIRE(caps[0] == 20000);
    REQUIRE(caps[1] == 25000);
    REQUIRE(caps[2] == 30000);
    REQUIRE(caps[3] == 35000);

    /*Determinism: repeated lookups / listings are identical data.*/
    REQUIRE(
        jta::StageGraphByName("jtml-production").stages.size() ==
        production.stages.size());
    REQUIRE(jta::ListStageGraphs().size() == graphs.size());

    /*Reserved stubs are recorded data-only (future kinds, R6) and fail fast
     * with the stub error when requested.*/
    const std::vector<std::string>& stubs = jta::ReservedStubGraphNames();
    REQUIRE(stubs.size() == 3);
    for (const std::string& stub : stubs) {
        RequireInvalidArgument(
            [&]() { jta::StageGraphByName(stub); }, "reserved stub graph");
    }

    /*Unknown names fail fast too.*/
    RequireInvalidArgument(
        [&]() { jta::StageGraphByName("no-such-graph"); }, "unknown stage graph");
}
