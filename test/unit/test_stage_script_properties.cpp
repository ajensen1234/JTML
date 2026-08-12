// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel property-based tests for the pure stage-script surface (plan 008 U7,
// Cut A). PBT complements the deterministic twin (test_stage_script.cpp) by
// probing invariants across randomized-but-valid settings:
//   - budget sums: CumulativeStageCaps reproduces the cumulative-budget
//     arithmetic (budget_ = trunk at the trunk, += per branch repeat, += leaf)
//     — the caps gate the multi-stage oracle lands on;
//   - stage order: the produced script is always [Trunk, Branch?, Leaf?] —
//     the loop's enabled-flag gating never reorders;
//   - no negative/zero budgets and positive search repeats in the produced
//     script (the error domain — negative budgets / unknown directives — is
//     the deterministic twin's, deliberately excluded from the drawn domain);
//   - directive invariance: the five normal directives share one stage shape;
//   - Sym_Trap: the U6-corrected leaf-only no-search script [{Leaf, repeat=0}]
//     with empty caps;
//   - determinism: same settings -> same script.
//
// Pure: the builder + optimizer_settings + data_structures_6D sources compile
// directly (Qt only via the header-only QMetaType macro), with the standard
// hegel link/rpath recipe.

#include <string>
#include <vector>

#include <hegel/hegel.h>

#include <catch2/catch_test_macros.hpp>

#include "coordinator/optimizer_stage_script.h"

namespace gs = hegel::generators;

namespace {

/*A randomized-but-valid settings draw: positive budgets, non-negative branch
 * count, random stage flags, canonical ranges (ranges are shape-carried but
 * invariant-neutral — the policy fields are what the invariants read).*/
struct DrawnSettings {
    OptimizerSettings s;
    int trunk_budget = 0;
    int branch_budget = 0;
    int leaf_budget = 0;
    int number_branches = 0;
    bool enable_branch = false;
    bool enable_leaf = false;
};

DrawnSettings DrawSettings(hegel::TestCase& tc) {
    DrawnSettings out;
    out.trunk_budget = tc.draw(
        "trunk_budget", gs::integers<int>({.min_value = 1, .max_value = 40000}));
    out.branch_budget = tc.draw(
        "branch_budget", gs::integers<int>({.min_value = 1, .max_value = 20000}));
    out.leaf_budget = tc.draw(
        "leaf_budget", gs::integers<int>({.min_value = 1, .max_value = 20000}));
    out.number_branches = tc.draw(
        "number_branches", gs::integers<int>({.min_value = 0, .max_value = 8}));
    out.enable_branch = tc.draw("enable_branch", gs::booleans());
    out.enable_leaf = tc.draw("enable_leaf", gs::booleans());

    OptimizerSettings s;  // canonical ranges via the settings ctor
    s.trunk_budget = out.trunk_budget;
    s.branch_budget = out.branch_budget;
    s.leaf_budget = out.leaf_budget;
    s.number_branches = out.number_branches;
    s.enable_branch_ = out.enable_branch;
    s.enable_leaf_ = out.enable_leaf;
    out.s = s;
    return out;
}

bool SameSpec(const jta::StageSpec& a, const jta::StageSpec& b) {
    return a.kind == b.kind && a.budget == b.budget &&
           a.repeat == b.repeat && a.cfm_index == b.cfm_index &&
           a.range.x == b.range.x && a.range.y == b.range.y &&
           a.range.z == b.range.z && a.range.xa == b.range.xa &&
           a.range.ya == b.range.ya && a.range.za == b.range.za;
}

bool SameScript(const jta::StageScript& a, const jta::StageScript& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!SameSpec(a[i], b[i])) return false;
    }
    return true;
}

const char* const kNormalDirectives[] = {
    "Single", "All", "Each", "From", "Backward"};

}  // namespace

TEST_CASE(
    "stage_script[PBT]: budget sums to the cumulative caps for randomized "
    "settings",
    "[stage_script][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            const DrawnSettings d = DrawSettings(tc);
            const jta::StageScript script =
                jta::BuildStageScript(d.s, "Single");
            const std::vector<unsigned int> caps =
                jta::CumulativeStageCaps(script);

            /*One cap entry per search run: trunk, then branch per repeat,
             * then leaf.*/
            const size_t expected_cap_count =
                1u + (d.enable_branch && d.number_branches > 0
                          ? static_cast<size_t>(d.number_branches)
                          : 0u) +
                (d.enable_leaf ? 1u : 0u);
            REQUIRE(caps.size() == expected_cap_count);

            /*budget_ = trunk_budget at the trunk (the loop's reset).*/
            REQUIRE(caps.front() == static_cast<unsigned int>(d.trunk_budget));

            /*Strictly increasing (each search run adds a positive budget).*/
            for (size_t i = 1; i < caps.size(); ++i) {
                REQUIRE(caps[i] > caps[i - 1]);
            }

            /*The final cap == the independent recomputation of the cumulative
             * total (trunk + per-repeat branch + leaf).*/
            const unsigned int expected_total =
                static_cast<unsigned int>(d.trunk_budget) +
                (d.enable_branch && d.number_branches > 0
                     ? static_cast<unsigned int>(d.branch_budget) *
                           static_cast<unsigned int>(d.number_branches)
                     : 0u) +
                (d.enable_leaf ? static_cast<unsigned int>(d.leaf_budget) : 0u);
            REQUIRE(caps.back() == expected_total);
        },
        hegel::Settings{.test_cases = 400});
}

TEST_CASE(
    "stage_script[PBT]: stage order preserved — the script is always "
    "[Trunk, Branch?, Leaf?]",
    "[stage_script][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            const DrawnSettings d = DrawSettings(tc);
            const jta::StageScript script =
                jta::BuildStageScript(d.s, "Single");

            /*Trunk is always first and present; kinds never reorder
             * (StageKind::Trunk < Branch < Leaf); cfm_index follows the kind
             * (0/1/2 -> trunk/branch/leaf managers).*/
            REQUIRE(script.size() >= 1);
            REQUIRE(script.front().kind == jta::StageKind::Trunk);
            for (size_t i = 0; i < script.size(); ++i) {
                if (i > 0) {
                    REQUIRE(
                        static_cast<unsigned char>(script[i].kind) >=
                        static_cast<unsigned char>(script[i - 1].kind));
                }
                REQUIRE(
                    script[i].cfm_index ==
                    static_cast<unsigned int>(script[i].kind));
            }

            /*Exact presence mapping: branch iff the loop's init gate
             * (enable_branch_ && number_branches > 0), leaf iff
             * enable_leaf_ — no phantom stages.*/
            const bool branch_present =
                script.size() >= 2 &&
                script[1].kind == jta::StageKind::Branch;
            const bool leaf_present =
                script.back().kind == jta::StageKind::Leaf;
            REQUIRE(
                branch_present ==
                (d.enable_branch && d.number_branches > 0));
            REQUIRE(leaf_present == d.enable_leaf);

            /*Repeat semantics: trunk/leaf repeat=1, branch repeat =
             * number_branches (materialized).*/
            REQUIRE(script[0].repeat == 1u);
            if (branch_present) {
                REQUIRE(script[1].repeat ==
                        static_cast<unsigned int>(d.number_branches));
            }
            if (leaf_present) {
                REQUIRE(script.back().repeat == 1u);
            }
        },
        hegel::Settings{.test_cases = 400});
}

TEST_CASE(
    "stage_script[PBT]: no negative/zero budgets and positive search repeats "
    "in the produced script",
    "[stage_script][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            const DrawnSettings d = DrawSettings(tc);
            const jta::StageScript script =
                jta::BuildStageScript(d.s, "Single");
            for (const jta::StageSpec& spec : script) {
                REQUIRE(spec.budget > 0u);
                REQUIRE(spec.repeat > 0u);  // every spec here is a search stage
            }
        },
        hegel::Settings{.test_cases = 400});
}

TEST_CASE(
    "stage_script[PBT]: the five normal directives share one stage shape",
    "[stage_script][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            const DrawnSettings d = DrawSettings(tc);
            const jta::StageScript reference =
                jta::BuildStageScript(d.s, "Single");
            for (const char* directive : kNormalDirectives) {
                REQUIRE(SameScript(
                    jta::BuildStageScript(d.s, directive), reference));
            }
        },
        hegel::Settings{.test_cases = 300});
}

TEST_CASE(
    "stage_script[PBT]: Sym_Trap yields the leaf-only no-search script with "
    "empty caps",
    "[stage_script][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            const DrawnSettings d = DrawSettings(tc);
            const jta::StageScript script =
                jta::BuildStageScript(d.s, "Sym_Trap");

            /*U6-corrected: leaf-only (gated on enable_leaf_ like the engine's
             * leaf-init block), repeat=0 (no search), cfm 2 (the leaf
             * manager), carrying the leaf range/budget.*/
            REQUIRE(script.size() == (d.enable_leaf ? 1u : 0u));
            if (d.enable_leaf) {
                const jta::StageSpec& spec = script[0];
                REQUIRE(spec.kind == jta::StageKind::Leaf);
                REQUIRE(spec.repeat == 0u);
                REQUIRE(spec.cfm_index == 2u);
                REQUIRE(spec.budget ==
                        static_cast<unsigned int>(d.leaf_budget));
                REQUIRE(spec.range.x == d.s.leaf_range.x);
                REQUIRE(spec.range.y == d.s.leaf_range.y);
                REQUIRE(spec.range.z == d.s.leaf_range.z);
                REQUIRE(spec.range.xa == d.s.leaf_range.xa);
                REQUIRE(spec.range.ya == d.s.leaf_range.ya);
                REQUIRE(spec.range.za == d.s.leaf_range.za);
            }

            /*repeat=0 contributes nothing — costCalls stays 0 (the U6 pin).*/
            REQUIRE(jta::CumulativeStageCaps(script).empty());
        },
        hegel::Settings{.test_cases = 300});
}

TEST_CASE(
    "stage_script[PBT]: determinism — same settings, same script",
    "[stage_script][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            const DrawnSettings d = DrawSettings(tc);
            const jta::StageScript first =
                jta::BuildStageScript(d.s, "Single");
            const jta::StageScript second =
                jta::BuildStageScript(d.s, "Single");
            REQUIRE(SameScript(first, second));
        },
        hegel::Settings{.test_cases = 300});
}
