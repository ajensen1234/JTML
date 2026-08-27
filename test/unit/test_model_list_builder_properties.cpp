// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel property-based tests for the pure ModelListBuilder (plan U7, U10).
//
// PBT complements the deterministic unit tests by probing invariants the name-
// uniquification must satisfy across many random draws, regardless of the
// quirky two-pass dedup implementation:
//   - length preservation: exactly one display name per input file,
//   - collision-freedom: no display name equals any *existing* loaded name
//     (the contract the load slot relies on before adding to the model list),
//   - determinism: same input -> same output (a regression drifts renames
//     silently otherwise).
// These are the properties we want to lock down after extracting the dedup
// out of MainScreen, so a future edit can't silently change what is displayed
// for duplicate-name loads (R15 / R9).

#include <hegel/hegel.h>

#include <catch2/catch_test_macros.hpp>
#include <set>
#include <string>
#include <vector>

#include "domain/model_list_builder.h"

namespace gs = hegel::generators;

namespace {

// Draws a list of model names from a small vocabulary so collisions are likely.
auto NameVector() {
    auto name_gen = gs::sampled_from<std::string>({
        "A",
        "B",
        "C",
        "femur",
        "tibia",
    });
    return gs::vectors(name_gen, {.min_size = 0, .max_size = 6});
}

}  // namespace

TEST_CASE(
    "ModelListBuilder[PBT]: length preserved per input",
    "[model_list_builder][pbt]") {
    hegel::test([&](hegel::TestCase& tc) {
        auto new_names = tc.draw(NameVector());
        auto existing = tc.draw(NameVector());

        auto out =
            jta::ModelListBuilder::UniquifyModelNames(new_names, existing);
        REQUIRE(out.size() == new_names.size());  // one name per new file
    });
}

TEST_CASE(
    "ModelListBuilder[PBT]: no output name collides with existing",
    "[model_list_builder][pbt]") {
    hegel::test([&](hegel::TestCase& tc) {
        auto new_names = tc.draw(NameVector());
        auto existing = tc.draw(NameVector());

        auto out =
            jta::ModelListBuilder::UniquifyModelNames(new_names, existing);
        std::set<std::string> existing_set(existing.begin(), existing.end());
        for (const auto& n : out) {
            REQUIRE(existing_set.count(n) == 0);
        }
    });
}

TEST_CASE(
    "ModelListBuilder[PBT]: deterministic given the same input",
    "[model_list_builder][pbt]") {
    hegel::test([&](hegel::TestCase& tc) {
        auto new_names = tc.draw(NameVector());
        auto existing = tc.draw(NameVector());

        auto out1 =
            jta::ModelListBuilder::UniquifyModelNames(new_names, existing);
        auto out2 =
            jta::ModelListBuilder::UniquifyModelNames(new_names, existing);
        REQUIRE(out1 == out2);
    });
}
