// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel (property-based testing) harness smoke test -- plan U6 experiment.
//
// Verifies that hegel-cpp configures, links, and runs inside a Catch2 target on
// this pixi/CMake build, and that the test-cases/shrinking path works at all.
// This is the "does CMake play nice" gate; if this ever becomes a maintenance
// burden it can be dropped without touching the DirectOptimizer PBT layer.

#include <hegel/hegel.h>

#include <catch2/catch_test_macros.hpp>

namespace gs = hegel::generators;

TEST_CASE("hegel harness: integer addition commutes",
          "[hegel][harness]") {
    hegel::test([](hegel::TestCase& tc) {
        auto a = tc.draw(gs::integers<int64_t>());
        auto b = tc.draw(gs::integers<int64_t>());
        REQUIRE(a + b == b + a);
    });
}

TEST_CASE("hegel harness: hypothesis-style generation runs many cases",
          "[hegel][harness]") {
    // Default is 100 cases; make the count explicit so a CI hiccup in the
    // server handshake is visible rather than silently passing on zero cases.
    int64_t seen = 0;
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto n = tc.draw(gs::integers<int64_t>({.min_value = 1,
                                                    .max_value = 1000}));
            REQUIRE(n > 0);
            seen++;
        },
        hegel::Settings{.test_cases = 200});
    REQUIRE(seen > 0);
}
