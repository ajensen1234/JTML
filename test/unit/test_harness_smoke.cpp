// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Test-harness smoke test: proves the Catch2 toolchain + CTest registration
// work end-to-end. It intentionally exercises no JTML code.

#include <catch2/catch_test_macros.hpp>

TEST_CASE("JTML test harness smoke", "[harness]") {
    SECTION("arithmetic") {
        REQUIRE(1 + 1 == 2);
    }
    SECTION("catch2 macros available") {
        CHECK(true);
        CHECK_FALSE(false);
    }
}
