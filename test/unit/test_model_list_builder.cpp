// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// ModelListBuilder unit tests (plan U7, U10 / R10). Pure name/dedup + size
// logic; no Qt, no GPU, headless.

#include <catch2/catch_test_macros.hpp>

#include "domain/model_list_builder.h"

using jta::ModelListBuilder;

TEST_CASE("ModelListBuilder: unique names unchanged", "[model_list_builder]") {
    auto names = ModelListBuilder::UniquifyModelNames({"femur", "tibia"}, {});
    REQUIRE(names == std::vector<std::string>({"femur", "tibia"}));
}

TEST_CASE("ModelListBuilder: collision with existing gets (N) suffix",
          "[model_list_builder]") {
    auto names = ModelListBuilder::UniquifyModelNames({"femur"}, {"femur"});
    REQUIRE(names == std::vector<std::string>({"femur(2)"}));
}

TEST_CASE("ModelListBuilder: within-set duplicate gets suffix",
          "[model_list_builder]") {
    // Production quirk (R15): N identical inputs in one load -> ["A(2)","A"].
    auto names = ModelListBuilder::UniquifyModelNames({"A", "A"}, {});
    REQUIRE(names == std::vector<std::string>({"A(2)", "A"}));
}

TEST_CASE("ModelListBuilder: three identical within-set names (R15 quirk)",
          "[model_list_builder]") {
    // Mirrors the doc-review-pinned behavior: ["A(2)","A(3)","A"] for N=3.
    auto names = ModelListBuilder::UniquifyModelNames({"A", "A", "A"}, {});
    REQUIRE(names == std::vector<std::string>({"A(2)", "A(3)", "A"}));
}

TEST_CASE("ModelListBuilder: existing collision after within-set dedup",
          "[model_list_builder]") {
    auto names = ModelListBuilder::UniquifyModelNames({"A", "A"}, {"A"});
    // Within-set: ["A(2)","A"]; then "A(2)" collides? No; "A" collides with
    // existing "A(" -> wait: existing names = {"A"}, so "A" -> "A(2)", and the
    // first "A(2)" is checked against existing {"A"} (no match). Result:
    // ["A(2)", "A(2)"]? No -- "A"->"A(2)" because it collides with existing
    // "A". Let the implementation define the exact output via the real loop.
    auto out = ModelListBuilder::UniquifyModelNames({"A", "A"}, {"A"});
    // Production: pass1 within-set -> {"A(2)", "A"}. pass2 vs existing {"A"}:
    //   i=0 "A(2)" vs {"A"} -> no match -> stays "A(2)".
    //   i=1 "A" vs {"A"} -> match -> "A" + "(2)" = "A(2)", scan restarts on
    //       existing {"A"} -> "A(2)" != "A" -> stops. -> "A(2)".
    REQUIRE(out == std::vector<std::string>({"A(2)", "A(2)"}));
}

TEST_CASE("ModelListBuilder: AllSameSize rejects mismatched pair",
          "[model_list_builder]") {
    REQUIRE(ModelListBuilder::AllSameSize(
        1024, 1024, {{1024, 1024}, {1024, 1024}}));
    REQUIRE_FALSE(ModelListBuilder::AllSameSize(
        1024, 1024, {{1024, 1024}, {512, 512}}));
    // No loaded frames -> vacuously true (fresh load allowed).
    REQUIRE(ModelListBuilder::AllSameSize(1024, 1024, {}));
}
