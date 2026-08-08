// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Deterministic Catch2 twin for the ambiguous-pose post-processing (complements
// test_ambiguous_pose_processing_properties.cpp). Pins concrete cases: varus/valgus
// in [0, pi/2] on fixed poses, and the tibial selector returning the tibia or its
// mirror for the "worse VV wins" decision.

#include <cmath>

#include <catch2/catch_test_macros.hpp>

#include "domain/ambiguous_pose_processing.h"
#include "domain/data_structures_6D.h"

namespace {
bool PoseEqual(const Point6D& a, const Point6D& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.xa == b.xa &&
           a.ya == b.ya && a.za == b.za;
}
}  // namespace

TEST_CASE("ambiguous_pose: varus/valgus stays in [0, pi/2]", "[ambiguous_pose]") {
    Point6D femur(10, 0, 60, 0, 0, 0);
    Point6D tibia(10, 5, 55, 10, -5, 20);
    float vv = varus_valgus_calculation(femur, tibia);
    REQUIRE(std::isfinite(vv));
    REQUIRE(vv >= 0.0f);
    REQUIRE(vv <= static_cast<float>(3.14159265358979323846 / 2.0));
}

TEST_CASE("ambiguous_pose: selector returns tibia or its mirror",
          "[ambiguous_pose]") {
    Point6D femur(20, 0, 80, 5, 0, 0);
    Point6D tibia(20, 3, 74, 12, -8, 30);
    Point6D mirror = compute_mirror_pose(tibia);
    Point6D chosen = tibial_pose_selector(femur, tibia);
    REQUIRE((PoseEqual(chosen, tibia) || PoseEqual(chosen, mirror)));
    // The chosen pose must be the one with the smaller varus/valgus.
    float vv_orig = varus_valgus_calculation(femur, tibia);
    float vv_mirror = varus_valgus_calculation(femur, mirror);
    if (PoseEqual(chosen, mirror)) {
        REQUIRE(vv_mirror <= vv_orig + 1e-6f);
    } else {
        REQUIRE(vv_orig <= vv_mirror + 1e-6f);
    }
}
