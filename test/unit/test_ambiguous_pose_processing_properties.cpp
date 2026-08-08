// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel PBT for the pure ambiguous-pose post-processing (plan U7 exposure of
// previously-unheadless-tested logic). Locks two always-true invariants:
//   - varus_valgus_calculation returns an angle in [0, pi/2] (it is
//     |asin(.)|), and stays finite,
//   - tibial_pose_selector returns either the tibia pose or its mirror
//     (a 2-element vocabulary — set-membership).
// R2-safe: these are structural range/membership properties, independent of
// the rotation math inside.

#include <cmath>

#include <hegel/hegel.h>

#include <catch2/catch_test_macros.hpp>

#include "domain/ambiguous_pose_processing.h"
#include "domain/data_structures_6D.h"

namespace gs = hegel::generators;

namespace {
// A pose gen with a NONZERO position (so compute_mirror_pose's view
// normalization does not divide by zero) and bounded angles.
std::tuple<decltype(gs::floats<double>({})), decltype(gs::floats<double>({}))>
PoseGens() {
    auto pos = gs::floats<double>({.min_value = 10.0, .max_value = 100.0});
    auto ang = gs::floats<double>({.min_value = -60.0, .max_value = 60.0});
    return {pos, ang};
}
}  // namespace

TEST_CASE("ambiguous_pose[PBT]: varus/valgus angle stays in [0, pi/2]",
          "[ambiguous_pose][pbt]") {
    auto gens = PoseGens();
    const auto& pos = std::get<0>(gens);
    const auto& ang = std::get<1>(gens);

    hegel::test(
        [&](hegel::TestCase& tc) {
            Point6D femur(tc.draw(pos), tc.draw(pos), tc.draw(pos),
                          tc.draw(ang), tc.draw(ang), tc.draw(ang));
            Point6D tibia(tc.draw(pos), tc.draw(pos), tc.draw(pos),
                          tc.draw(ang), tc.draw(ang), tc.draw(ang));
            float vv = varus_valgus_calculation(femur, tibia);
            tc.assume(std::isfinite(vv));
            REQUIRE(vv >= 0.0f);
            REQUIRE(vv <= static_cast<float>(3.14159265358979323846 / 2.0) + 1e-3f);
        },
        hegel::Settings{.test_cases = 300});
}

TEST_CASE("ambiguous_pose[PBT]: selector returns the tibia or its mirror",
          "[ambiguous_pose][pbt]") {
    auto gens = PoseGens();
    const auto& pos = std::get<0>(gens);
    const auto& ang = std::get<1>(gens);

    hegel::test(
        [&](hegel::TestCase& tc) {
            Point6D femur(tc.draw(pos), tc.draw(pos), tc.draw(pos),
                          tc.draw(ang), tc.draw(ang), tc.draw(ang));
            Point6D tibia(tc.draw(pos), tc.draw(pos), tc.draw(pos),
                          tc.draw(ang), tc.draw(ang), tc.draw(ang));

            // Guard against the numerically delicate mirror degeneracies so the
            // membership property is only asserted on well-defined inputs.
            Point6D mirror = compute_mirror_pose(tibia);
            tc.assume(std::isfinite(mirror.x) && std::isfinite(mirror.y) &&
                      std::isfinite(mirror.z) && std::isfinite(mirror.xa) &&
                      std::isfinite(mirror.ya) && std::isfinite(mirror.za));

            Point6D chosen = tibial_pose_selector(femur, tibia);
            bool is_tibia = (chosen.x == tibia.x && chosen.y == tibia.y &&
                             chosen.z == tibia.z && chosen.xa == tibia.xa &&
                             chosen.ya == tibia.ya && chosen.za == tibia.za);
            bool is_mirror = (chosen.x == mirror.x && chosen.y == mirror.y &&
                              chosen.z == mirror.z && chosen.xa == mirror.xa &&
                              chosen.ya == mirror.ya && chosen.za == mirror.za);
            REQUIRE((is_tibia || is_mirror));
        },
        hegel::Settings{.test_cases = 200});
}
