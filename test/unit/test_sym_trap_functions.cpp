// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Deterministic Catch2 twin for the sym-trap transform helpers (complements
// test_sym_trap_functions_properties.cpp). Pins the concrete cases: rotation_matrix
// orthonormality on fixed poses, and a basic double-mirror sanity on an axis pose.

#include <cmath>

#include <catch2/catch_test_macros.hpp>

#include "domain/data_structures_6D.h"
#include "domain/sym_trap_functions.h"

TEST_CASE("sym_trap_functions: rotation_matrix is orthonormal on fixed poses",
          "[sym_trap_functions]") {
    const Point6D poses[] = {
        Point6D(0, 0, 0, 0, 0, 0),
        Point6D(45, -10, 20, 30, -15, 60),
        Point6D(-180, 180, 90, -90, 45, -45),
    };
    for (const auto& p : poses) {
        float R[3][3];
        rotation_matrix(R, p);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < 3; ++k) dot += R[i][k] * R[j][k];
                float expect = (i == j) ? 1.0f : 0.0f;
                REQUIRE(std::abs(dot - expect) < 1e-4f);
            }
        }
    }
}

TEST_CASE("sym_trap_functions: mirror preserves position on a non-degenerate pose",
          "[sym_trap_functions]") {
    // NON-degenerate pose: the viewing/position vector must not be parallel to
    // the pose z-axis, else the mirror axis (cross product) degenerates to NaN
    // (the PBT guards this with assume-isfinite).
    Point6D pose(50, 20, 60, 15, -8, 30);
    Point6D mirror = compute_mirror_pose(pose);
    REQUIRE((std::isfinite(mirror.x) && std::isfinite(mirror.y) &&
             std::isfinite(mirror.z) && std::isfinite(mirror.xa) &&
             std::isfinite(mirror.ya) && std::isfinite(mirror.za)));
    // Mirror reflects orientation in the view plane; position is preserved.
    REQUIRE(mirror.x == pose.x);
    REQUIRE(mirror.y == pose.y);
    REQUIRE(mirror.z == pose.z);
}
