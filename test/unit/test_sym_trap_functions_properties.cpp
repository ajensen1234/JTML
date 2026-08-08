// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel PBT for the pure sym-trap transform helpers (plan U7 exposure of
// previously-unheadless-tested logic).
//
// Locks the strongest always-true invariant across many random euler triples:
//   - rotation_matrix(pose) builds an ORTHONORMAL matrix: R * R^T ~= I.
// This is an independent math property (a rotation matrix is orthogonal by
// construction), so it is R2-safe and, unlike the double-mirror probe, it
// never touches the numerically delicate compute_mirror_pose/view-normalization
// code paths or their debug prints.

#include <cmath>

#include <hegel/hegel.h>

#include <catch2/catch_test_macros.hpp>

#include "domain/data_structures_6D.h"
#include "domain/sym_trap_functions.h"

namespace gs = hegel::generators;

TEST_CASE("sym_trap_functions[PBT]: rotation_matrix builds an orthonormal R",
          "[sym_trap_functions][pbt]") {
    auto ang = gs::floats<float>({.min_value = -180.0f, .max_value = 180.0f});

    hegel::test(
        [&](hegel::TestCase& tc) {
            Point6D pose(0.0, 0.0, 0.0, tc.draw(ang), tc.draw(ang), tc.draw(ang));
            float R[3][3];
            rotation_matrix(R, pose);

            // R * R^T must be the identity up to float rounding.
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    float dot = 0.0f;
                    for (int k = 0; k < 3; ++k) dot += R[i][k] * R[j][k];
                    float expect = (i == j) ? 1.0f : 0.0f;
                    REQUIRE(std::abs(dot - expect) < 1e-4f);
                }
            }
        },
        hegel::Settings{.test_cases = 300});
}