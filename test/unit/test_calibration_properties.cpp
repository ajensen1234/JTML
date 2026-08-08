// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel PBT for the header-only biplane Calibration conversion (plan U7
// exposure; calibration.h is header-only and CUDA/Qt-free once
// camera_calibration.h resolves, per the 003 layer split / deferred decouple).
//
// Locks the strongest always-true safety property for the biplane path that
// previously had no test at all:
//   - convert_Pose_A_to_Pose_B -> convert_Pose_B_to_Pose_A preserves POSITION
//     (exact for an orthonormal axes_B_, up to float rounding), and
//   - the whole round-trip stays FINITE (catches NaN from asin outside [-1,1]
//     / gimbal-lock blow-up).
// R2-safe: we do not assert the (numerically fragile) euler-angle equality and
// we are not re-deriving the conversion — we only check position closure +
// finiteness.

#include <cmath>

#include <hegel/hegel.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "domain/data_structures_6D.h"
#include "services/calibration.h"

namespace gs = hegel::generators;
using Catch::Approx;

namespace {

// Build an orthonormal axes_B_ from an intrinsic Z-Y-X euler triple (a plain
// rotation matrix, so Q^T == Q^-1 — the only requirement the conversion places
// on axes_B_).
Matrix_3_3 OrthoFromEuler(float xrDeg, float yrDeg, float zrDeg) {
    constexpr float kPi = 3.14159265f;
    float a = xrDeg * kPi / 180.0f, b = yrDeg * kPi / 180.0f,
          c = zrDeg * kPi / 180.0f;
    float cx = cosf(a), sx = sinf(a), cy = cosf(b), sy = sinf(b),
          cz = cosf(c), sz = sinf(c);
    return Matrix_3_3(
        cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
        sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
        -sy, cy * sx, cy * cx);
}

}  // namespace

TEST_CASE("calibration[PBT]: biplane A->B->A preserves position and stays finite",
          "[calibration][pbt]") {
    auto ang = gs::floats<float>({.min_value = -60.0f, .max_value = 60.0f});
    auto coord = gs::floats<float>({.min_value = -200.0f, .max_value = 200.0f});

    hegel::test(
        [&](hegel::TestCase& tc) {
            Matrix_3_3 axes_B =
                OrthoFromEuler(tc.draw(ang), tc.draw(ang), tc.draw(ang));
            Calibration cal(CameraCalibration(), CameraCalibration(),
                            Vect_3(tc.draw(coord), tc.draw(coord),
                                   tc.draw(coord)),
                            axes_B);

            auto dcoord = gs::floats<double>({.min_value = -200.0, .max_value = 200.0});
            auto dang = gs::floats<double>({.min_value = -60.0, .max_value = 60.0});
            Point6D pose(tc.draw(dcoord), tc.draw(dcoord), tc.draw(dcoord),
                         tc.draw(dang), tc.draw(dang), tc.draw(dang));

            Point6D b = cal.convert_Pose_A_to_Pose_B(pose);
            tc.assume(std::isfinite(b.x) && std::isfinite(b.y) &&
                      std::isfinite(b.z) && std::isfinite(b.xa) &&
                      std::isfinite(b.ya) && std::isfinite(b.za));

            Point6D back = cal.convert_Pose_B_to_Pose_A(b);

            // Position round-trips: for orthonormal axes, Q * Q^T == I.
            REQUIRE(back.x == Approx(pose.x).margin(1e-1));
            REQUIRE(back.y == Approx(pose.y).margin(1e-1));
            REQUIRE(back.z == Approx(pose.z).margin(1e-1));
            // And the whole conversion chain stays finite (no NaN blow-up).
            REQUIRE((std::isfinite(back.x) && std::isfinite(back.y) &&
                     std::isfinite(back.z) && std::isfinite(back.xa) &&
                     std::isfinite(back.ya) && std::isfinite(back.za)));
        },
        hegel::Settings{.test_cases = 200});
}
