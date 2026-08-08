// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Deterministic Catch2 twin for the biplane Calibration conversion
// (complements test_calibration_properties.cpp). Pins the concrete A->B->A
// position round-trip on a fixed orthonormal axes_B_.

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "domain/data_structures_6D.h"
#include "services/calibration.h"

using Catch::Approx;

namespace {
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

TEST_CASE("calibration: biplane A->B->A preserves position",
          "[calibration]") {
    Matrix_3_3 Q = OrthoFromEuler(25.0f, -10.0f, 40.0f);
    // Named temps avoid the most-vexing-parse of Calibration(CameraCalibration(), ...).
    CameraCalibration cam_a;
    CameraCalibration cam_b;
    Calibration cal(cam_a, cam_b, Vect_3(-50.0f, 30.0f, 200.0f), Q);

    Point6D pose(12.0, -8.0, 90.0, 5.0, -10.0, 15.0);
    Point6D b = cal.convert_Pose_A_to_Pose_B(pose);
    Point6D back = cal.convert_Pose_B_to_Pose_A(b);

    // Position round-trips for an orthonormal axes_B_ (Q * Q^T == I).
    REQUIRE(back.x == Approx(pose.x).margin(1e-1));
    REQUIRE(back.y == Approx(pose.y).margin(1e-1));
    REQUIRE(back.z == Approx(pose.z).margin(1e-1));
    // The chain stays finite.
    REQUIRE(std::isfinite(back.xa));
    REQUIRE(std::isfinite(back.ya));
    REQUIRE(std::isfinite(back.za));
}

TEST_CASE("calibration: monoplane conversion is the identity",
          "[calibration]") {
    CameraCalibration cam;
    Calibration cal(cam);
    Point6D pose(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    Point6D b = cal.convert_Pose_A_to_Pose_B(pose);
    REQUIRE(b.x == pose.x);
    REQUIRE(b.y == pose.y);
    REQUIRE(b.z == pose.z);
    REQUIRE(b.xa == pose.xa);
}
