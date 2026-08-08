// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Deterministic Catch2 twin for the LocationStorage pose-matrix service
// (complements test_location_storage_properties.cpp). Pins the concrete
// rectangle + round-trip edges on a fixed load interleaving.

#include <catch2/catch_test_macros.hpp>

#include "domain/data_structures_6D.h"
#include "services/location_storage.h"

TEST_CASE("location_storage: rectangle shape across interleaved loads",
          "[location_storage]") {
    LocationStorage ls;
    // 2 models, 1 frame, 1 model, 2 frames (arbitrary interleaving).
    ls.LoadNewModel(1000.0, 0.1);
    ls.LoadNewModel(1000.0, 0.1);
    ls.LoadNewFrame();      // frame 0: 2 models
    ls.LoadNewFrame();      // frame 1: 2 models
    REQUIRE(ls.GetFrameCount() == 2);
    REQUIRE(ls.GetModelCount() == 2);
    ls.LoadNewFrame();      // frame 2: 2 models
    REQUIRE(ls.GetFrameCount() == 3);
    REQUIRE(ls.GetModelCount() == 2);

    // Out-of-range reads return the zero pose without crashing.
    Point6D zero = ls.GetPose(99, 99);
    REQUIRE((zero.x == 0.0 && zero.y == 0.0 && zero.z == 0.0));
}

TEST_CASE("location_storage: SavePose/GetPose round-trips exactly",
          "[location_storage]") {
    LocationStorage ls;
    ls.LoadNewModel(1000.0, 0.1);
    ls.LoadNewModel(1000.0, 0.1);
    ls.LoadNewFrame();
    ls.LoadNewFrame();

    Point6D pose(1.5, -2.25, 30.0, 10.0, -5.0, 90.0);
    ls.SavePose(1, 0, pose);
    Point6D got = ls.GetPose(1, 0);
    REQUIRE(got.x == pose.x);
    REQUIRE(got.y == pose.y);
    REQUIRE(got.z == pose.z);
    REQUIRE(got.xa == pose.xa);
    REQUIRE(got.ya == pose.ya);
    REQUIRE(got.za == pose.za);

    // A different cell is untouched: still the LoadNewModel default pose
    // (0,0,-0.25*principal_distance/pixel_pitch) = -0.25*1000/0.1 = -2500.
    Point6D other = ls.GetPose(1, 1);
    REQUIRE(other.x == 0.0);
    REQUIRE(other.y == 0.0);
    REQUIRE(other.z == -2500.0);
    REQUIRE(other.xa == 0.0);
    REQUIRE(other.ya == 0.0);
    REQUIRE(other.za == 0.0);
}
