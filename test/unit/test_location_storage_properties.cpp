// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel PBT for the LocationStorage pose-matrix service (plan U7 exposure of a
// previously-unheadless-tested service; now compilable headlessly from
// src/services per the 003 U2 layer split).
//
// Locks two invariants across arbitrary interleavings of LoadNewModel /
// LoadNewFrame:
//   - the matrix stays RECTANGULAR (every frame holds the same number of
//     models), so GetFrameCount()/GetModelCount() agree with the operation
//     counts, and
//   - in-bounds SavePose -> GetPose round-trips exactly.
// R2-safe: purely count/shape/round-trip, no re-derivation of the default pose.

#include <cmath>

#include <hegel/hegel.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "domain/data_structures_6D.h"
#include "services/location_storage.h"

namespace gs = hegel::generators;
using Catch::Approx;

TEST_CASE("location_storage[PBT]: matrix stays rectangular + Save/Get round-trip",
          "[location_storage][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            LocationStorage ls;
            int models = 0, frames = 0;
            auto ops = tc.draw(gs::vectors(gs::sampled_from<int>({0, 1}),
                                            {.min_size = 0, .max_size = 14}));
            for (int op : ops) {
                if (op == 0) {
                    ls.LoadNewModel(1000.0, 0.1);
                    ++models;
                } else {
                    ls.LoadNewFrame();
                    ++frames;
                }
            }

            REQUIRE(ls.GetFrameCount() == frames);
            // Rectangle invariant: when any frame exists, model count == number
            // of models loaded across every frame. (When frames == 0 the count
            // getter reports 0 even with models loaded, so only assert when a
            // frame exists.)
            if (frames > 0) REQUIRE(ls.GetModelCount() == models);

            if (frames > 0 && models > 0) {
                auto coord =
                    gs::floats<double>({.min_value = -50.0, .max_value = 50.0});
                Point6D pose(tc.draw(coord), tc.draw(coord), tc.draw(coord),
                             tc.draw(coord), tc.draw(coord), tc.draw(coord));
                int f = tc.draw(gs::integers<int>({.min_value = 0,
                                                   .max_value = frames - 1}));
                int m = tc.draw(gs::integers<int>({.min_value = 0,
                                                   .max_value = models - 1}));
                ls.SavePose(f, m, pose);
                Point6D got = ls.GetPose(f, m);
                REQUIRE(got.x == Approx(pose.x).margin(1e-9));
                REQUIRE(got.y == Approx(pose.y).margin(1e-9));
                REQUIRE(got.z == Approx(pose.z).margin(1e-9));
                REQUIRE(got.xa == Approx(pose.xa).margin(1e-9));
                REQUIRE(got.ya == Approx(pose.ya).margin(1e-9));
                REQUIRE(got.za == Approx(pose.za).margin(1e-9));
            }
        },
        hegel::Settings{.test_cases = 300});
}
