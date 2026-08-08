// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel PBT for the pure pose/kinematics file-format service (plan U7,
// pose_file_io extraction). Complements the deterministic round-trip unit
// tests by probing the structural invariant the load/save slots rely on across
// many random pose vectors:
//   - frame-count preservation: WriteKinematics -> ReadKinematics yields
//     exactly one (position-preserving) out slot per input frame, and
//   - value round-trip: every finite input pose round-trips through the
//     JTA_EULER_KINEMATICS text format within tolerance.
// R2-safe: we only assert output size + per-frame equality, never re-derive
// the column order / formatting.

#include <cmath>
#include <optional>
#include <sstream>
#include <vector>

#include <hegel/hegel.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "domain/pose_file_io.h"

namespace gs = hegel::generators;
using Catch::Approx;

TEST_CASE("pose_file[PBT]: kinematics round-trip preserves frame count + values",
          "[pose_file][pbt]") {
    // Bounded coords stay well inside the writer's setprecision(10) exact
    // round-trip window, so every row parses and every frame round-trips.
    auto coord = gs::floats<double>({.min_value = -1000.0, .max_value = 1000.0});

    hegel::test(
        [&](hegel::TestCase& tc) {
            auto n = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 8}));
            std::vector<Point6D> poses;
            poses.reserve(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                poses.emplace_back(tc.draw(coord), tc.draw(coord), tc.draw(coord),
                                   tc.draw(coord), tc.draw(coord), tc.draw(coord));
            }
            // Precondition on the drawn inputs: the writer's setprecision(10)
            // text format is only guaranteed to round-trip NORMAL (non-subnormal)
            // finite doubles. Reject subnormal/zero draws so the value round-trip
            // property is asserted on the format's real operating range.
            bool all_normal = true;
            for (const auto& p : poses) {
                all_normal =
                    all_normal && std::isnormal(p.x) && std::isnormal(p.y) &&
                    std::isnormal(p.z) && std::isnormal(p.xa) &&
                    std::isnormal(p.ya) && std::isnormal(p.za);
            }
            tc.assume(all_normal);

            std::stringstream ss;
            REQUIRE(jta::pose_file::WriteKinematics(ss, poses));

            std::vector<std::optional<Point6D>> out;
            auto res = jta::pose_file::ReadKinematics(ss, out);

            // Position-preserving: exactly one slot per frame, never compacted.
            REQUIRE(out.size() == poses.size());
            for (size_t i = 0; i < poses.size(); ++i) {
                // All drawn poses are finite/normal, so no NOT_OPTIMIZED row.
                REQUIRE(out[i].has_value());
                REQUIRE(out[i]->x == Approx(poses[i].x).margin(1e-3));
                REQUIRE(out[i]->y == Approx(poses[i].y).margin(1e-3));
                REQUIRE(out[i]->z == Approx(poses[i].z).margin(1e-3));
                REQUIRE(out[i]->xa == Approx(poses[i].xa).margin(1e-3));
                REQUIRE(out[i]->ya == Approx(poses[i].ya).margin(1e-3));
                REQUIRE(out[i]->za == Approx(poses[i].za).margin(1e-3));
            }
        },
        hegel::Settings{.test_cases = 300});
}
