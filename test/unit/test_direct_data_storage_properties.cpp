// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel PBT for the DIRECT container (DirectDataStorage, plan U7 exposure of
// previously-unheadless-tested logic).
//
// Locks the cache-vs-container consistency invariant that the DIRECT loop's
// performance depends on, across many random box insertions:
//   - GetMinimumHyperboxValue(col) always equals the column's tail (back())
//     box value — the cached min never drifts from the actual minimum, and
//   - GetSizeStoredInColumn(col) equals that box's size (column homogeneity).
// R2-safe: asserts public-API consistency, never re-derives the ordering.

#include <hegel/hegel.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "domain/data_structures_6D.h"
#include "domain/direct_data_storage.h"

namespace gs = hegel::generators;
using Catch::Approx;

TEST_CASE("direct_data_storage[PBT]: min-cache consistent with column tail",
          "[direct_data_storage][pbt]") {
    auto val_gen = gs::floats<double>({.min_value = -10.0, .max_value = 10.0});
    // Small size vocabulary so columns collide often (and new sizes appear).
    auto size_gen = gs::sampled_from<double>({0.5, 1.0, 2.0});

    hegel::test(
        [&](hegel::TestCase& tc) {
            DirectDataStorage ds;
            auto adds = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 20}));

            for (int i = 0; i < adds; ++i) {
                double v = tc.draw(val_gen);
                double s = tc.draw(size_gen);
                // Owner of the pointer: AddHyperBox stores it; DeleteAllStoredHyperboxes
                // frees it below.
                auto* box = new HyperBox6D(
                    v, Point6D(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                    Point6D(s, s, s, s, s, s));
                ds.AddHyperBox(box);

                int cols = static_cast<int>(ds.GetNumberColumns());
                for (int c = 0; c < cols; ++c) {
                    HyperBox6D minbox = ds.GetMinimumHyperbox(c);
                    // Cached min == tail box value; cached size == that box size.
                    REQUIRE(ds.GetMinimumHyperboxValue(c) ==
                            Approx(minbox.value_).margin(1e-12));
                    REQUIRE(ds.GetSizeStoredInColumn(c) ==
                            Approx(minbox.size_).margin(1e-12));
                }
            }

            ds.DeleteAllStoredHyperboxes();
        },
        hegel::Settings{.test_cases = 300});
}
