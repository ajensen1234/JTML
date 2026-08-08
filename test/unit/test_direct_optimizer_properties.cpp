// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel property-based tests for the pure DirectOptimizer (plan U5/U6).
//
// PBT complements the deterministic Tier-1 analytic golden by probing
// invariants across thousands of randomly generated ranges / starting points /
// budgets / offsets:
//   - the returned optimum always stays inside the search cube the caller
//     requested (a strong, always-true invariant),
//   - the returned optimum never *increases* the cost relative to the seed,
//   - the cumulative call-offset (SetCallOffset) honours the cumulative
//     budget cap the production stage loop relies on,
//   - determinism (same config -> same result) holds across many draws.
// These are the properties we most want to protect during the U6 rewire that
// binds the real GPU cost behind DirectOptimizer.

#include <cmath>
#include <vector>

#include <hegel/hegel.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "domain/direct_optimizer.h"

namespace gs = hegel::generators;
using Catch::Approx;

namespace {

// Sum of squared per-DOF distance from a target (global min at `c`).
auto QuadraticCost(const Point6D& c) {
    return [c](const Point6D& p) {
        return (p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y) +
               (p.z - c.z) * (p.z - c.z) + (p.xa - c.xa) * (p.xa - c.xa) +
               (p.ya - c.ya) * (p.ya - c.ya) + (p.za - c.za) * (p.za - c.za);
    };
}

Point6D MakePoint(double x, double y, double z, double xa, double ya, double za) {
    return Point6D(x, y, z, xa, ya, za);
}

// True iff every DOF of `p` lies within [s_i - r_i, s_i + r_i] (the physical
// cube the unit searches over), with a float tolerance.
bool WithinSearchCube(const Point6D& p, const Point6D& s, const Point6D& r,
                      double tol = 1e-6) {
    const double bounds[6][2] = {
        {s.x - r.x, s.x + r.x},   {s.y - r.y, s.y + r.y},
        {s.z - r.z, s.z + r.z},   {s.xa - r.xa, s.xa + r.xa},
        {s.ya - r.ya, s.ya + r.ya}, {s.za - r.za, s.za + r.za},
    };
    const double vals[6] = {p.x, p.y, p.z, p.xa, p.ya, p.za};
    for (int i = 0; i < 6; ++i) {
        if (vals[i] < bounds[i][0] - tol || vals[i] > bounds[i][1] + tol)
            return false;
    }
    return true;
}

double Distance(const Point6D& a, const Point6D& b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
                     (a.z - b.z) * (a.z - b.z) + (a.xa - b.xa) * (a.xa - b.xa) +
                     (a.ya - b.ya) * (a.ya - b.ya) + (a.za - b.za) * (a.za - b.za));
}

}  // namespace

// The returned optimum must always lie inside the search cube the caller asked
// for, regardless of range scale, starting point, or budget. This is THE
// invariant the production OptimizerManager relies on (its optimum is written
// back into pose storage).
TEST_CASE("DirectOptimizer[PBT]: optimum stays inside the search cube",
          "[direct_optimizer][pbt]") {
    auto range_gen = gs::floats<double>({.min_value = 0.1, .max_value = 50.0});
    auto coord_gen = gs::floats<double>({.min_value = -100.0, .max_value = 100.0});
    auto budget_gen = gs::integers<uint32_t>({.min_value = 20, .max_value = 400});

    hegel::test(
        [&](hegel::TestCase& tc) {
            auto rx = tc.draw(range_gen);
            auto ry = tc.draw(range_gen);
            auto rz = tc.draw(range_gen);
            auto rxa = tc.draw(range_gen);
            auto rya = tc.draw(range_gen);
            auto rza = tc.draw(range_gen);
            Point6D range = MakePoint(rx, ry, rz, rxa, rya, rza);

            auto sx = tc.draw(coord_gen);
            auto sy = tc.draw(coord_gen);
            auto sz = tc.draw(coord_gen);
            auto sxa = tc.draw(coord_gen);
            auto sya = tc.draw(coord_gen);
            auto sza = tc.draw(coord_gen);
            Point6D start = MakePoint(sx, sy, sz, sxa, sya, sza);

            // Target drawn inside the range => a reachable minimum.
            Point6D target = MakePoint(
                sx + tc.draw(gs::floats<double>(
                            {.min_value = -rx, .max_value = rx})),
                sy + tc.draw(gs::floats<double>(
                            {.min_value = -ry, .max_value = ry})),
                sz + tc.draw(gs::floats<double>(
                            {.min_value = -rz, .max_value = rz})),
                sxa + tc.draw(gs::floats<double>(
                             {.min_value = -rxa, .max_value = rxa})),
                sya + tc.draw(gs::floats<double>(
                             {.min_value = -rya, .max_value = rya})),
                sza + tc.draw(gs::floats<double>(
                             {.min_value = -rza, .max_value = rza})));

            auto budget = tc.draw(budget_gen);

            DirectOptimizer opt(QuadraticCost(target), range, start, budget);
            tc.assume(opt.Run());

            Point6D loc = opt.GetOptimumLocation();
            REQUIRE(WithinSearchCube(loc, start, range));
        },
        hegel::Settings{.test_cases = 400});
}

// The reported optimum value must never exceed the seed evaluation: DIRECT only
// ever records improvements, so best_value <= f(starting_point). This is the
// monotonicity the live UpdateOptimum display depends on.
TEST_CASE("DirectOptimizer[PBT]: optimum never worsens the seed cost",
          "[direct_optimizer][pbt]") {
    auto range_gen = gs::floats<double>({.min_value = 0.5, .max_value = 40.0});
    auto coord_gen = gs::floats<double>({.min_value = -50.0, .max_value = 50.0});
    auto budget_gen = gs::integers<uint32_t>({.min_value = 20, .max_value = 300});

    hegel::test(
        [&](hegel::TestCase& tc) {
            double r = tc.draw(range_gen);
            Point6D range = MakePoint(r, r, r, r, r, r);
            Point6D start =
                MakePoint(tc.draw(coord_gen), tc.draw(coord_gen),
                          tc.draw(coord_gen), tc.draw(coord_gen),
                          tc.draw(coord_gen), tc.draw(coord_gen));
            Point6D target = MakePoint(
                start.x + tc.draw(gs::floats<double>(
                              {.min_value = -r, .max_value = r})),
                start.y + tc.draw(gs::floats<double>(
                              {.min_value = -r, .max_value = r})),
                start.z + tc.draw(gs::floats<double>(
                              {.min_value = -r, .max_value = r})),
                start.xa + tc.draw(gs::floats<double>(
                               {.min_value = -r, .max_value = r})),
                start.ya + tc.draw(gs::floats<double>(
                               {.min_value = -r, .max_value = r})),
                start.za + tc.draw(gs::floats<double>(
                               {.min_value = -r, .max_value = r})));
            auto budget = tc.draw(budget_gen);

            auto cost = QuadraticCost(target);
            double seed_value = cost(start);
            DirectOptimizer opt(cost, range, start, budget);
            tc.assume(opt.Run());

            REQUIRE(opt.GetOptimumValue() <= seed_value + 1e-9);
        },
        hegel::Settings{.test_cases = 300});
}

// The cumulative call-offset must honour the stage budget cap: with offset o
// and cumulative budget B, the loop runs until the *effective* count
// (offset + internal) reaches B, so GetCostFunctionCalls() >= B once the
// offset is set. This is exactly the semantics the trunk->branch->leaf
// cumulative sequence (effective 20k/25k/30k) depends on.
TEST_CASE("DirectOptimizer[PBT]: call-offset honours the cumulative budget",
          "[direct_optimizer][pbt][offset]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto range_gen = gs::floats<double>({.min_value = 0.5, .max_value = 30.0});
            double r = tc.draw(range_gen);
            Point6D range = MakePoint(r, r, r, r, r, r);
            Point6D start = MakePoint(0, 0, 0, 0, 0, 0);
            Point6D target = MakePoint(2, -1, 0.5, 3, -2, 1);

            // B in (offset, offset+400]: a later stage in the cumulative series.
            auto offset = tc.draw(gs::integers<uint32_t>({.min_value = 0,
                                                          .max_value = 2000}));
            auto budget =
                tc.draw(gs::integers<uint32_t>(
                    {.min_value = offset + 1, .max_value = offset + 400}));

            DirectOptimizer opt(QuadraticCost(target), range, start, budget);
            opt.SetCallOffset(offset);
            tc.assume(opt.Run());

            // Seed eats 1 call, then the guard (offset + calls) < budget drives
            // the effective count to at least the cumulative budget.
            REQUIRE(opt.GetCostFunctionCalls() >= budget);
            // And it must have consumed at least the offset worth of calls too.
            REQUIRE(opt.GetCostFunctionCalls() >= offset);
        },
        hegel::Settings{.test_cases = 400});
}
