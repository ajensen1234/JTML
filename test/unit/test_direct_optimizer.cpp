// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Tier-1 analytic golden for the extracted generic DIRECT optimizer (plan U5).
// These use analytic cost functions as independent ground truth (R2) -- no
// Qt/VTK/CUDA, fully deterministic and CI-runnable.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cfloat>

#include "core/direct_optimizer.h"

using Catch::Approx;

namespace {

// f(p) = sum((p_i - c_i)^2) on the physical space; returns the cost lambda.
auto QuadraticCost(const Point6D& c) {
    return [c](const Point6D& p) {
        return (p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y) +
               (p.z - c.z) * (p.z - c.z) + (p.xa - c.xa) * (p.xa - c.xa) +
               (p.ya - c.ya) * (p.ya - c.ya) + (p.za - c.za) * (p.za - c.za);
    };
}

Point6D Origin() {
    return Point6D(0, 0, 0, 0, 0, 0);
}

Point6D UnitSideRange(double r) {
    return Point6D(r, r, r, r, r, r);
}

double MaxAbs(const Point6D& p) {
    return std::max(
        {std::fabs(p.x), std::fabs(p.y), std::fabs(p.z), std::fabs(p.xa),
         std::fabs(p.ya), std::fabs(p.za)});
}

}  // namespace

TEST_CASE("DirectOptimizer preserves a minimum located at the search center",
          "[direct_optimizer][convergence]") {
    // Minimum at origin, which is exactly the starting point; DIRECT must not
    // lose it. Range +-10 per DOF maps the unit cube to [-10,10]^6.
    DirectOptimizer opt(QuadraticCost(Origin()), UnitSideRange(10.0), Origin(),
                        20000);
    REQUIRE(opt.Run());
    REQUIRE(opt.GetOptimumValue() == Approx(0.0).margin(1e-6));
    REQUIRE(MaxAbs(opt.GetOptimumLocation()) == Approx(0.0).margin(1e-6));
}

TEST_CASE("DirectOptimizer locates a shifted minimum", "[direct_optimizer][convergence]") {
    // Minimum at (3,3,3,3,3,3), starting at origin: DIRECT must move toward it.
    Point6D target(3, 3, 3, 3, 3, 3);
    DirectOptimizer opt(QuadraticCost(target), UnitSideRange(10.0), Origin(),
                        20000);
    REQUIRE(opt.Run());

    Point6D loc = opt.GetOptimumLocation();
    // The DIRECT optimum should land near the known analytic minimum.
    double err = std::sqrt((loc.x - target.x) * (loc.x - target.x) +
                           (loc.y - target.y) * (loc.y - target.y) +
                           (loc.z - target.z) * (loc.z - target.z) +
                           (loc.xa - target.xa) * (loc.xa - target.xa) +
                           (loc.ya - target.ya) * (loc.ya - target.ya) +
                           (loc.za - target.za) * (loc.za - target.za));
    CAPTURE(loc.x, loc.y, loc.z, loc.xa, loc.ya, loc.za, err,
            opt.GetOptimumValue());
    // f_min must be non-increasing from the starting value (900 at origin).
    REQUIRE(opt.GetOptimumValue() <= 900.0);
    // ...and it should substantially improve: well inside a 6D ball, not stuck
    // at the origin. err < 4.0 of the analytic minimum.
    REQUIRE(err < 4.0);
}

TEST_CASE("DirectOptimizer enforces the budget guard",
          "[direct_optimizer][budget]") {
    DirectOptimizer opt(QuadraticCost(Origin()), UnitSideRange(10.0), Origin(),
                        1000);
    REQUIRE(opt.Run());
    // The check is while(calls < budget); calls grow by 2 per trisected box, so
    // the loop terminates with calls >= budget (never infinite), and the seed
    // evaluation counts as one call.
    REQUIRE(opt.GetCostFunctionCalls() >= 1000u);
}

TEST_CASE("DirectOptimizer seed evaluation counts toward the budget",
          "[direct_optimizer][budget]") {
    DirectOptimizer opt(QuadraticCost(Origin()), UnitSideRange(10.0), Origin(),
                        0);
    REQUIRE(opt.Run());
    // budget == 0: no loop iterations, but the seed center evaluation still ran
    // (mirrors the original cost_function_calls_++ on the seed).
    REQUIRE(opt.GetCostFunctionCalls() == 1u);
}

TEST_CASE("DirectOptimizer is deterministic across runs",
          "[direct_optimizer][determinism]") {
    Point6D target(2, -1, 0.5, 3, -2, 1);
    auto run = [&]() {
        DirectOptimizer opt(QuadraticCost(target), UnitSideRange(10.0),
                            Origin(), 5000);
        opt.Run();
        return opt.GetOptimumLocation();
    };
    Point6D a = run();
    Point6D b = run();
    REQUIRE(a.x == Approx(b.x));
    REQUIRE(a.ya == Approx(b.ya));
    REQUIRE(a.GetDistanceFrom(b) == Approx(0.0).margin(1e-12));
}

TEST_CASE("DirectOptimizer rejects an all-zero range without looping",
          "[direct_optimizer][edge]") {
    DirectOptimizer opt(QuadraticCost(Origin()), Point6D(0, 0, 0, 0, 0, 0),
                        Origin(), 1);
    REQUIRE_FALSE(opt.Run());  // valid_range_ is false -> immediate error
    REQUIRE(opt.GetCostFunctionCalls() == 0u);
}

TEST_CASE("DirectOptimizer honors an early stop request",
          "[direct_optimizer][edge]") {
    DirectOptimizer opt(QuadraticCost(Origin()), UnitSideRange(10.0), Origin(),
                        100000);
    opt.Stop();
    // Stop() is checked before each iteration, so with budget > 0 the loop is
    // skipped and only the seed evaluation runs.
    REQUIRE(opt.Run());
    REQUIRE(opt.GetCostFunctionCalls() == 1u);
}

TEST_CASE("DirectOptimizer call-offset shifts the cumulative budget guard",
          "[direct_optimizer][budget][offset]") {
    // Budget 2000, offset 1000: the loop must run only until the *effective*
    // count (offset + internal) reaches the budget, mirroring a mid-stream
    // stage in the cumulative trunk/branch/leaf sequence.
    DirectOptimizer opt(QuadraticCost(Origin()), UnitSideRange(10.0), Origin(),
                        2000);
    opt.SetCallOffset(1000);
    REQUIRE(opt.Run());
    // Effective calls >= budget (loop exits after crossing the guard)...
    REQUIRE(opt.GetCostFunctionCalls() >= 2000u);
    // ...but internal calls only spanned the remaining 1000.
    // GetCostFunctionCalls() == offset + internal, so the internal count is
    // bounded by (budget - offset), i.e. the effective 10k/20k/30k span.
    REQUIRE(opt.GetCostFunctionCalls() - 1000u < 2000u);
}

TEST_CASE("DirectOptimizer fires the iteration callback per iteration",
          "[direct_optimizer][callback]") {
    DirectOptimizer opt(QuadraticCost(Origin()), UnitSideRange(10.0), Origin(),
                        20000);
    int iterations = 0;
    opt.SetIterationCallback([&]() { iterations++; });
    REQUIRE(opt.Run());
    // At least one ConvexHull+Trisect iteration ran under a growing budget.
    REQUIRE(iterations > 0);
}

TEST_CASE("DirectOptimizer fires the improvement callback on improvements",
          "[direct_optimizer][callback]") {
    // Minimum at origin, seed at origin: exactly the start point, so no
    // improvement should fire.
    DirectOptimizer no_improve(QuadraticCost(Origin()), UnitSideRange(10.0),
                               Origin(), 20000);
    int improvements_no = 0;
    no_improve.SetImprovementCallback(
        [&](const Point6D&, double) { improvements_no++; });
    REQUIRE(no_improve.Run());
    REQUIRE(improvements_no == 0);

    // Minimum shifted to (3,3,3,3,3,3), seed at origin: DIRECT must improve and
    // each improvement must report a better (lower) value.
    Point6D target(3, 3, 3, 3, 3, 3);
    DirectOptimizer improve(QuadraticCost(target), UnitSideRange(10.0),
                            Origin(), 20000);
    int improvements = 0;
    double last_value = DBL_MAX;
    improve.SetImprovementCallback([&](const Point6D& loc, double val) {
        improvements++;
        REQUIRE(val <= last_value);  // non-increasing over the run
        REQUIRE(improve.GetOptimumLocation().GetDistanceFrom(loc) ==
                Approx(0.0).margin(1e-12));
        last_value = val;
    });
    REQUIRE(improve.Run());
    REQUIRE(improvements > 0);
}
