// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Tier-1 analytic golden for the extracted generic DIRECT optimizer (plan U5).
// These use analytic cost functions as independent ground truth (R2) -- no
// Qt/VTK/CUDA, fully deterministic and CI-runnable.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cfloat>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "domain/direct_optimizer.h"

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

bool AllFinite(const Point6D& p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
           std::isfinite(p.xa) && std::isfinite(p.ya) && std::isfinite(p.za);
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

// ---------------------------------------------------------------------------
// U3 (plan 008): CUDA-free finite-check at DirectOptimizer::EvaluateCostFunction
// -- the one shared eval chokepoint (production/oracle/coordinator). The probe
// drives a fake cost lambda returning NaN / +Inf / -Inf / sNaN and asserts
// (a) no crash, (b) the counter increments per non-finite eval, (c) the
// optimum never reflects a non-finite eval, (d) a NaN sequence is followed by
// a normally-resuming search, (e) the iteration callback fires once per
// non-finite eval with ordering preserved. "Surviving storage columns stay
// finite" is a construction invariant: a non-finite eval is never stored (the
// trisection caller skips AddHyperBox), so no NaN can enter the storage the
// hull reads; the observable proxies are the finite optimum, the consumed
// budget, and the resumed search.
// ---------------------------------------------------------------------------

TEST_CASE("DirectOptimizer keeps the non-finite counter at zero on the finite path",
          "[direct_optimizer][finite-check]") {
    // Happy path (plan U3): finite evals behave identically -- the counter
    // stays 0 and the optimum updates normally (covered by the existing
    // convergence/golden cases above).
    DirectOptimizer opt(QuadraticCost(Origin()), UnitSideRange(10.0), Origin(),
                        5000);
    REQUIRE(opt.Run());
    REQUIRE(opt.GetNonFiniteCount() == 0u);
}

TEST_CASE("DirectOptimizer counts NaN/Inf/sNaN evals and keeps the optimum finite",
          "[direct_optimizer][finite-check]") {
    // Edge case (plan U3): a cost cycling quiet-NaN / +Inf / -Inf / sNaN.
    // Every eval is non-finite: no crash, the counter equals the call count,
    // and the optimum never reflects a non-finite result (it stays at the
    // finite no-finite-optimum-yet sentinel, never NaN/Inf).
    int phase = 0;
    DirectOptimizer opt([&](const Point6D&) -> double {
        switch (phase++ % 4) {
            case 0:
                return std::numeric_limits<double>::quiet_NaN();
            case 1:
                return std::numeric_limits<double>::infinity();
            case 2:
                return -std::numeric_limits<double>::infinity();
            default:
                return std::numeric_limits<double>::signaling_NaN();
        }
    },
    UnitSideRange(10.0), Origin(), 2000);
    REQUIRE(opt.Run());  // (a) no crash: the run completes
    // The budget was consumed, i.e. the search kept iterating over a finite
    // storage instead of stalling on poisoned columns.
    REQUIRE(opt.GetCostFunctionCalls() >= 2000u);
    // (b) counter increments per non-finite eval.
    REQUIRE(opt.GetNonFiniteCount() == opt.GetCostFunctionCalls());
    // (c) the optimum never reflects a non-finite eval.
    REQUIRE(std::isfinite(opt.GetOptimumValue()));
    REQUIRE(AllFinite(opt.GetOptimumLocation()));
}

TEST_CASE("DirectOptimizer resumes the search after a run of non-finite evals",
          "[direct_optimizer][finite-check]") {
    // Edge case (plan U3): the first 5 evals (seed + first two trisections)
    // return NaN, then the quadratic takes over. The search must resume
    // normally: the run completes, the counter records exactly the 5
    // non-finite evals, and the optimum still converges to the analytic
    // minimum (a stored NaN would have poisoned the column minimum and the
    // hull's slope comparisons).
    Point6D target(3, 3, 3, 3, 3, 3);
    const unsigned int kNaNPrefix = 5;
    unsigned int evals = 0;
    DirectOptimizer opt([&](const Point6D& p) -> double {
        if (evals++ < kNaNPrefix) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return QuadraticCost(target)(p);
    },
    UnitSideRange(10.0), Origin(), 20000);
    REQUIRE(opt.Run());
    REQUIRE(opt.GetNonFiniteCount() == kNaNPrefix);
    REQUIRE(opt.GetCostFunctionCalls() >= 20000u);
    REQUIRE(std::isfinite(opt.GetOptimumValue()));
    Point6D loc = opt.GetOptimumLocation();
    REQUIRE(AllFinite(loc));
    double err = std::sqrt((loc.x - target.x) * (loc.x - target.x) +
                           (loc.y - target.y) * (loc.y - target.y) +
                           (loc.z - target.z) * (loc.z - target.z) +
                           (loc.xa - target.xa) * (loc.xa - target.xa) +
                           (loc.ya - target.ya) * (loc.ya - target.ya) +
                           (loc.za - target.za) * (loc.za - target.za));
    CAPTURE(loc.x, loc.y, loc.z, loc.xa, loc.ya, loc.za, err,
            opt.GetOptimumValue());
    REQUIRE(err < 4.0);
}

TEST_CASE("DirectOptimizer fires the iteration callback per non-finite eval",
          "[direct_optimizer][finite-check][callback]") {
    // Integration (plan U3): the iteration-callback event fires once per
    // non-finite eval with ordering preserved. Events: 'e' pushed by the cost
    // lambda, 'c' by the callback. Each non-finite eval must fire the callback
    // immediately (every 'e' is followed by 'c' before any other event), so
    // the fire count equals the non-finite eval count in eval order.
    std::vector<char> events;
    DirectOptimizer opt([&](const Point6D&) -> double {
        events.push_back('e');
        return std::numeric_limits<double>::quiet_NaN();
    },
    UnitSideRange(10.0), Origin(), 1000);
    opt.SetIterationCallback([&]() { events.push_back('c'); });
    REQUIRE(opt.Run());
    REQUIRE(opt.GetNonFiniteCount() == opt.GetCostFunctionCalls());
    REQUIRE(events.size() >= 2);
    for (std::size_t i = 0; i < events.size(); i++) {
        if (events[i] == 'e') {
            REQUIRE(i + 1 < events.size());
            REQUIRE(events[i + 1] == 'c');
        }
    }
}

// ---------------------------------------------------------------------------
// U8 (plan 008): DirectOptimizer::Options with bit-identical defaults (Cut C,
// origin R3). The slot's STRUCTURE and the default-path bit-identity are what
// this unit proves; non-default fields are FAIL-FAST STUBS (review-resolved
// scope boundary) -- the divergence branches land with the algorithm plan.
//
// The identity trace below records the full convergence + budget-accounting
// surface (optimum value/location, call count, iteration count, improvement
// sequence) for the 4-arg pre-Options form and the explicit-default Options
// form on the same deterministic config and requires EXACT equality -- the
// default path must not touch a single division or branch differently.
// ---------------------------------------------------------------------------

TEST_CASE("DirectOptimizer default Options reproduce the pre-Options search bit-identically",
          "[direct_optimizer][options]") {
    Point6D target(3, 3, 3, 3, 3, 3);
    const unsigned int kBudget = 5000;

    auto run_with = [&](const DirectOptimizer::Options& opts) {
        std::vector<double> improvement_values;
        std::vector<Point6D> improvement_locations;
        unsigned int iterations = 0;
        DirectOptimizer opt(QuadraticCost(target), UnitSideRange(10.0),
                            Origin(), kBudget, opts);
        opt.SetIterationCallback([&]() { iterations++; });
        opt.SetImprovementCallback(
            [&](const Point6D& loc, double v) {
                improvement_values.push_back(v);
                improvement_locations.push_back(loc);
            });
        REQUIRE(opt.Run());
        return std::make_tuple(
            opt.GetOptimumValue(), opt.GetOptimumLocation(),
            opt.GetCostFunctionCalls(), opt.GetNonFiniteCount(), iterations,
            improvement_values, improvement_locations);
    };
    auto run_pre_options = [&]() {
        // The 4-arg pre-Options form must keep compiling (5th arg defaulted).
        std::vector<double> improvement_values;
        std::vector<Point6D> improvement_locations;
        unsigned int iterations = 0;
        DirectOptimizer opt(QuadraticCost(target), UnitSideRange(10.0),
                            Origin(), kBudget);
        opt.SetIterationCallback([&]() { iterations++; });
        opt.SetImprovementCallback(
            [&](const Point6D& loc, double v) {
                improvement_values.push_back(v);
                improvement_locations.push_back(loc);
            });
        REQUIRE(opt.Run());
        return std::make_tuple(
            opt.GetOptimumValue(), opt.GetOptimumLocation(),
            opt.GetCostFunctionCalls(), opt.GetNonFiniteCount(), iterations,
            improvement_values, improvement_locations);
    };

    auto with_opts = run_with(DirectOptimizer::Options{});
    auto without_opts = run_pre_options();

    // Exact (bit-for-bit) equality: the deterministic DIRECT loop over the
    // pure quadratic cost is identical code on the default path, so any
    // difference is a guarded-divergence failure.
    REQUIRE(std::get<0>(with_opts) == std::get<0>(without_opts));
    REQUIRE(std::get<1>(with_opts).GetDistanceFrom(std::get<1>(without_opts)) ==
            Approx(0.0).margin(1e-12));
    REQUIRE(std::get<2>(with_opts) == std::get<2>(without_opts));
    REQUIRE(std::get<3>(with_opts) == std::get<3>(without_opts));
    REQUIRE(std::get<4>(with_opts) == std::get<4>(without_opts));
    REQUIRE(std::get<5>(with_opts) == std::get<5>(without_opts));
    REQUIRE(std::get<6>(with_opts).size() ==
            std::get<6>(without_opts).size());
    for (std::size_t i = 0; i < std::get<6>(with_opts).size(); ++i) {
        REQUIRE(std::get<6>(with_opts)[i].GetDistanceFrom(
                    std::get<6>(without_opts)[i]) ==
                Approx(0.0).margin(1e-12));
    }
    // The recorded trace must be non-trivial (the budget was actually
    // consumed), so the equality above is meaningful.
    REQUIRE(std::get<4>(with_opts) > 0u);
    REQUIRE(std::get<5>(with_opts).size() > 0u);
}

TEST_CASE("DirectOptimizer non-default Options fields fail fast (plan-008 stub semantics)",
          "[direct_optimizer][options]") {
    using Options = DirectOptimizer::Options;

    auto make = [](Options opts) {
        return DirectOptimizer(QuadraticCost(Origin()), UnitSideRange(10.0),
                               Origin(), 100, opts);
    };

    // Default Options construct fine (the happy path).
    REQUIRE_NOTHROW(make(Options{}));

    // Edge case (plan 008): each non-default field hits the fail-fast guard
    // with a clear error. Stub semantics -- no variant behavior ships here.
    Options o;
    o.epsilon = 1e-9;  // any non-zero epsilon is a stub
    REQUIRE_THROWS_AS(make(o), std::invalid_argument);

    o = Options{};
    o.delta_limit = true;
    REQUIRE_THROWS_AS(make(o), std::invalid_argument);

    o = Options{};
    o.delta_limit_subdivisions = 1;
    REQUIRE_THROWS_AS(make(o), std::invalid_argument);

    o = Options{};
    o.hidden_constraints = true;
    REQUIRE_THROWS_AS(make(o), std::invalid_argument);

    o = Options{};
    o.globally_biased = true;
    REQUIRE_THROWS_AS(make(o), std::invalid_argument);

    // Enum fields: only the default enumerator is accepted (defensive guard
    // against a future variant's value leaking in before the algorithm plan).
    o = Options{};
    o.selection = static_cast<Options::SelectionMode>(1);
    REQUIRE_THROWS_AS(make(o), std::invalid_argument);

    o = Options{};
    o.size_measure = static_cast<Options::SizeMeasure>(1);
    REQUIRE_THROWS_AS(make(o), std::invalid_argument);

    o = Options{};
    o.split_rule = static_cast<Options::SplitRule>(1);
    REQUIRE_THROWS_AS(make(o), std::invalid_argument);

    o = Options{};
    o.ties = static_cast<Options::TieSelection>(1);
    REQUIRE_THROWS_AS(make(o), std::invalid_argument);
}
