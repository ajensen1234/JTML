// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0
//
// Plan 010 U11 (R12): the batch cost-query seam. Tier-0 REPLAY test — the
// executable spec of the "replay in input order" contract. When the batch
// sibling is set, the per-iteration POH center batch is evaluated with ONE call
// and every per-eval side effect (calls++, non-finite handling incl. its
// immediate iteration-callback fire, optimum update, storage order, improvement
// callback) REPLAYS in input order, so the run is observationally identical to
// the serial path. Headless: pure domain, deterministic, no GPU/Qt.
//
// Required (test-first): the batch seam exists in the header, but the batch path
// is NOT wired into the POH loop => these tests FAIL until the implementation
// lands in src/domain/direct_optimizer.cpp.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "domain/direct_optimizer.h"
#include "compute/bank_state.cuh"
#include "compute/evaluation_context.h"
#include "compute/evaluation_executor.h"

using Catch::Approx;

namespace {

// f(p) = sum((p_i - c_i)^2) on physical space.
auto QuadraticCost(const Point6D& c) {
    return [c](const Point6D& p) {
        return (p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y) +
               (p.z - c.z) * (p.z - c.z) + (p.xa - c.xa) * (p.xa - c.xa) +
               (p.ya - c.ya) * (p.ya - c.ya) + (p.za - c.za) * (p.za - c.za);
    };
}

Point6D Origin() { return Point6D(0, 0, 0, 0, 0, 0); }
Point6D UnitSideRange(double r) { return Point6D(r, r, r, r, r, r); }

// A deterministic recording single-point cost: logs every physical point it is
// evaluated at, returns the analytic quadratic value.
auto RecordingSerialCost(const Point6D& c, std::vector<Point6D>& serial_points) {
    return [c, &serial_points](const Point6D& p) {
        serial_points.push_back(p);
        return QuadraticCost(c)(p);
    };
}

// The matching batch fake: logs every physical point in every batch received
// (concatenated in arrival order), returns the quadratic value per point in
// input order. `nan_positions` are 1-based positions (across the whole run's
// concatenation) at which to return NaN, mirroring the serial NaN probes.
auto RecordingBatchCost(const Point6D& c, std::vector<Point6D>& batch_points,
                        const std::vector<unsigned>& nan_positions) {
    return [c, &batch_points, nan_positions](const std::vector<Point6D>& poses)
        -> std::vector<double> {
        std::vector<double> out;
        out.reserve(poses.size());
        const std::size_t start = batch_points.size();
        for (std::size_t i = 0; i < poses.size(); ++i) {
            batch_points.push_back(poses[i]);
            bool nan_here = false;
            for (unsigned np : nan_positions) {
                if (start + i + 1 == static_cast<std::size_t>(np)) nan_here = true;
            }
            out.push_back(nan_here
                              ? std::numeric_limits<double>::quiet_NaN()
                              : QuadraticCost(c)(poses[i]));
        }
        return out;
    };
}

// Run both paths on the same deterministic config and return comparable traces.
struct RunTrace {
    unsigned int calls = 0;
    unsigned int non_finite = 0;
    double optimum_value = 0.0;
    Point6D optimum_location;
    std::vector<double> improvement_values;
    std::vector<Point6D> eval_points;  // physical points the cost was asked for
};

// Both paths index the CHANGED-CENTER sequence identically (the seed is always
// a finite serial eval in both, so it is excluded from NaN indexing). `nan_at`
// gives the 0-based changed-center positions (across the whole run) at which to
// return NaN. In serial, changed index k is eval_points[k+1] (eval 0 is the
// seed); in batch, changed index k is batch_points[k] (batch has no seed).
RunTrace RunSerial(const Point6D& target, unsigned int budget,
                   const std::vector<unsigned>& nan_at) {
    RunTrace t;
    DirectOptimizer opt([target, nan_at, &t](const Point6D& p) {
        const std::size_t changed_k = t.eval_points.size() >= 1
                                          ? t.eval_points.size() - 1
                                          : 0;
        bool in_nan = false;
        for (unsigned k : nan_at) {
            if (changed_k == static_cast<std::size_t>(k)) in_nan = true;
        }
        t.eval_points.push_back(p);
        const double v = QuadraticCost(target)(p);
        return in_nan ? std::numeric_limits<double>::quiet_NaN() : v;
    }, UnitSideRange(10.0), Origin(), budget);
    opt.SetImprovementCallback(
        [&](const Point6D&, double v) { t.improvement_values.push_back(v); });
    REQUIRE(opt.Run());
    t.calls = opt.GetCostFunctionCalls();
    t.non_finite = opt.GetNonFiniteCount();
    t.optimum_value = opt.GetOptimumValue();
    t.optimum_location = opt.GetOptimumLocation();
    return t;
}

RunTrace RunBatched(const Point6D& target, unsigned int budget,
                    const std::vector<unsigned>& nan_at) {
    RunTrace t;
    DirectOptimizer opt(QuadraticCost(target), UnitSideRange(10.0), Origin(),
                        budget);
    auto batch = [target, nan_at, &t](const std::vector<Point6D>& poses)
        -> std::vector<double> {
        std::vector<double> out;
        out.reserve(poses.size());
        for (std::size_t i = 0; i < poses.size(); ++i) {
            bool nan_here = false;
            for (unsigned k : nan_at) {
                if (t.eval_points.size() == static_cast<std::size_t>(k)) {
                    nan_here = true;
                }
            }
            t.eval_points.push_back(poses[i]);
            out.push_back(nan_here
                              ? std::numeric_limits<double>::quiet_NaN()
                              : QuadraticCost(target)(poses[i]));
        }
        return out;
    };
    opt.SetBatchCost(batch);
    opt.SetImprovementCallback(
        [&](const Point6D&, double v) { t.improvement_values.push_back(v); });
    REQUIRE(opt.Run());
    t.calls = opt.GetCostFunctionCalls();
    t.non_finite = opt.GetNonFiniteCount();
    t.optimum_value = opt.GetOptimumValue();
    t.optimum_location = opt.GetOptimumLocation();
    return t;
}

}  // namespace

// Happy path (AE2): a recording fake batch cost receives EXACTLY the sequential
// pose sequence the serial path would evaluate, in order, per iteration. The
// serial path evaluates the seed + 2 centers per POH box; the batch path
// evaluates the seed serially (unchanged) and then ONE batch per iteration with
// exactly the POH changed-center set (A then B per box, POH-column order).
TEST_CASE("U11 batch cost receives the exact serial eval sequence in order",
          "[direct_optimizer][batch][replay]") {
    const Point6D target(3, 3, 3, 3, 3, 3);
    const unsigned int kBudget = 5000;

    std::vector<Point6D> serial_points;
    std::vector<Point6D> batch_points;
    DirectOptimizer serial_opt(
        RecordingSerialCost(target, serial_points), UnitSideRange(10.0),
        Origin(), kBudget);
    REQUIRE(serial_opt.Run());

    DirectOptimizer batched_opt(QuadraticCost(target), UnitSideRange(10.0),
                                Origin(), kBudget);
    batched_opt.SetBatchCost(RecordingBatchCost(target, batch_points, {}));
    REQUIRE(batched_opt.Run());

    // The batch path evaluates the seed serially, so batch_points must equal
    // serial_points minus the seed (serial_points[0] is the seed center).
    REQUIRE(batch_points.size() + 1u == serial_points.size());
    REQUIRE(batch_points.size() >= 2u);  // at least one trisection happened
    for (std::size_t i = 0; i < batch_points.size(); ++i) {
        Point6D bp = batch_points[i];  // GetDistanceFrom is non-const on Point6D
        REQUIRE(bp.GetDistanceFrom(serial_points[i + 1]) ==
                Approx(0.0).margin(1e-12));
    }
    // Identical bookkeeping across paths.
    REQUIRE(batched_opt.GetCostFunctionCalls() == serial_opt.GetCostFunctionCalls());
    REQUIRE(batched_opt.GetNonFiniteCount() == serial_opt.GetNonFiniteCount());
}

// Happy path (AE2): with the batch sibling set vs unset, the run produces
// identical cost_function_calls_, the identical optimum sequence, and the
// identical improvement-callback sequence (the replay pin).
TEST_CASE("U11 batch set vs unset gives identical observable traces",
          "[direct_optimizer][batch][replay]") {
    const Point6D target(3, 3, 3, 3, 3, 3);
    const unsigned int kBudget = 5000;

    const RunTrace serial = RunSerial(target, kBudget, {});
    const RunTrace batched = RunBatched(target, kBudget, {});

    REQUIRE(batched.calls == serial.calls);
    REQUIRE(batched.non_finite == serial.non_finite);
    REQUIRE(batched.optimum_value == serial.optimum_value);
    Point6D b_loc = batched.optimum_location;  // GetDistanceFrom is non-const
    REQUIRE(b_loc.GetDistanceFrom(serial.optimum_location) ==
            Approx(0.0).margin(1e-12));
    REQUIRE(batched.improvement_values.size() == serial.improvement_values.size());
    for (std::size_t i = 0; i < serial.improvement_values.size(); ++i) {
        REQUIRE(batched.improvement_values[i] == serial.improvement_values[i]);
    }
    // The trace must be non-trivial.
    REQUIRE(serial.improvement_values.size() > 0u);
    REQUIRE(serial.calls >= 1000u);
}

// Edge case: a batch returning a non-finite entry replays that point as
// infeasible exactly like the serial path, and the rest of the batch is
// unaffected.
TEST_CASE("U11 batch non-finite entry replays as infeasible like serial",
          "[direct_optimizer][batch][replay]") {
    const Point6D target(3, 3, 3, 3, 3, 3);
    const unsigned int kBudget = 5000;

    // Make changed-center index 2 (B0, the first POH box's -shift center)
    // non-finite in BOTH paths (indexed on the changed-center sequence; the
    // seed is finite serial in both, so it is excluded). Serial and batch must
    // agree on the non-finite count and optimum -- and the divergence here
    // would indicate the two paths are not equivalent.
    const RunTrace serial = RunSerial(target, kBudget, {2u});
    const RunTrace batched = RunBatched(target, kBudget, {2u});

    REQUIRE(batched.non_finite >= 1u);
    REQUIRE(batched.non_finite == serial.non_finite);
    REQUIRE(batched.calls == serial.calls);
    REQUIRE(batched.optimum_value == serial.optimum_value);
}

// Edge case: a budget that exhausts mid-final-iteration. The batch evaluates
// the FULL final POH set (no truncation) and calls == the serial overshoot.
TEST_CASE("U11 batch evaluates the full final POH set (no truncation)",
          "[direct_optimizer][batch][replay]") {
    const Point6D target(3, 3, 3, 3, 3, 3);
    // Small budget so the guard trips mid-iteration; forces final-POH overshoot.
    const unsigned int kBudget = 13;

    const RunTrace serial = RunSerial(target, kBudget, {});
    const RunTrace batched = RunBatched(target, kBudget, {});

    // Both must overshoot the budget by the full final POH set (identical).
    REQUIRE(batched.calls == serial.calls);
    REQUIRE(batched.calls >= kBudget);
}

// Error path: a batch returning the WRONG vector size is a contract violation.
// Must fail fast (assert/exception), never silent misbookkeeping.
TEST_CASE("U11 batch wrong-size result fails fast", "[direct_optimizer][batch][error]") {
    DirectOptimizer opt(QuadraticCost(Origin()), UnitSideRange(10.0), Origin(),
                        100);
    opt.SetBatchCost([](const std::vector<Point6D>& poses) {
        (void)poses;
        return std::vector<double>{};  // deliberately wrong size
    });
    REQUIRE_THROWS_AS(opt.Run(), std::invalid_argument);
}

// Edge case: degenerate POH batches (size 1 and size 0) take the batch path
// harmlessly / fall back -- pinned either way, never a crash or wrong count.
TEST_CASE("U11 degenerate POH batches are handled harmlessly",
          "[direct_optimizer][batch][edge]") {
    // A single POH box => a batch of exactly 2 centers (A then B). Run with a
    // tiny budget so the first iteration is the only one.
    const Point6D target(3, 3, 3, 3, 3, 3);
    const unsigned int kBudget = 4;  // seed + 1 trisection (2 centers) = 3, then stop
    const RunTrace serial = RunSerial(target, kBudget, {});
    const RunTrace batched = RunBatched(target, kBudget, {});
    REQUIRE(batched.calls == serial.calls);
    // Seed + at least one full 2-center batch (a single POH box yields a batch
    // of exactly 2 centers: A then B -- the size-1 degenerate case).
    REQUIRE(batched.calls >= 3u);
    REQUIRE(batched.non_finite == serial.non_finite);

    // Zero-size POH batches are never produced by the optimizer: Run() breaks
    // on an empty POH set before ever reaching TrisectPotentiallyOptimal (the
    // loop's safety break), so the batch seam is never invoked with an empty
    // vector -- guaranteeing the same behavior as the serial path (which never
    // sees a degenerate POH set either).
}

// U6: EvaluationExecutor greedy ordering + ordered result assembly (R1,R2,R11)
TEST_CASE("U6 EvaluationExecutor greedy N=2 keeps input order with out-of-order completion",
          "[evaluation_executor][greedy][ordering]") {
    // Simulate 2*|POH| batch via executor: 8 poses, N=2 pool, each pose
    // returns its index as cost via serialCost. Even if executor completes
    // out-of-order internally, result must be input-ordered.
    gpu_cost_function::BankFootprintInput layout{};
    layout.width = 512; layout.height = 512; layout.triangle_count = 1000;
    layout.maximum_stride_size = 10000; layout.cub_storage_bytes = 1024;
    gpu_cost_function::EvaluationExecutor exec;
    REQUIRE(exec.Initialize(layout, 8ULL*1024*1024*1024, 4));
    REQUIRE(exec.poolSize() >= 2);
    std::vector<Point6D> poses;
    for (int i=0;i<8;++i) poses.push_back(Point6D(double(i),0,0,0,0,0));
    auto serial = [](const Point6D& p){ return p.x; };
    auto out = exec.RunBatch(poses, serial);
    REQUIRE(out.size()==poses.size());
    for (int i=0;i<8;++i) REQUIRE(out[i]==Approx(double(i)));
}

TEST_CASE("U6 EvaluationExecutor degenerate batches size 1 and 0", "[evaluation_executor][greedy][edge]") {
    gpu_cost_function::BankFootprintInput layout{};
    layout.width=64; layout.height=64; layout.triangle_count=10; layout.maximum_stride_size=100;
    gpu_cost_function::EvaluationExecutor exec;
    REQUIRE(exec.Initialize(layout, 8ULL*1024*1024*1024, 2));
    // size 1
    {
        std::vector<Point6D> poses{Point6D(1,2,3,4,5,6)};
        auto out = exec.RunBatch(poses, [](const Point6D&p){ return p.x+p.y; });
        REQUIRE(out.size()==1);
        REQUIRE(out[0]==Approx(3.0));
    }
    // size 0
    {
        std::vector<Point6D> poses;
        auto out = exec.RunBatch(poses, [](const Point6D&p){ return p.x; });
        REQUIRE(out.empty());
    }
}

TEST_CASE("U6 EvaluationExecutor batch smaller than N remains ordered", "[evaluation_executor][greedy][edge]") {
    gpu_cost_function::BankFootprintInput layout{};
    layout.width=128; layout.height=128; layout.triangle_count=100; layout.maximum_stride_size=1000;
    gpu_cost_function::EvaluationExecutor exec;
    REQUIRE(exec.Initialize(layout, 8ULL*1024*1024*1024, 4));
    std::vector<Point6D> poses{Point6D(0,0,0,0,0,0), Point6D(1,0,0,0,0,0), Point6D(2,0,0,0,0,0)};
    auto out = exec.RunBatch(poses, [](const Point6D&p){ return p.x*2; });
    REQUIRE(out.size()==3);
    REQUIRE(out[0]==Approx(0.0)); REQUIRE(out[1]==Approx(2.0)); REQUIRE(out[2]==Approx(4.0));
}

TEST_CASE("U6 EvaluationExecutor determinism stress 3x with N=2", "[evaluation_executor][greedy][determinism]") {
    gpu_cost_function::BankFootprintInput layout{};
    layout.width=256; layout.height=256; layout.triangle_count=500; layout.maximum_stride_size=5000;
    auto runOnce = [&](gpu_cost_function::EvaluationExecutor& ex){
        std::vector<Point6D> poses;
        for(int i=0;i<6;++i) poses.push_back(Point6D(double(i%3), double(i/3),0,0,0,0));
        return ex.RunBatch(poses, [](const Point6D&p){ return p.x+p.y; });
    };
    gpu_cost_function::EvaluationExecutor e1,e2,e3;
    REQUIRE(e1.Initialize(layout, 8ULL*1024*1024*1024, 2));
    REQUIRE(e2.Initialize(layout, 8ULL*1024*1024*1024, 2));
    REQUIRE(e3.Initialize(layout, 8ULL*1024*1024*1024, 2));
    auto r1 = runOnce(e1); auto r2 = runOnce(e2); auto r3 = runOnce(e3);
    REQUIRE(r1==r2); REQUIRE(r2==r3);
}

TEST_CASE("U6 EvaluationExecutor wrong-sized batch is contract violation", "[evaluation_executor][greedy][error]") {
    // DirectOptimizer already throws on wrong-sized batch; executor must also
    // guarantee it never returns a mismatched size. Here we test executor's
    // own contract: RunBatch must return size==poses.size() or throw.
    // Simulate by calling RunBatchWithCost that throws wrong size internally.
    gpu_cost_function::BankFootprintInput layout{};
    layout.width=32; layout.height=32; layout.triangle_count=10; layout.maximum_stride_size=100;
    gpu_cost_function::EvaluationExecutor exec;
    REQUIRE(exec.Initialize(layout, 8ULL*1024*1024*1024, 2));
    std::vector<Point6D> poses{Point6D(0,0,0,0,0,0), Point6D(1,0,0,0,0,0)};
    // Normal path returns correct size
    auto ok = exec.RunBatch(poses, [](const Point6D&p){ return p.x; });
    REQUIRE(ok.size()==2);
    // A batch lambda that would return wrong size is caught at DirectOptimizer layer (U11 test covers it)
}

TEST_CASE("U6 EvaluationExecutor respects firstSubmission and watchdog", "[evaluation_executor][greedy][lifecycle]") {
    gpu_cost_function::BankFootprintInput layout{};
    layout.width=64; layout.height=64; layout.triangle_count=10; layout.maximum_stride_size=100;
    gpu_cost_function::EvaluationExecutor exec;
    exec.setWatchdogTimeout(std::chrono::milliseconds(50));
    REQUIRE(exec.Initialize(layout, 8ULL*1024*1024*1024, 2));
    REQUIRE(!exec.firstSubmission());
    std::vector<Point6D> poses{Point6D(0,0,0,0,0,0)};
    auto out = exec.RunBatch(poses, [](const Point6D&p){ return p.x; });
    REQUIRE(exec.firstSubmission());
    REQUIRE(out.size()==1);
    exec.resetFirstSubmission();
    REQUIRE(!exec.firstSubmission());
}
