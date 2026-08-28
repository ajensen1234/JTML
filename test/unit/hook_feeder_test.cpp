// Plan 012 U4 — hook-driven greedy feeder + ComposeDirectDilationScore
// Headless injection tests. Tests fail until U4 hooks land.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "compute/batch_outcome.h"
#include "compute/evaluation_executor.h"
#include "compute/graph_recipe.h"
#include "domain/data_structures_6D.h"

using Catch::Approx;
using gpu_cost_function::BatchOutcome;
using gpu_cost_function::EvaluationExecutor;
using gpu_cost_function::PollResult;

// ---------------------------------------------------------------------------
// Composition helper (pure, CUDA-free). Shared by complete() and
// completeFromPins().
// ---------------------------------------------------------------------------

TEST_CASE(
    "ComposeDirectDilationScore matches known composition",
    "[hook_feeder][composition]") {
    // white_sum=100, pixel=10, distance=5, edge=9 => 100 + (-10) + 5/9.1
    double s = gpu_cost_function::ComposeDirectDilationScore(100, 10, 5, 9);
    REQUIRE(s == Approx(90.54945054945055).epsilon(1e-12));
    // edge_count=0 => guard +0.1 => 5/0.1 = 50 => 100-10+50 = 140
    double s0 = gpu_cost_function::ComposeDirectDilationScore(100, 10, 5, 0);
    REQUIRE(s0 == Approx(140.0));
    // another pin
    double s2 = gpu_cost_function::ComposeDirectDilationScore(50, 20, 10, 4);
    // 50 -20 + 10/4.1 ≈ 30 + 2.4390243902 = 32.4390243902
    REQUIRE(s2 == Approx(32.4390243902439).epsilon(1e-12));
    // zero distances
    double s3 = gpu_cost_function::ComposeDirectDilationScore(0, 0, 0, 0);
    REQUIRE(s3 == Approx(0.0));
}

// ---------------------------------------------------------------------------
// Hook-driven greedy feeder — ordered store, no re-poll, error paths
// ---------------------------------------------------------------------------

namespace {
Point6D MakePose(double v) {
    return Point6D(v, v, v, v, v, v);
}
}  // namespace

TEST_CASE(
    "hook feeder completes out of order yet stores input order",
    "[hook_feeder][ordering]") {
    EvaluationExecutor exec;
    exec.pool().InitForTest(3);
    REQUIRE_FALSE(exec.firstSubmission());

    std::vector<std::pair<std::size_t, std::size_t>> enqueueCalls;
    int pollCalls = 0;
    int completeCalls = 0;
    bool teardownCalled = false;

    exec.InstallEnqueueHook(
        [&](std::size_t ctxIdx, std::size_t inputPos, const Point6D&) -> bool {
            enqueueCalls.emplace_back(ctxIdx, inputPos);
            return true;
        });
    exec.InstallPollHook([&](std::size_t) -> PollResult {
        ++pollCalls;
        return PollResult::Done;
    });
    exec.InstallCompleteFromPinsHook([&](std::size_t ctxIdx) -> double {
        ++completeCalls;
        // Return 10 + inputPos via lookup of last enqueued inputPos for this
        // ctx? For determinism, use ctxIdx mapping: but we want per-inputPos
        // value. Instead derive from ctxIdx's last inputPos stored in
        // enqueueCalls. Simpler: return based on count order — not ideal. We'll
        // map via ctx. The executor's Lease remembers inputPos, but hook only
        // gets ctxIdx. For this test, return distinct per ctx and rely on
        // executor's ordered store. To make result checkable, return 10 +
        // ctxIdx (still distinct) and verify ordering via inputPos mapping
        // below. Better: stash inputPos per ctxIdx in map. But we don't have
        // that map here; instead return 10 + ctxIdx and assert ordered after
        // mapping.
        return 10.0 + static_cast<double>(ctxIdx);
    });
    exec.InstallTeardownHook([&]() { teardownCalled = true; });

    // Use per-pose cost with index to generate inputPos-based values as serial
    // fallback cross-check. For hook path, costWithIndex is still passed but
    // hook-driven path should use completeFromPinsHook values. We'll make
    // completeFromPins return 10 + inputPos by capturing via map built in
    // enqueue. Re-install with map-aware version to get correct values:
    std::unordered_map<std::size_t, std::size_t> ctxToInput;
    exec.InstallEnqueueHook(
        [&](std::size_t ctxIdx, std::size_t inputPos, const Point6D&) -> bool {
            enqueueCalls.emplace_back(ctxIdx, inputPos);
            ctxToInput[ctxIdx] = inputPos;
            return true;
        });
    // Need to re-install poll/complete with map awareness (poll count already
    // tracked above, reset)
    pollCalls = 0;
    completeCalls = 0;
    enqueueCalls.clear();
    ctxToInput.clear();
    exec.InstallPollHook([&](std::size_t) -> PollResult {
        ++pollCalls;
        return PollResult::Done;
    });
    exec.InstallCompleteFromPinsHook([&](std::size_t ctxIdx) -> double {
        ++completeCalls;
        auto it = ctxToInput.find(ctxIdx);
        REQUIRE(it != ctxToInput.end());
        return 10.0 + static_cast<double>(it->second);
    });

    std::vector<Point6D> poses{MakePose(1), MakePose(2), MakePose(3)};
    auto costWithIndex = [](const Point6D&, std::size_t idx) -> double {
        return 10.0 + static_cast<double>(idx);
    };

    auto outcome = exec.RunBatchWithCost(poses, costWithIndex);
    REQUIRE(outcome.isOrderedScores());
    REQUIRE(outcome.scores.size() == 3);
    REQUIRE(outcome.scores[0] == Approx(10.0));
    REQUIRE(outcome.scores[1] == Approx(11.0));
    REQUIRE(outcome.scores[2] == Approx(12.0));
    REQUIRE(enqueueCalls.size() == 3);
    // One poll+complete per lease, no re-poll of completed context
    REQUIRE(pollCalls == 3);
    REQUIRE(completeCalls == 3);
    REQUIRE(exec.firstSubmission());
    REQUIRE_FALSE(teardownCalled);
}

TEST_CASE(
    "out-of-order completion still yields input-ordered scores (AE1)",
    "[hook_feeder][ordering][ae1]") {
    EvaluationExecutor exec;
    exec.pool().InitForTest(4);
    // Two poses -> two contexts. Force the SECOND-launched context (inputPos 1)
    // to complete FIRST by making the loop poll it Done while the first
    // (inputPos 0) stays Pending. A buggy executor appending in completion
    // order (push_back) would produce [11,10]; the correct indexed store must
    // produce [10,11].
    std::unordered_map<std::size_t, std::size_t> ctxToInput;
    int pollForCtxA = 0;
    int completeCalls = 0;
    std::vector<std::size_t> completeOrder;
    exec.InstallEnqueueHook(
        [&](std::size_t ctxIdx, std::size_t inputPos, const Point6D&) -> bool {
            ctxToInput[ctxIdx] = inputPos;
            return true;
        });
    exec.InstallPollHook([&](std::size_t ctxIdx) -> PollResult {
        auto it = ctxToInput.find(ctxIdx);
        REQUIRE(it != ctxToInput.end());
        if (it->second == 1) {
            return PollResult::Done;
        }
        ++pollForCtxA;
        return pollForCtxA >= 2 ? PollResult::Done : PollResult::Pending;
    });
    exec.InstallCompleteFromPinsHook([&](std::size_t ctxIdx) -> double {
        ++completeCalls;
        completeOrder.push_back(ctxIdx);
        auto it = ctxToInput.find(ctxIdx);
        REQUIRE(it != ctxToInput.end());
        return 10.0 + static_cast<double>(it->second);
    });
    exec.InstallTeardownHook([&]() {});
    std::vector<Point6D> poses{MakePose(1), MakePose(2)};
    auto costWithIndex = [](const Point6D&, std::size_t) -> double {
        return 0.0;
    };
    auto outcome = exec.RunBatchWithCost(poses, costWithIndex);
    REQUIRE(outcome.isOrderedScores());
    REQUIRE(outcome.scores.size() == 2);
    // Input order regardless of completion order: [10 (inputPos 0), 11
    // (inputPos 1)]
    REQUIRE(outcome.scores[0] == Approx(10.0));
    REQUIRE(outcome.scores[1] == Approx(11.0));
    // Prove completion was out of order: inputPos 1's context completed first.
    REQUIRE(completeOrder.size() == 2);
    REQUIRE(completeOrder[0] != completeOrder[1]);
    REQUIRE(ctxToInput[completeOrder[0]] == 1);
    REQUIRE(ctxToInput[completeOrder[1]] == 0);
    REQUIRE(completeCalls == 2);
}

TEST_CASE(
    "Pending leases stay in flight and are completed when ready (no re-poll)",
    "[hook_feeder][pending]") {
    EvaluationExecutor exec;
    exec.pool().InitForTest(2);

    std::unordered_map<std::size_t, int> pollCountPerCtx;
    std::unordered_set<std::size_t> completed;
    std::unordered_map<std::size_t, std::size_t> ctxToInput;

    exec.InstallEnqueueHook(
        [&](std::size_t ctxIdx, std::size_t inputPos, const Point6D&) -> bool {
            ctxToInput[ctxIdx] = inputPos;
            pollCountPerCtx[ctxIdx] = 0;
            return true;
        });
    exec.InstallPollHook([&](std::size_t ctxIdx) -> PollResult {
        int& c = pollCountPerCtx[ctxIdx];
        ++c;
        if (c == 1) {
            return PollResult::Pending;
        }
        return PollResult::Done;
    });
    exec.InstallCompleteFromPinsHook([&](std::size_t ctxIdx) -> double {
        REQUIRE(completed.find(ctxIdx) == completed.end());
        completed.insert(ctxIdx);
        auto it = ctxToInput.find(ctxIdx);
        REQUIRE(it != ctxToInput.end());
        return 100.0 + static_cast<double>(it->second);
    });
    exec.InstallTeardownHook([]() {});

    std::vector<Point6D> poses{MakePose(1), MakePose(2)};
    auto outcome = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t idx) -> double {
            return 100.0 + static_cast<double>(idx);
        });
    REQUIRE(outcome.isOrderedScores());
    REQUIRE(outcome.scores.size() == 2);
    REQUIRE(outcome.scores[0] == Approx(100.0));
    REQUIRE(outcome.scores[1] == Approx(101.0));
    // Each ctx polled exactly twice (Pending then Done), no extra re-poll after
    // Done
    for (auto& kv : pollCountPerCtx) {
        REQUIRE(kv.second == 2);
    }
    REQUIRE(completed.size() == 2);
}

TEST_CASE(
    "1-pose and batch < pool size still ordered with hooks",
    "[hook_feeder][edge]") {
    {
        EvaluationExecutor exec;
        exec.pool().InitForTest(4);
        std::unordered_map<std::size_t, std::size_t> ctxToInput;
        exec.InstallEnqueueHook(
            [&](std::size_t ctxIdx,
                std::size_t inputPos,
                const Point6D&) -> bool {
                ctxToInput[ctxIdx] = inputPos;
                return true;
            });
        exec.InstallPollHook(
            [](std::size_t) -> PollResult { return PollResult::Done; });
        exec.InstallCompleteFromPinsHook([&](std::size_t ctxIdx) -> double {
            return 5.0 + static_cast<double>(ctxToInput[ctxIdx]);
        });
        exec.InstallTeardownHook([]() {});

        std::vector<Point6D> poses{MakePose(7)};
        auto o =
            exec.RunBatchWithCost(poses, [](const Point6D&, std::size_t idx) {
                return 5.0 + static_cast<double>(idx);
            });
        REQUIRE(o.isOrderedScores());
        REQUIRE(o.scores.size() == 1);
        REQUIRE(o.scores[0] == Approx(5.0));
    }
    {
        EvaluationExecutor exec;
        exec.pool().InitForTest(4);
        std::unordered_map<std::size_t, std::size_t> ctxToInput;
        exec.InstallEnqueueHook(
            [&](std::size_t ctxIdx,
                std::size_t inputPos,
                const Point6D&) -> bool {
                ctxToInput[ctxIdx] = inputPos;
                return true;
            });
        exec.InstallPollHook(
            [](std::size_t) -> PollResult { return PollResult::Done; });
        exec.InstallCompleteFromPinsHook([&](std::size_t ctxIdx) -> double {
            return 1.0 + static_cast<double>(ctxToInput[ctxIdx] * 10);
        });
        exec.InstallTeardownHook([]() {});

        std::vector<Point6D> poses{MakePose(1), MakePose(2)};
        auto o =
            exec.RunBatchWithCost(poses, [](const Point6D&, std::size_t idx) {
                return 1.0 + static_cast<double>(idx * 10);
            });
        REQUIRE(o.isOrderedScores());
        REQUIRE(o.scores.size() == 2);
        REQUIRE(o.scores[0] == Approx(1.0));
        REQUIRE(o.scores[1] == Approx(11.0));
    }
}

TEST_CASE(
    "pacing hook invoked exactly once per zero-completion sweep",
    "[hook_feeder][pacing]") {
    // Plan 013 U1: the greedy loop must call the injected pacing hook exactly
    // once when a sweep completes nothing (never zero-delay hot-spin, never
    // multiple pacing calls per zero-completion sweep). Pending->Done fake:
    // sweep 1 polls both ctxs (Pending, Pending) -> one zero-completion sweep
    // -> pacing must be invoked exactly once; sweep 2 both Done -> no pacing.
    EvaluationExecutor exec;
    exec.pool().InitForTest(2);
    exec.setWatchdogTimeout(std::chrono::milliseconds(50));

    std::unordered_map<std::size_t, int> pollCountPerCtx;
    std::unordered_map<std::size_t, std::size_t> ctxToInput;
    int pacingCalls = 0;

    exec.InstallEnqueueHook(
        [&](std::size_t ctxIdx, std::size_t inputPos, const Point6D&) -> bool {
            ctxToInput[ctxIdx] = inputPos;
            pollCountPerCtx[ctxIdx] = 0;
            return true;
        });
    exec.InstallPollHook([&](std::size_t ctxIdx) -> PollResult {
        int& c = pollCountPerCtx[ctxIdx];
        ++c;
        return c == 1 ? PollResult::Pending : PollResult::Done;
    });
    exec.InstallCompleteFromPinsHook([&](std::size_t ctxIdx) -> double {
        return 100.0 + static_cast<double>(ctxToInput[ctxIdx]);
    });
    exec.InstallTeardownHook([]() {});
    exec.InstallPacingHook([&]() { ++pacingCalls; });

    std::vector<Point6D> poses{MakePose(1), MakePose(2)};
    auto o = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t) -> double { return 0.0; });

    REQUIRE(o.isOrderedScores());
    REQUIRE(o.scores[0] == Approx(100.0));
    REQUIRE(o.scores[1] == Approx(101.0));
    // Exactly one zero-completion sweep -> exactly one pacing call.
    REQUIRE(pacingCalls == 1);
    // Poll-count pins still hold: each ctx polled exactly twice (Pending,
    // Done).
    for (auto& kv : pollCountPerCtx) {
        REQUIRE(kv.second == 2);
    }
}

TEST_CASE(
    "no pacing invocation when the sweep completes work",
    "[hook_feeder][pacing]") {
    // Plan 013 U1: completions must never be delayed by the pacing hook
    // (continue-immediately on Done). All-Done fake -> pacing count == 0.
    EvaluationExecutor exec;
    exec.pool().InitForTest(2);
    int pacingCalls = 0;

    exec.InstallEnqueueHook(
        [](std::size_t, std::size_t, const Point6D&) -> bool { return true; });
    exec.InstallPollHook(
        [](std::size_t) -> PollResult { return PollResult::Done; });
    exec.InstallCompleteFromPinsHook([](std::size_t) -> double { return 1.0; });
    exec.InstallTeardownHook([]() {});
    exec.InstallPacingHook([&]() { ++pacingCalls; });

    std::vector<Point6D> poses{MakePose(1), MakePose(2)};
    auto o = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t) -> double { return 0.0; });

    REQUIRE(o.isOrderedScores());
    REQUIRE(o.scores.size() == 2);
    REQUIRE(pacingCalls == 0);
}

TEST_CASE(
    "sole remaining context is polled once per sweep, watchdog-porous",
    "[hook_feeder][u2][sole]") {
    // Plan 013 U2: when only one context remains in flight (pool 1, 1 pose),
    // the loop must poll it once per iteration (Pending->Done, exactly 2
    // polls), never hot-spin, and the watchdog must still poison on a hang. The
    // general sweep already satisfies this; the pin guards regressions to the
    // sole tail.
    EvaluationExecutor exec;
    exec.pool().InitForTest(1);
    exec.setWatchdogTimeout(std::chrono::milliseconds(500));

    int polls = 0;
    int pacingCalls = 0;
    exec.InstallEnqueueHook(
        [](std::size_t, std::size_t, const Point6D&) -> bool { return true; });
    exec.InstallPollHook([&](std::size_t) -> PollResult {
        ++polls;
        return polls == 1 ? PollResult::Pending : PollResult::Done;
    });
    exec.InstallCompleteFromPinsHook([](std::size_t) -> double { return 7.0; });
    exec.InstallTeardownHook([]() {});
    exec.InstallPacingHook([&]() { ++pacingCalls; });

    std::vector<Point6D> poses{MakePose(1)};
    auto o = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t) -> double { return 0.0; });

    REQUIRE(o.isOrderedScores());
    REQUIRE(o.scores[0] == Approx(7.0));
    // Exactly 2 polls (Pending, Done) — never a third poll of the completed
    // lease.
    REQUIRE(polls == 2);
    // One zero-completion sweep (the first poll was Pending) -> exactly one
    // pacing call.
    REQUIRE(pacingCalls == 1);
}

TEST_CASE(
    "sole-context hang still poisons the executor",
    "[hook_feeder][u2][sole][poison]") {
    // Plan 013 U2: a hung sole context must still poison (bounded wait, not a
    // blocking cudaEventSynchronize that would defeat the watchdog).
    EvaluationExecutor exec;
    exec.pool().InitForTest(1);
    exec.setWatchdogTimeout(std::chrono::milliseconds(1));
    exec.InstallEnqueueHook(
        [](std::size_t, std::size_t, const Point6D&) -> bool { return true; });
    exec.InstallPollHook(
        [](std::size_t) -> PollResult { return PollResult::Pending; });
    exec.InstallCompleteFromPinsHook([](std::size_t) -> double { return 0.0; });
    exec.InstallTeardownHook([]() {});

    std::vector<Point6D> poses{MakePose(1)};
    auto o = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t) -> double { return 0.0; });
    REQUIRE(o.kind == BatchOutcome::Kind::WatchdogPoisoned);
    REQUIRE(exec.isPoisoned());
    REQUIRE(exec.pool().IsPoisoned(0));
    REQUIRE(exec.pool().IsInFlight(0));
}

TEST_CASE(
    "poll Error returns PostLaunchAbort and drains",
    "[hook_feeder][error]") {
    EvaluationExecutor exec;
    exec.pool().InitForTest(3);
    bool teardownCalled = false;
    int enqueueCount = 0;
    exec.InstallEnqueueHook(
        [&](std::size_t, std::size_t, const Point6D&) -> bool {
            ++enqueueCount;
            return true;
        });
    int pollCalls = 0;
    exec.InstallPollHook([&](std::size_t ctxIdx) -> PollResult {
        ++pollCalls;
        // Fail on second distinct ctx polled
        if (pollCalls == 2) {
            return PollResult::Error;
        }
        return PollResult::Done;
    });
    exec.InstallCompleteFromPinsHook([](std::size_t) -> double { return 1.0; });
    exec.InstallTeardownHook([&]() { teardownCalled = true; });

    std::vector<Point6D> poses{MakePose(1), MakePose(2), MakePose(3)};
    auto o = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t) -> double { return 1.0; });
    REQUIRE(o.kind == BatchOutcome::Kind::PostLaunchAbort);
    REQUIRE(teardownCalled);
    REQUIRE(exec.firstSubmission());
    // After abort, pool contexts should be ForceReleased (idle)
    REQUIRE_FALSE(exec.pool().IsInFlight(0));
    REQUIRE_FALSE(exec.pool().IsInFlight(1));
    // Must not have produced OrderedScores
    REQUIRE_FALSE(o.isOrderedScores());
}

TEST_CASE(
    "overflow (non-finite completeFromPins) is an abort not a non-finite "
    "DIRECT cost",
    "[hook_feeder][error][overflow]") {
    EvaluationExecutor exec;
    exec.pool().InitForTest(2);
    exec.InstallEnqueueHook(
        [](std::size_t, std::size_t, const Point6D&) -> bool { return true; });
    exec.InstallPollHook(
        [](std::size_t) -> PollResult { return PollResult::Done; });
    exec.InstallCompleteFromPinsHook([](std::size_t) -> double {
        return std::numeric_limits<double>::quiet_NaN();
    });
    exec.InstallTeardownHook([]() {});

    std::vector<Point6D> poses{MakePose(1), MakePose(2)};
    auto o = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t) -> double { return 1.0; });
    REQUIRE(o.kind == BatchOutcome::Kind::PostLaunchAbort);
    REQUIRE_FALSE(o.isOrderedScores());
}

TEST_CASE(
    "poll Error after some Done does not leave a completed context re-polled",
    "[hook_feeder][error][deep]") {
    EvaluationExecutor exec;
    exec.pool().InitForTest(3);
    std::unordered_map<std::size_t, int> pollCount;
    std::unordered_set<std::size_t> doneSet;
    bool teardownCalled = false;
    bool firstPollDone = false;

    exec.InstallEnqueueHook(
        [&](std::size_t, std::size_t, const Point6D&) -> bool {
            return true;  // do NOT pre-populate pollCount — the loop enqueues
                          // all up front
        });
    exec.InstallPollHook([&](std::size_t ctxIdx) -> PollResult {
        // A completed context must never be polled again (C6).
        if (doneSet.find(ctxIdx) != doneSet.end()) {
            FAIL("re-polled a completed context");
        }
        ++pollCount[ctxIdx];
        // The FIRST context the loop polls completes (Done); every later
        // distinct poll errors. The greedy loop polls all in-flight then aborts
        // on the first Error, so exactly one context completes before the
        // abort.
        if (!firstPollDone) {
            firstPollDone = true;
            return PollResult::Done;
        }
        return PollResult::Error;
    });
    exec.InstallCompleteFromPinsHook([&](std::size_t ctxIdx) -> double {
        doneSet.insert(ctxIdx);
        return 42.0;
    });
    exec.InstallTeardownHook([&]() { teardownCalled = true; });

    std::vector<Point6D> poses{MakePose(1), MakePose(2), MakePose(3)};
    auto o = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t) -> double { return 42.0; });
    REQUIRE(o.kind == BatchOutcome::Kind::PostLaunchAbort);
    REQUIRE(teardownCalled);
    // Exactly one context completed (Done) before the Error aborted the batch.
    REQUIRE(doneSet.size() == 1);
    // No context was polled more than once (no re-poll of a completed context).
    for (auto& kv : pollCount) {
        REQUIRE(kv.second == 1);
    }
}

// ---------------------------------------------------------------------------
// Plan 012 U5 — watchdog hang poisons, refuses further work, shutdown safety
// ---------------------------------------------------------------------------

TEST_CASE(
    "watchdog hang poisons in-flight contexts and returns WatchdogPoisoned",
    "[hook_feeder][u5][poison]") {
    EvaluationExecutor exec;
    exec.pool().InitForTest(2);
    exec.setWatchdogTimeout(std::chrono::milliseconds(1));
    exec.InstallEnqueueHook(
        [](std::size_t, std::size_t, const Point6D&) -> bool { return true; });
    exec.InstallPollHook(
        [](std::size_t) -> PollResult { return PollResult::Pending; });
    exec.InstallCompleteFromPinsHook([](std::size_t) -> double { return 0.0; });
    exec.InstallTeardownHook([]() {});
    std::vector<Point6D> poses{MakePose(1), MakePose(2)};
    auto o = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t) -> double { return 1.0; });
    REQUIRE(o.kind == BatchOutcome::Kind::WatchdogPoisoned);
    REQUIRE(exec.isPoisoned());
    // Hang is terminal: BOTH in-flight contexts must be LeavePoisoned (kept
    // checked out, marked poisoned), NOT ForceReleased. A ForceRelease-on-
    // watchdog impl would leave them not-poisoned / reusable and fail here.
    REQUIRE(exec.pool().IsPoisoned(0));
    REQUIRE(exec.pool().IsPoisoned(1));
    // Operational half: poisoned contexts STAY checked out (Checkout refuses).
    REQUIRE(exec.pool().IsInFlight(0));
    REQUIRE(exec.pool().IsInFlight(1));
}

TEST_CASE(
    "a poisoned executor refuses further RunBatch and Prepare",
    "[hook_feeder][u5][poison]") {
    EvaluationExecutor exec;
    exec.pool().InitForTest(2);
    exec.setWatchdogTimeout(std::chrono::milliseconds(1));
    exec.InstallEnqueueHook(
        [](std::size_t, std::size_t, const Point6D&) -> bool { return true; });
    exec.InstallPollHook(
        [](std::size_t) -> PollResult { return PollResult::Pending; });
    exec.InstallCompleteFromPinsHook([](std::size_t) -> double { return 0.0; });
    exec.InstallTeardownHook([]() {});
    std::vector<Point6D> poses{MakePose(1), MakePose(2)};
    auto first = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t) -> double { return 1.0; });
    REQUIRE(first.kind == BatchOutcome::Kind::WatchdogPoisoned);
    REQUIRE(exec.isPoisoned());

    bool enqueued = false;
    exec.InstallEnqueueHook(
        [&](std::size_t, std::size_t, const Point6D&) -> bool {
            enqueued = true;
            return true;
        });
    exec.InstallPollHook(
        [](std::size_t) -> PollResult { return PollResult::Done; });
    exec.InstallCompleteFromPinsHook(
        [](std::size_t) -> double { return 42.0; });
    auto second = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t) -> double { return 42.0; });
    REQUIRE(second.kind == BatchOutcome::Kind::WatchdogPoisoned);
    REQUIRE_FALSE(enqueued);

    bool prepared = false;
    exec.InstallPrepareHook(
        [&](std::size_t, const gpu_cost_function::GraphRecipeKey&) -> void* {
            prepared = true;
            return reinterpret_cast<void*>(0x1);
        });
    gpu_cost_function::GraphRecipeKey key;
    key.recipeId = "direct_dilation_monoplane";
    auto prep = exec.Prepare(key, 1);
    REQUIRE(prep.kind == BatchOutcome::Kind::WatchdogPoisoned);
    REQUIRE_FALSE(prepared);
}

TEST_CASE("Shutdown after poison does not crash", "[hook_feeder][u5][poison]") {
    EvaluationExecutor exec;
    exec.pool().InitForTest(2);
    exec.setWatchdogTimeout(std::chrono::milliseconds(1));
    exec.InstallEnqueueHook(
        [](std::size_t, std::size_t, const Point6D&) -> bool { return true; });
    exec.InstallPollHook(
        [](std::size_t) -> PollResult { return PollResult::Pending; });
    exec.InstallCompleteFromPinsHook([](std::size_t) -> double { return 0.0; });
    exec.InstallTeardownHook([]() {});
    std::vector<Point6D> poses{MakePose(1), MakePose(2)};
    auto o = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t) -> double { return 1.0; });
    REQUIRE(o.kind == BatchOutcome::Kind::WatchdogPoisoned);
    REQUIRE(exec.isPoisoned());
    exec.Shutdown();
    REQUIRE(exec.pool().size() == 0);
}

TEST_CASE(
    "ordinary poll Error is PostLaunchAbort not poison",
    "[hook_feeder][u5][poison]") {
    EvaluationExecutor exec;
    exec.pool().InitForTest(2);
    exec.InstallEnqueueHook(
        [](std::size_t, std::size_t, const Point6D&) -> bool { return true; });
    exec.InstallPollHook(
        [](std::size_t) -> PollResult { return PollResult::Error; });
    exec.InstallCompleteFromPinsHook([](std::size_t) -> double { return 0.0; });
    exec.InstallTeardownHook([]() {});
    std::vector<Point6D> poses{MakePose(1), MakePose(2)};
    auto o = exec.RunBatchWithCost(
        poses, [](const Point6D&, std::size_t) -> double { return 1.0; });
    REQUIRE(o.kind == BatchOutcome::Kind::PostLaunchAbort);
    REQUIRE_FALSE(exec.isPoisoned());
    for (std::size_t i = 0; i < 2; ++i) {
        REQUIRE_FALSE(exec.pool().IsPoisoned(i));
        // Ordinary error is a drain + ForceRelease: contexts must be released
        // (reusable), NOT left checked-out (that would be a leak/poison).
        REQUIRE_FALSE(exec.pool().IsInFlight(i));
    }
}
