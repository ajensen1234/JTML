// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U5: OptimizerRunControllerCore pins (R7, R13, R15, R16; F1;
// AE1) — the shared Qt/GPU-free run core (gate + state machine + epoch +
// progress mapping + seed lifecycle). Ports the pure cases of
// test/unit/experimental_optimizer_gate_test.cpp against the shared core
// (the SingleModelOnly bridge policy stays in the bridge test), plus the
// state-machine/progress/seed surfaces the controller shell drives.
// Deterministic Catch2, no Qt, no event loop, no GPU.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "coordinator/optimizer_run_controller_core.h"

using Catch::Approx;
using jta::OptimizerRunControllerCore;

namespace {

/*The gate test's runnable input shape, ported to the core's GateInput (the
 * same plain values the intent tests pin; previous is derived, H2).*/
OptimizerRunControllerCore::GateInput RunnableInput() {
    OptimizerRunControllerCore::GateInput in;
    in.selected_model_rows = {0};
    in.current_frame = 0;
    in.frame_count = 3;
    in.model_current_index = 0;
    in.model_count = 2;
    in.pose_frame_count = 3;
    in.pose_model_count = 2;
    return in;
}

OptimizerRunControllerCore::ProgressBudgets Budgets() {
    OptimizerRunControllerCore::ProgressBudgets b;
    b.trunk_budget = 1000;
    b.branch_budget = 500;
    b.number_branches = 2;
    b.enable_branch = true;
    b.leaf_budget = 200;
    b.enable_leaf = true;
    return b;  // cumulative = 2200
}

}  // namespace

TEST_CASE("run_core: gate accepts the controller-test runnable input",
          "[run_core]") {
    /*The shared gate mirrors OptimizeIntentController's tested semantics:
     * the runnable input passes and the primary + frame pack through.*/
    const auto result = OptimizerRunControllerCore::EvaluateGate(RunnableInput());
    REQUIRE(result.status == jta::OptimizeIntentController::Status::Ok);
    REQUIRE(result.intent.status == jta::OptimizeIntentController::Status::Ok);
    REQUIRE(result.intent.primary_model_index == 0);
    REQUIRE(result.intent.current_frame == 0);
}

TEST_CASE("run_core: no selection rejects", "[run_core]") {
    auto in = RunnableInput();
    in.selected_model_rows = {};
    in.model_current_index = -1;
    REQUIRE(OptimizerRunControllerCore::EvaluateGate(in).status ==
            jta::OptimizeIntentController::Status::SelectFrameAndModel);
}

TEST_CASE("run_core: pose-dimension mismatch rejects", "[run_core]") {
    auto in = RunnableInput();
    in.pose_frame_count = 2;  // != frame_count
    REQUIRE(OptimizerRunControllerCore::EvaluateGate(in).status ==
            jta::OptimizeIntentController::Status::PoseMatrixDimensionMismatch);
}

TEST_CASE("run_core: no current frame rejects", "[run_core]") {
    auto in = RunnableInput();
    in.current_frame = -1;
    REQUIRE(OptimizerRunControllerCore::EvaluateGate(in).status ==
            jta::OptimizeIntentController::Status::SelectFrameAndModel);
}

TEST_CASE("run_core: BuildGateInput assembles previous == current (H2)",
          "[run_core]") {
    /*H2: the gate Input is assembled with previous == current regardless of
     * any mirrors — a frame jump between sync and run still passes when the
     * selection is valid (the OptimizerBridge behavior, pinned: the gate
     * never sees the raw previous-selection mirrors).*/
    const auto in = RunnableInput();
    const auto packed = OptimizerRunControllerCore::BuildGateInput(in);
    REQUIRE(packed.previous_frame_index == in.current_frame);
    REQUIRE(packed.current_frame == in.current_frame);
    REQUIRE(packed.selected_model_rows == std::vector<int>{0});
    REQUIRE(packed.frame_count == 3);
    REQUIRE(packed.model_current_index == 0);
    REQUIRE(packed.model_count == 2);
    REQUIRE(packed.pose_frame_count == 3);
    REQUIRE(packed.pose_model_count == 2);

    /*A frame jump: current 2 (mirrors would still say 1) — the assembled
     * input passes with previous == current == 2.*/
    auto jumped = RunnableInput();
    jumped.current_frame = 2;
    REQUIRE(OptimizerRunControllerCore::EvaluateGate(jumped).status ==
            jta::OptimizeIntentController::Status::Ok);
    REQUIRE(
        OptimizerRunControllerCore::BuildGateInput(jumped).previous_frame_index ==
        2);
}

TEST_CASE("run_core: state machine transitions", "[run_core]") {
    OptimizerRunControllerCore core;
    REQUIRE(core.state() == OptimizerRunControllerCore::RunState::Idle);
    REQUIRE(core.canStart());
    REQUIRE(!core.running());

    core.onRunStarted();
    REQUIRE(core.state() == OptimizerRunControllerCore::RunState::Running);
    REQUIRE(core.running());
    REQUIRE(!core.canStart());

    core.requestStop();
    REQUIRE(core.state() == OptimizerRunControllerCore::RunState::Stopping);
    REQUIRE(core.running());
    REQUIRE(!core.canStart());

    /*The terminal frame completes a stopped run (re-runnable).*/
    core.onTerminalFrame(false);
    REQUIRE(core.state() == OptimizerRunControllerCore::RunState::Completed);
    REQUIRE(core.canStart());
    REQUIRE(!core.running());
}

TEST_CASE("run_core: terminal frame keeps Error (pinned QML semantics)",
          "[run_core]") {
    /*OptimizerError -> Error; the terminal OptimizedFrame keeps the state
     * at Error (the widgets mapper unlocks on the relay regardless — M8).*/
    OptimizerRunControllerCore core;
    core.onRunStarted();
    core.onOptimizerError();
    REQUIRE(core.state() == OptimizerRunControllerCore::RunState::Error);
    core.onTerminalFrame(true);
    REQUIRE(core.state() == OptimizerRunControllerCore::RunState::Error);
    /*Error is re-runnable (the shell's Start gate adds !threadActive).*/
    REQUIRE(core.canStart());

    /*A fresh run moves out of Error.*/
    core.onRunStarted();
    REQUIRE(core.state() == OptimizerRunControllerCore::RunState::Running);
}

TEST_CASE("run_core: Initialize failure -> Error; epoch bumps per run",
          "[run_core]") {
    OptimizerRunControllerCore core;
    REQUIRE(core.epoch() == 0);
    core.onRunStarted();
    REQUIRE(core.epoch() == 1);
    REQUIRE(core.state() == OptimizerRunControllerCore::RunState::Running);
    core.onInitializeFailed();
    REQUIRE(core.state() == OptimizerRunControllerCore::RunState::Error);
    core.onRunStarted();  // re-run: fresh epoch (stale relays dropped by the
                          // shell via the epoch + sender guard, H1)
    REQUIRE(core.epoch() == 2);
    REQUIRE(core.state() == OptimizerRunControllerCore::RunState::Running);
}

TEST_CASE("run_core: progress mapping mirrors the widgets stage math",
          "[run_core]") {
    /*The UpdateDisplay core: stage/calls/min/progress mirror the widgets
     * onUpdateDisplay arithmetic (mainscreen.cpp:4526-4548) against the
     * cumulative budgets — Trunk while calls < trunk, then Branch n per
     * branch-budget block, then Extra Z-Translation, then Finished at the
     * cumulative cap; progress = calls / cumulative (2200).*/
    const auto b = Budgets();

    OptimizerRunControllerCore core;
    core.refreshProgress(b, 500, -0.25);
    REQUIRE(core.stageText() == "Trunk");
    REQUIRE(core.costCalls() == 500);
    REQUIRE(core.currentMinimum() == Approx(-0.25));
    REQUIRE(core.progress() == Approx(500.0 / 2200.0));

    core.refreshProgress(b, 1001, 0.0);
    REQUIRE(core.stageText() == "Branch 1");
    core.refreshProgress(b, 1501, 0.0);
    REQUIRE(core.stageText() == "Branch 2");
    core.refreshProgress(b, 2001, 0.0);
    REQUIRE(core.stageText() == "Extra Z-Translation");

    core.refreshProgress(b, 2200, -1.0);
    REQUIRE(core.stageText() == "Finished");
    REQUIRE(core.progress() == Approx(1.0));
    core.refreshProgress(b, 4400, -1.0);
    REQUIRE(core.progress() == Approx(1.0));
}

TEST_CASE("run_core: progress with disabled branch/leaf is trunk-only",
          "[run_core]") {
    /*Degenerate config (branch/leaf disabled): the cumulative budget is the
     * trunk budget; the guarded divisor never divides by zero.*/
    OptimizerRunControllerCore::ProgressBudgets b;
    b.trunk_budget = 1000;
    b.enable_branch = false;
    b.enable_leaf = false;

    OptimizerRunControllerCore core;
    core.refreshProgress(b, 999, 0.0);
    REQUIRE(core.stageText() == "Trunk");
    REQUIRE(core.progress() == Approx(999.0 / 1000.0));
    core.refreshProgress(b, 1000, 0.0);
    REQUIRE(core.stageText() == "Finished");
    REQUIRE(core.progress() == Approx(1.0));

    /*Zero cumulative budget -> progress pinned at 0, no divide-by-zero.*/
    OptimizerRunControllerCore::ProgressBudgets zero;
    core.refreshProgress(zero, 50, 0.0);
    REQUIRE(core.progress() == Approx(0.0));
}

TEST_CASE("run_core: seed lifecycle (one-shot, stale guards, clear)",
          "[run_core]") {
    /*The seed API the shell + QML bridge drive (M10a): one-shot apply with
     * stale guards; a rejected run never consumes it (the shell takes it
     * after the gate); a stale-frame/model seed is dropped silently.*/
    OptimizerRunControllerCore core;
    REQUIRE(!core.hasSeedPose());

    core.setSeedPose(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, /*frame=*/0, /*model=*/0);
    REQUIRE(core.hasSeedPose());

    /*Apply for the matching run: popped one-shot.*/
    auto seed = core.takeSeedForRun(/*current_frame=*/0,
                                    /*primary_model_index=*/0,
                                    /*model_count=*/2);
    REQUIRE(seed.applied);
    REQUIRE(seed.frame == 0);
    REQUIRE(seed.model == 0);
    REQUIRE(seed.pose.x == Approx(1.0));
    REQUIRE(seed.pose.za == Approx(6.0));
    REQUIRE(!core.hasSeedPose());

    /*A second take with no seed is a no-op.*/
    REQUIRE(!core.takeSeedForRun(0, 0, 2).applied);

    /*Stale frame: dropped (and cleared — a stale estimate never overrides a
     * different frame's pose).*/
    core.setSeedPose(9.0, 0.0, 0.0, 0.0, 0.0, 0.0, /*frame=*/0, /*model=*/0);
    REQUIRE(!core.takeSeedForRun(/*current_frame=*/1, 0, 2).applied);
    REQUIRE(!core.hasSeedPose());

    /*Stale model (not primary anymore) / out-of-range model: dropped.*/
    core.setSeedPose(9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1);
    REQUIRE(!core.takeSeedForRun(0, /*primary=*/0, 2).applied);
    core.setSeedPose(9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0);
    REQUIRE(!core.takeSeedForRun(0, 0, /*model_count=*/0).applied);

    /*clearSeedPose: the next apply is a no-op; a NEW seed applies.*/
    core.setSeedPose(8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0);
    core.clearSeedPose();
    REQUIRE(!core.takeSeedForRun(0, 0, 2).applied);
    core.setSeedPose(7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0);
    REQUIRE(core.takeSeedForRun(0, 0, 2).applied);
}
