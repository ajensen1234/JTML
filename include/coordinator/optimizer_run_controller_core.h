// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U5: OptimizerRunControllerCore — the Qt/GPU-free core of the
// shared optimizer-run controller (R7, R13, R15, R16; F1; AE1). Both
// front-ends drive the same run logic through the OptimizerRunController
// QObject shell; this core holds the decision/state/progress/seed logic that
// must be headless-testable without Qt, VTK, CUDA, or torch:
//
//  - the entry gate: Input assembly with previous == current (H2 — the
//    mirrors mean "last-selected, == current in steady state"; the gate
//    never sees the raw mirrors) + evaluation via the existing
//    jta::OptimizeIntentController;
//  - the 5-state run machine (Idle -> Running -> Stopping -> Completed /
//    Error; Error is not a dead-end for the widgets mapper — the terminal
//    OptimizedFrame relay drives the view-side unlock, M8) + the epoch
//    counter (H1 — relays from stale epochs are dropped by the shell);
//  - the progress/stage mapping (widgets onUpdateDisplay level math, calls
//    vs cumulative budget — the oracle seam, M12);
//  - the seed lifecycle (M10a — apply after the gate, one-shot stale
//    guards, snapshot restore on Initialize failure is shell-side).
//
// The shell owns the OptimizerRunDriver seam, the thread lifecycle, the
// destructor contract, and the by-value relay signals (QTBUG-2842).

#ifndef OPTIMIZER_RUN_CONTROLLER_CORE_H
#define OPTIMIZER_RUN_CONTROLLER_CORE_H

#include <string>
#include <vector>

#include "domain/data_structures_6D.h"
#include "domain/optimize_intent_controller.h"

namespace jta {

class OptimizerRunControllerCore {
public:
    /*Run-state machine (plan 005 U6 review fix + plan 006 M8): idle ->
     * running -> stopping -> completed/error. QML compares runState against
     * these values; the widgets mapper unlocks on the terminal-frame relay
     * (never purely on state).*/
    enum class RunState {
        Idle = 0,
        Running = 1,
        Stopping = 2,
        Completed = 3,
        Error = 4
    };

    /*Severity-carrying message channel (L14): the widgets preserves its
     * box-type distinctions (critical/info/warning); QML ignores severity
     * (single Dialog).*/
    enum class Severity { Info = 0, Warning = 1, Critical = 2 };

    /*Typed run directive (the widgets' strings preserved verbatim by the
     * shell's mapping; QML v1 = Single).*/
    enum class Directive {
        Single = 0,
        All = 1,
        Each = 2,
        From = 3,
        Backward = 4,
        SymTrap = 5
    };

    /*Everything the entry gate compares — plain values mirroring everything
     * LaunchOptimizer reads at the gate (no widgets, no mirrors: previous
     * is derived, H2).*/
    struct GateInput {
        std::vector<int> selected_model_rows;  // selected model rows (any order)
        int current_frame = -1;                // current frame row
        int frame_count = 0;                   // loaded_frames.size()
        int model_current_index = -1;          // ui.model_list current row
        int model_count = 0;                   // loaded_models.size()
        int pose_frame_count = 0;              // model_locations_.GetFrameCount()
        int pose_model_count = 0;              // model_locations_.GetModelCount()
    };

    struct GateResult {
        OptimizeIntentController::Status status =
            OptimizeIntentController::Status::SelectFrameAndModel;
        OptimizeIntentController::Intent intent;
    };

    /*The cumulative-budget arithmetic the progress mapping + the oracle
     * seam consume (from OptimizerSettings; plain values keep the core
     * Qt-free).*/
    struct ProgressBudgets {
        int trunk_budget = 0;
        int branch_budget = 0;
        int number_branches = 0;
        bool enable_branch = false;
        int leaf_budget = 0;
        bool enable_leaf = false;
    };

    /*The outcome of takeSeedForRun: the pending seed popped for the run
     * (one-shot). The shell applies it to the storage (snapshotting first
     * for the M10a restore) and maps the scene write. `applied == false`
     * means no seed or a stale one (dropped silently).*/
    struct AppliedSeed {
        bool applied = false;
        Point6D pose;
        int frame = -1;
        int model = -1;
    };

    /*---- Gate (H2) --------------------------------------------------------
     * The Input is assembled with previous == current regardless of the
     * session mirrors (exactly as OptimizerBridge did: buildGateInput sets
     * previous_frame_index = currentFrame), so a frame jump between sync
     * and run still passes when the selection is valid.*/
    static OptimizeIntentController::Input BuildGateInput(const GateInput& in);
    static GateResult EvaluateGate(const GateInput& in);

    /*---- Run-state machine (M8) ------------------------------------------*/
    RunState state() const { return state_; }
    /*True while a run is in flight (running or stopping).*/
    bool running() const {
        return state_ == RunState::Running || state_ == RunState::Stopping;
    }
    /*True in idle/completed/error — the run button's enabled binding. The
     * shell's Start gate adds the !threadActive condition (H1/M6).*/
    bool canStart() const {
        return state_ == RunState::Idle || state_ == RunState::Completed ||
               state_ == RunState::Error;
    }
    /*Run accepted: Running + a fresh epoch (stale relays are dropped by the
     * shell via the epoch + sender guard).*/
    void onRunStarted() {
        ++epoch_;
        state_ = RunState::Running;
    }
    void requestStop() {
        if (state_ == RunState::Running) {
            state_ = RunState::Stopping;
        }
    }
    void onInitializeFailed() { state_ = RunState::Error; }
    void onOptimizerError() { state_ = RunState::Error; }
    /*Terminal OptimizedFrame: Completed unless an OptimizerError already
     * moved the run to Error (pinned QML semantics — applyOptimizedFrame
     * keeps Error; the widgets mapper unlocks on the relay regardless, M8).*/
    void onTerminalFrame(bool /*error_occurred*/) {
        if (state_ != RunState::Error) {
            state_ = RunState::Completed;
        }
    }

    /*---- Epoch (H1) ------------------------------------------------------*/
    int epoch() const { return epoch_; }

    /*---- Progress (oracle seam, M12) --------------------------------------*/
    /*Widgets onUpdateDisplay level classification (mainscreen.cpp:4526-4548)
     * with the QML bridge's degenerate-settings guard (identical labels; the
     * widgets divides by branch_budget unguarded — the guard only prevents a
     * divide-by-zero on a disabled/zero branch config).*/
    static std::string StageLabel(const ProgressBudgets& b, int calls);
    void refreshProgress(const ProgressBudgets& b, int calls, double minimum);
    std::string stageText() const { return stage_text_; }
    int costCalls() const { return cost_calls_; }
    double currentMinimum() const { return current_minimum_; }
    /*0..1 progress from calls vs the cumulative budget.*/
    double progress() const { return progress_; }

    /*---- Seed lifecycle (M10a) --------------------------------------------*/
    /*One-shot starting-pose seed (R8 — the ML estimate seeds the optimizer).
     * The seed is estimated for (frame, model); run() takes it AFTER the
     * gate so a rejected run never consumes it (the shell applies it before
     * Initialize so the estimate wins over the SaveLastPose mirror).*/
    void setSeedPose(
        double x, double y, double z, double xa, double ya, double za,
        int frame, int model);
    void clearSeedPose() { has_seed_pose_ = false; }
    bool hasSeedPose() const { return has_seed_pose_; }
    /*Pop the pending seed for a run at (current_frame, primary_model_index)
     * with model_count models. Stale guards preserved (OptimizerBridge.cpp:
     * 276-291): the seed applies only when the run's frame is still the
     * seeded frame and the seeded model is still the primary selection;
     * otherwise it is dropped silently — a stale-frame estimate must never
     * override a different frame's pose.*/
    AppliedSeed takeSeedForRun(
        int current_frame, int primary_model_index, int model_count);

private:
    RunState state_ = RunState::Idle;
    int epoch_ = 0;
    std::string stage_text_ = "Idle";
    int cost_calls_ = 0;
    double current_minimum_ = 0.0;
    double progress_ = 0.0;
    bool has_seed_pose_ = false;
    Point6D seed_pose_;
    int seed_frame_ = -1;
    int seed_model_ = -1;
};

}  // namespace jta

#endif /* OPTIMIZER_RUN_CONTROLLER_CORE_H */
