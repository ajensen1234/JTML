// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U5: OptimizerRunControllerCore implementation — see the header
// for the contract. Qt/GPU-free: no QObject, no event loop, no CUDA/torch.

#include "coordinator/optimizer_run_controller_core.h"

#include <algorithm>
#include <cmath>

namespace jta {

/*---- Gate (H2) ----*/

OptimizeIntentController::Input OptimizerRunControllerCore::BuildGateInput(
    const GateInput& in) {
    OptimizeIntentController::Input out;
    out.selected_model_rows = in.selected_model_rows;
    /*H2: previous == current — the gate never sees the raw session mirrors
     * (they feed save-last-pose only); this is exactly what the QML bridge's
     * buildGateInput did (previous_frame_index = currentFrame).*/
    out.previous_frame_index = in.current_frame;
    out.current_frame = in.current_frame;
    out.frame_count = in.frame_count;
    out.model_current_index = in.model_current_index;
    out.model_count = in.model_count;
    out.pose_frame_count = in.pose_frame_count;
    out.pose_model_count = in.pose_model_count;
    return out;
}

OptimizerRunControllerCore::GateResult OptimizerRunControllerCore::EvaluateGate(
    const GateInput& in) {
    GateResult result;
    result.intent = OptimizeIntentController::Evaluate(BuildGateInput(in));
    result.status = result.intent.status;
    return result;
}

/*---- Progress (oracle seam, M12) ----*/

std::string OptimizerRunControllerCore::StageLabel(
    const ProgressBudgets& b,
    int calls) {
    const int branch_budget =
        b.enable_branch ? std::max(1, b.branch_budget) : 1;
    const int branch_total =
        b.enable_branch ? b.number_branches * b.branch_budget : 0;
    if (calls < b.trunk_budget) {
        return "Trunk";
    }
    if (calls < b.trunk_budget + branch_total) {
        return "Branch " +
            std::to_string((calls - b.trunk_budget) / branch_budget + 1);
    }
    if (calls <
        b.trunk_budget + branch_total + (b.enable_leaf ? b.leaf_budget : 0)) {
        return "Extra Z-Translation";
    }
    return "Finished";
}

void OptimizerRunControllerCore::refreshProgress(
    const ProgressBudgets& b,
    int calls,
    double minimum) {
    cost_calls_ = calls;
    current_minimum_ = minimum;
    const int cumulative = b.trunk_budget +
        (b.enable_branch ? b.number_branches * b.branch_budget : 0) +
        (b.enable_leaf ? b.leaf_budget : 0);
    stage_text_ = StageLabel(b, calls);
    progress_ =
        cumulative > 0 ? std::min(1.0, calls / double(cumulative)) : 0.0;
}

/*---- Seed lifecycle (M10a) ----*/

void OptimizerRunControllerCore::setSeedPose(
    double x,
    double y,
    double z,
    double xa,
    double ya,
    double za,
    int frame,
    int model) {
    seed_pose_ = Point6D(x, y, z, xa, ya, za);
    seed_frame_ = frame;
    seed_model_ = model;
    has_seed_pose_ = true;
}

OptimizerRunControllerCore::AppliedSeed
OptimizerRunControllerCore::takeSeedForRun(
    int current_frame,
    int primary_model_index,
    int model_count) {
    AppliedSeed result;
    if (!has_seed_pose_) {
        return result;
    }
    /*One-shot + stale guards (OptimizerBridge.cpp:276-291 preserved): the
     * seed applies only when the run's frame is still the seeded frame and
     * the seeded model is still the primary selection; any other state drops
     * the seed silently.*/
    if (current_frame != seed_frame_ || primary_model_index != seed_model_ ||
        seed_model_ < 0 || seed_model_ >= model_count) {
        clearSeedPose();
        return result;
    }
    result.applied = true;
    result.pose = seed_pose_;
    result.frame = seed_frame_;
    result.model = seed_model_;
    has_seed_pose_ = false;
    return result;
}

}  // namespace jta
