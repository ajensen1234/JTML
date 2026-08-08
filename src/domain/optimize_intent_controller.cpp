// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "domain/optimize_intent_controller.h"

#include <algorithm>

namespace jta {

OptimizeIntentController::Intent OptimizeIntentController::Evaluate(
    const Input& in) {
    Intent intent;

    // Selection/frame preconditions (mirrors LaunchOptimizer's first guard).
    if (in.selected_model_rows.empty() || in.previous_frame_index < 0 ||
        in.current_frame != in.previous_frame_index ||
        in.current_frame >= in.frame_count ||
        in.model_current_index >= in.model_count) {
        intent.status = Status::SelectFrameAndModel;
        return intent;
    }

    // Pose-matrix dimension precondition (mirrors the second guard).
    if (in.pose_frame_count != in.frame_count ||
        in.pose_model_count != in.model_count) {
        intent.status = Status::PoseMatrixDimensionMismatch;
        return intent;
    }

    // Runnable: primary model = first selected row (same rule as the view's
    // selected[0].row() / session_state_.GetPrimaryModelIndex()).
    intent.status = Status::Ok;
    intent.primary_model_index = in.selected_model_rows.front();
    intent.current_frame = in.current_frame;
    return intent;
}

bool OptimizeIntentController::CanOptimize(const Input& in) {
    return Evaluate(in).status == Status::Ok;
}

}  // namespace jta
