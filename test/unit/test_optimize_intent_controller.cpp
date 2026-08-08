// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Optimize-intent controller unit tests (plan U7, R8/R9, AE4). The controller
// owns the widget-free "can I optimize / what do I need" decision that used
// to live inline in MainScreen::LaunchOptimizer. Pure logic; no Qt, no GPU.

#include <catch2/catch_test_macros.hpp>

#include "core/optimize_intent_controller.h"

using jta::OptimizeIntentController;

namespace {
// A runnable input: one selected model, a matched previous/current frame,
// valid counts, and a pose matrix the same size as the loaded lists.
OptimizeIntentController::Input RunnableInput() {
    OptimizeIntentController::Input in;
    in.selected_model_rows = {2};
    in.previous_frame_index = 0;
    in.current_frame = 0;
    in.frame_count = 3;
    in.model_current_index = 2;
    in.model_count = 3;
    in.pose_frame_count = 3;
    in.pose_model_count = 3;
    return in;
}
}  // namespace

TEST_CASE("OptimizeIntentController: runnable common path returns Ok",
          "[optimize_intent]") {
    auto intent = OptimizeIntentController::Evaluate(RunnableInput());
    REQUIRE(intent.status == OptimizeIntentController::Status::Ok);
    REQUIRE(OptimizeIntentController::CanOptimize(RunnableInput()));
}

TEST_CASE("OptimizeIntentController: primary model is first selected row",
          "[optimize_intent]") {
    auto intent = OptimizeIntentController::Evaluate(RunnableInput());
    REQUIRE(intent.primary_model_index == 2);
    REQUIRE(intent.current_frame == 0);
}

TEST_CASE("OptimizeIntentController: empty selection fails cleanly",
          "[optimize_intent]") {
    auto in = RunnableInput();
    in.selected_model_rows = {};
    auto intent = OptimizeIntentController::Evaluate(in);
    REQUIRE(intent.status ==
            OptimizeIntentController::Status::SelectFrameAndModel);
    REQUIRE(!OptimizeIntentController::CanOptimize(in));
}

TEST_CASE("OptimizeIntentController: no current frame fails",
          "[optimize_intent]") {
    auto in = RunnableInput();
    in.previous_frame_index = -1;
    in.current_frame = -1;
    auto intent = OptimizeIntentController::Evaluate(in);
    REQUIRE(intent.status ==
            OptimizeIntentController::Status::SelectFrameAndModel);
}

TEST_CASE("OptimizeIntentController: current frame differs from last-viewed "
          "fails",
          "[optimize_intent]") {
    auto in = RunnableInput();
    in.current_frame = 1;  // differs from previous_frame_index_ == 0
    REQUIRE(OptimizeIntentController::Evaluate(in).status ==
            OptimizeIntentController::Status::SelectFrameAndModel);
}

TEST_CASE("OptimizeIntentController: frame out of range fails",
          "[optimize_intent]") {
    auto in = RunnableInput();
    in.current_frame = 3;  // >= frame_count
    REQUIRE(OptimizeIntentController::Evaluate(in).status ==
            OptimizeIntentController::Status::SelectFrameAndModel);
}

TEST_CASE("OptimizeIntentController: model out of range fails",
          "[optimize_intent]") {
    auto in = RunnableInput();
    in.model_current_index = 5;  // >= model_count
    REQUIRE(OptimizeIntentController::Evaluate(in).status ==
            OptimizeIntentController::Status::SelectFrameAndModel);
}

TEST_CASE("OptimizeIntentController: pose-matrix dimension mismatch fails",
          "[optimize_intent]") {
    auto in = RunnableInput();
    in.pose_frame_count = 2;  // != frame_count
    REQUIRE(OptimizeIntentController::Evaluate(in).status ==
            OptimizeIntentController::Status::PoseMatrixDimensionMismatch);
}

TEST_CASE("OptimizeIntentController: multi-select still Oks (primary = first)",
          "[optimize_intent]") {
    auto in = RunnableInput();
    in.selected_model_rows = {3, 5, 7};
    auto intent = OptimizeIntentController::Evaluate(in);
    REQUIRE(intent.status == OptimizeIntentController::Status::Ok);
    REQUIRE(intent.primary_model_index == 3);
}
