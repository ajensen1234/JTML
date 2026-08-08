// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#pragma once

#include <string>
#include <vector>

// Widget-free "optimize intent" controller (plan U7, R8/R9/AE4).
//
// Owns the pure, headless-testable decision of whether an optimize directive
// can proceed and packages the intent the view hands to the (GPU) optimizer.
// It takes PLAIN VALUES mirroring everything MainScreen::LaunchOptimizer reads
// at the entry gate, so it holds no widgets, no Qt event loop, and no render
// binding -- it is trivially unit-testable headless.
//
// Deliberately NOT bound to OptimizeCoordinator or OptimizerManager: this
// controller only decides + packages. MainScreen keeps the real GPU binding
// (R15 -- production does not rewire to the coordinator stub).
namespace jta {

class OptimizeIntentController {
public:
    // Failure taxonomy mirrors the exact guards in LaunchOptimizer.
    enum class Status {
        Ok = 0,
        SelectFrameAndModel,      // no selected model, or no current frame
        PoseMatrixDimensionMismatch,  // pose matrix size != frames/models loaded
    };

    // Everything the entry gate compares. Plain values, no widgets.
    struct Input {
        std::vector<int> selected_model_rows;  // selected model rows (any order)
        int previous_frame_index = -1;  // last-viewed frame index
        int current_frame = -1;         // current frame row
        int frame_count = 0;            // loaded_frames.size()
        int model_current_index = -1;   // ui.model_list current row
        int model_count = 0;            // loaded_models.size()
        int pose_frame_count = 0;       // model_locations_.GetFrameCount()
        int pose_model_count = 0;       // model_locations_.GetModelCount()
    };

    // The packaged intent for a runnable directive: what the view needs to
    // call OptimizerManager::Initialize (primary model + current frame).
    struct Intent {
        Status status = Status::SelectFrameAndModel;
        int primary_model_index = -1;  // first selected row; -1 if none
        int current_frame = -1;
    };

    // Evaluate the entry-gate predicate. Returns Intent with status Ok and the
    // packed primary model + current frame, or a typed failure reason.
    static Intent Evaluate(const Input& in);

    // True iff Evaluate(in).status == Ok.
    static bool CanOptimize(const Input& in);
};

}  // namespace jta
