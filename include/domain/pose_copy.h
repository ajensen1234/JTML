// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#pragma once

// Pure pose-copy seam for the MainScreen pose slots (plan 004 U4, R8/R13).
// Owns the copy-prev/next index math, the +-1 boundary fallback rule, the
// primary-vs-current index split, and the shared selection guard. No
// services/vtk/Qt includes (jtml_domain purity boundary): the view keeps the
// LocationStorage GetPose/SavePose calls and passes the plan's raw indices.
//
// R13 preservation facts this seam is built to keep byte-identical:
//  1. Copy_Previous / Copy_Next READ GetPose(row +- 1, PRIMARY model) but
//     WRITE SavePose(row, CURRENT model row). The split is explicit in
//     CopyPlan (read_model vs write_model, read_frame vs write_frame) and is
//     never aligned by the seam.
//  2. GetPose(-1, ...) / GetPose(count(), ...) fall through to the no-image
//     default pose (a VALID pose -- the model's initial pose) and the copy
//     overwrites frame 0 / the last frame with it. The caller must pass the
//     RAW read_frame to GetPose and store the returned pose verbatim:
//     clamping the index would substitute a real frame pose and change
//     behavior (boundary_fallback marks the case so tests can pin it).
//  3. Copying does NO A<->B camera conversion: the plan carries indices only,
//     the pose read at (read_frame, read_model) is stored un-converted
//     (calibration stays view-side, out of this seam).
//  4. The multi-model guard tests the RADIO (multiple_model_radio_button
//     ->isChecked()), not the selection mode; the frame/model selection check
//     runs first.
namespace jta {
namespace pose_copy {

// Shared guard decision for the pose save/load/copy slots. The two checks run
// in the original order: the frame/model selection check first, then the
// multi-model radio check.
enum class SelectionGuard {
    Ok,
    NoFrameOrModel,  // current frame row < 0 || no model rows selected
    MultiModelMode,  // the multiple-model radio is checked
};

SelectionGuard CheckSelection(
    int current_frame_row,
    int selected_model_count,
    bool multi_model_radio_checked);

// Copy plan for the Copy_Previous / Copy_Next slots: READ at
// (read_frame, read_model), WRITE at (write_frame, write_model).
struct CopyPlan {
    int read_frame;          // current_frame_row +- 1 (may be -1 / frame_count)
    int read_model;          // primary (first-selected) model row
    int write_frame;         // current_frame_row
    int write_model;         // CURRENT model row (R13 split, not the primary)
    bool boundary_fallback;  // read_frame outside [0, frame_count)
};

// Copy-previous plan: read_frame = current_frame_row - 1.
CopyPlan PreviousPose(
    int current_frame_row,
    int current_model_row,
    int primary_model_row,
    int frame_count);

// Copy-next plan: read_frame = current_frame_row + 1.
CopyPlan NextPose(
    int current_frame_row,
    int current_model_row,
    int primary_model_row,
    int frame_count);

}  // namespace pose_copy
}  // namespace jta
