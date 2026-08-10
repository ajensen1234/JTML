// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "domain/pose_copy.h"

namespace jta {
namespace pose_copy {

SelectionGuard CheckSelection(int current_frame_row, int selected_model_count,
                              bool multi_model_radio_checked) {
    // R13: the frame/model selection check comes first, exactly like the
    // original slots (a missing selection errors out even when the
    // multiple-model radio is checked).
    if (current_frame_row < 0 || selected_model_count == 0) {
        return SelectionGuard::NoFrameOrModel;
    }
    // R13: the multi-model guard tests the RADIO, not the selection mode.
    if (multi_model_radio_checked) {
        return SelectionGuard::MultiModelMode;
    }
    return SelectionGuard::Ok;
}

CopyPlan PreviousPose(int current_frame_row, int current_model_row,
                      int primary_model_row, int frame_count) {
    const int read_frame = current_frame_row - 1;
    return CopyPlan{read_frame,
                    primary_model_row,
                    current_frame_row,
                    current_model_row,
                    read_frame < 0 || read_frame >= frame_count};
}

CopyPlan NextPose(int current_frame_row, int current_model_row,
                  int primary_model_row, int frame_count) {
    const int read_frame = current_frame_row + 1;
    return CopyPlan{read_frame,
                    primary_model_row,
                    current_frame_row,
                    current_model_row,
                    read_frame < 0 || read_frame >= frame_count};
}

}  // namespace pose_copy
}  // namespace jta
