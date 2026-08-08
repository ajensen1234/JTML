// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "core/session_state.h"

#include <algorithm>

namespace jta {

void SessionState::SetModelCount(int count) {
    model_count_ = count < 0 ? 0 : count;
    // Prune any selection points beyond the (possibly) narrowed model list.
    if (!selected_models_.empty() &&
        static_cast<int>(selected_models_.back()) >= model_count_) {
        std::vector<int> kept;
        for (int r : selected_models_)
            if (r < model_count_) kept.push_back(r);
        selected_models_.swap(kept);
    }
}

int SessionState::GetModelCount() const { return model_count_; }

void SessionState::SetSelectedModels(const std::vector<int>& rows) {
    selected_models_ = rows;
    // Keep rows valid (>= 0, < model_count_) and sorted; primary = first.
    std::vector<int> kept;
    for (int r : rows) {
        if (r >= 0 && r < model_count_) kept.push_back(r);
    }
    std::sort(kept.begin(), kept.end());
    selected_models_.swap(kept);
}

const std::vector<int>& SessionState::GetSelectedModels() const {
    return selected_models_;
}

int SessionState::GetPrimaryModelIndex() const {
    return selected_models_.empty() ? -1 : selected_models_.front();
}

bool SessionState::IsSingleSelection() const {
    return selected_models_.size() == 1;
}

void SessionState::SetCurrentFrame(int frame) {
    // Out-of-range frame resolves to -1 (none) rather than clamping, so callers
    // can distinguish 'no frame' from a real index.
    current_frame_ = (frame >= 0 && frame < frame_count_) ? frame : -1;
}

int SessionState::GetCurrentFrame() const { return current_frame_; }

void SessionState::SetFrameCount(int count) {
    frame_count_ = count < 0 ? 0 : count;
    if (current_frame_ != -1 && current_frame_ >= frame_count_)
        current_frame_ = -1;
}

int SessionState::GetFrameCount() const { return frame_count_; }

}  // namespace jta
