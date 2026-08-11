// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "DelegateSelection.h"

void DelegateSelection::SetCurrentFrame(int index) {
    current_frame_ = index;
}

int DelegateSelection::GetCurrentFrame() const {
    return current_frame_;
}

void DelegateSelection::SetModelSelected(int row, bool selected) {
    if (selected) {
        selected_model_rows_.insert(row);
    } else {
        selected_model_rows_.erase(row);
    }
}

void DelegateSelection::ToggleModel(int row) {
    if (!selected_model_rows_.erase(row)) {
        selected_model_rows_.insert(row);
    }
}

void DelegateSelection::ClearModelSelection() {
    selected_model_rows_.clear();
}

bool DelegateSelection::IsModelSelected(int row) const {
    return selected_model_rows_.count(row) > 0;
}

int DelegateSelection::GetSelectedModelCount() const {
    return static_cast<int>(selected_model_rows_.size());
}

std::vector<int> DelegateSelection::GetSelectedModelRows() const {
    return {selected_model_rows_.begin(), selected_model_rows_.end()};
}

int DelegateSelection::GetPrimaryModelIndex() const {
    return selected_model_rows_.empty() ? -1 : *selected_model_rows_.begin();
}

bool DelegateSelection::HasModelSelection() const {
    return !selected_model_rows_.empty();
}
