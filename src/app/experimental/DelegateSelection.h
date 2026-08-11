// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U4: the delegate-based selection contract state. No
// QItemSelectionModel anywhere (the plan's deliberate non-reuse — the QML
// delegates drive this directly):
//  - frame: a single current index (-1 = none), the mirror of the QML
//    ListView currentIndex;
//  - models: a multi-select row set; the primary model is the first
//    selected row (ascending — the widgets selected[0].row() rule).
//
// Pure state, Qt-free: headless-testable (the U4 contract pins live in
// test/unit/experimental_selection_test.cpp). The StudyBridge guards the
// row bounds before touching this (out-of-range rows are ignored there).

#pragma once

#include <set>
#include <vector>

class DelegateSelection {
public:
    // --- Frame -----------------------------------------------------------
    void SetCurrentFrame(int index);  // -1 = none
    int GetCurrentFrame() const;

    // --- Models ----------------------------------------------------------
    void SetModelSelected(int row, bool selected);
    void ToggleModel(int row);
    void ClearModelSelection();
    bool IsModelSelected(int row) const;
    int GetSelectedModelCount() const;
    std::vector<int> GetSelectedModelRows() const;  // ascending
    int GetPrimaryModelIndex() const;               // -1 when nothing selected
    bool HasModelSelection() const;

private:
    int current_frame_ = -1;
    std::set<int> selected_model_rows_;
};
