// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#pragma once

#include <vector>

// App-state / session orchestration layer (plan U7, R8/E11: "app-state/command
// orchestration" out of the MainScreen view).
//
// SessionState owns the pure, widget-free session facts the view and the
// command/coordinator layers both depend on: the model list, which models are
// selected, the primary model identity, current frame navigation, and the
// previous-selection mirrors that feed save-last-pose. It holds NO widgets,
// NO Qt event loop, and NO render binding, so it is unit-testable headless.
// MainScreen keeps it current from widget events; the rest of the app reads it
// instead of reaching into the UI.
//
// Deliberately NOT an observable ViewModel: this is a Qt Widgets app with no
// data-binding framework, so this is a plain state holder (the plan's lesson,
// R12: no over-engineering / no binding framework). Signal/notification, where
// needed, lives in the (Qt-based) coordinator layer (e.g. OptimizeCoordinator).
namespace jta {

class SessionState {
public:
    SessionState() = default;

    // ---- Model list -------------------------------------------------------
    void SetModelCount(int count);
    int GetModelCount() const;

    // ---- Selection --------------------------------------------------------
    // Set the selected model rows (sorted). The primary model is the first
    // selected row, mirroring the original MainScreen rule (selected[0].row()).
    // Pass an empty vector to clear the selection.
    void SetSelectedModels(const std::vector<int>& rows);
    const std::vector<int>& GetSelectedModels() const;

    // First selected model index, or -1 if nothing is selected.
    int GetPrimaryModelIndex() const;
    // True iff exactly one model is selected.
    bool IsSingleSelection() const;

    // ---- Frame navigation -------------------------------------------------
    void SetCurrentFrame(int frame);
    int GetCurrentFrame() const;
    void SetFrameCount(int count);
    int GetFrameCount() const;

    // ---- Previous-selection mirror (save-last-pose) -----------------------
    // The last-selected frame/model set — equals the current selection in
    // steady state; consumed by save-last-pose only, never fed to the
    // optimizer gate (the run controller always builds gate input with
    // previous == current). Mirrors the widgets' previous_frame_index_ /
    // previous_model_indices_ bookkeeping. Purely additive: no consumer in
    // this unit; the session-state controller writes both mirrors together
    // and resets them on dataset clear.
    //
    // SetPreviousFrame: negative resolves to -1 (none), like SetCurrentFrame.
    // The value is not validated against frame_count_ — the mirror is
    // "last-selected", and it may name a frame of the pre-change dataset
    // state until the controller resets it.
    void SetPreviousFrame(int frame);
    int GetPreviousFrame() const;

    // SetPreviousModelRows: same rule as SetSelectedModels — rows are pruned
    // to [0, model_count_) and sorted on write; pass an empty vector to
    // clear.
    void SetPreviousModelRows(const std::vector<int>& rows);
    const std::vector<int>& GetPreviousModelRows() const;

    // True iff a previous frame AND at least one previous row are recorded.
    bool HasPreviousSelection() const;

private:
    int model_count_ = 0;
    int frame_count_ = 0;
    int current_frame_ = -1;
    std::vector<int> selected_models_;
    int previous_frame_ = -1;
    std::vector<int> previous_model_rows_;
};

}  // namespace jta
