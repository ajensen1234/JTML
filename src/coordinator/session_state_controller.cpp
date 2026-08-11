// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "coordinator/session_state_controller.h"

#include <algorithm>
#include <utility>

namespace {

/*Normalize exactly like the domain setters (session_state.cpp): an
 * out-of-range current frame resolves to -1 (none); rows are pruned to
 * [0, model_count) and sorted (no dedup — the domain keeps duplicates).
 * The controller normalizes BEFORE diffing so the comparison is
 * stored-to-stored.*/
int NormalizeFrame(int frame, int frame_count) {
    return (frame >= 0 && frame < frame_count) ? frame : -1;
}

std::vector<int> NormalizeRows(
    const std::vector<int>& rows, int model_count) {
    std::vector<int> kept;
    kept.reserve(rows.size());
    for (int r : rows) {
        if (r >= 0 && r < model_count) {
            kept.push_back(r);
        }
    }
    std::sort(kept.begin(), kept.end());
    return kept;
}

}  // namespace

SessionStateController::SessionStateController(
    jta::SessionState* state, std::function<bool()> run_in_flight,
    std::function<void()> clear_seed, QObject* parent)
    : QObject(parent),
      state_(state),
      run_in_flight_(std::move(run_in_flight)),
      clear_seed_(std::move(clear_seed)) {}

void SessionStateController::UpdateSession(
    int frame_count, int model_count, int current_frame,
    const std::vector<int>& selected_rows) {
    const int normalized_frame = NormalizeFrame(current_frame, frame_count);
    const std::vector<int> normalized_rows =
        NormalizeRows(selected_rows, model_count);

    const bool counts_changed =
        state_->GetFrameCount() != frame_count ||
        state_->GetModelCount() != model_count;
    const bool selection_changed =
        state_->GetCurrentFrame() != normalized_frame ||
        state_->GetSelectedModels() != normalized_rows;

    if (!counts_changed && !selection_changed) {
        /*Diff-based (M9): repeated syncs of identical facts are silent.*/
        return;
    }

    if (selection_changed) {
        /*Capture the PRE-CHANGE selection: the mirrors must still name it
         * while the view's save-last-pose runs (between UpdateSession and
         * CommitSelection) and in the deferred signal payload.*/
        pending_previous_frame_ = state_->GetCurrentFrame();
        pending_previous_rows_ = state_->GetSelectedModels();
        pending_selection_change_ = true;
    }

    /*Write counts first: a shrinking SetFrameCount/SetModelCount can reset
     * an out-of-range current frame / prune out-of-range rows; the
     * normalized values were validated against the incoming counts, so the
     * selection writes restore the intended state.*/
    if (counts_changed) {
        state_->SetFrameCount(frame_count);
        state_->SetModelCount(model_count);
    }
    if (selection_changed) {
        state_->SetCurrentFrame(normalized_frame);
        state_->SetSelectedModels(normalized_rows);
    }
    if (counts_changed) {
        emit datasetChanged();
    }
}

void SessionStateController::CommitSelection() {
    if (!pending_selection_change_) {
        return;
    }
    /*Mirrors advance to the now-current selection (H2 steady state) BEFORE
     * the emission (M9): a consumer observing selectionChanged reads a
     * consistent state (previous == current), and the payload carries the
     * pre-change mirrors for save-last-pose consumers.*/
    state_->SetPreviousFrame(state_->GetCurrentFrame());
    state_->SetPreviousModelRows(state_->GetSelectedModels());

    const int previous_frame = pending_previous_frame_;
    const std::vector<int> previous_rows = pending_previous_rows_;
    pending_selection_change_ = false;
    pending_previous_frame_ = -1;
    pending_previous_rows_.clear();

    emit selectionChanged(
        state_->GetCurrentFrame(),
        state_->GetPrimaryModelIndex(),
        state_->GetSelectedModels(),
        previous_frame,
        previous_rows);
}

void SessionStateController::ResetForDatasetClear() {
    /*Cross-dataset hygiene (H5/M10b): the mirrors must never name the wiped
     * dataset (a bogus save-last-pose write), and the optimizer's pending
     * seed must not leak into the next dataset's runs.*/
    state_->SetPreviousFrame(-1);
    state_->SetPreviousModelRows({});
    if (clear_seed_) {
        clear_seed_();
    }
    /*Drop any deferred selection emission — the pre-change capture names
     * the wiped dataset.*/
    pending_selection_change_ = false;
    pending_previous_frame_ = -1;
    pending_previous_rows_.clear();
    emit datasetChanged();
}

bool SessionStateController::runInFlight() const {
    return run_in_flight_ ? run_in_flight_() : false;
}
