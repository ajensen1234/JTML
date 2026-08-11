// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U6 unit tests: SessionStateController sync/diff logic (R5/R6,
// AE2). Catch2 over the same QObject shell (Qt6::Core only — direct
// lambda-connect counting, no event loop, no QtTest): the normalization
// rules (out-of-range current frame -> -1, rows pruned + sorted), the
// diff-vs-stored semantics (repeated identical syncs are silent), the
// two-phase mirror contract (UpdateSession leaves the pre-change selection
// readable for the view's save-last-pose; CommitSelection advances the
// mirrors to the now-current selection, H2), the count-shrink pruning
// interplay, ResetForDatasetClear (mirrors reset + injected seed-clear),
// and the runInFlight probe (M7). The signal behavior itself is pinned by
// the lifecycle test (test/lifecycle/session_state_controller_test.cpp).

#include <catch2/catch_test_macros.hpp>

#include <QObject>

#include "coordinator/session_state_controller.h"

using jta::SessionState;

TEST_CASE(
    "SessionStateController: sync writes the four facts; mirrors advance on "
    "commit",
    "[session_state_controller]") {
    SessionState state;
    SessionStateController ctrl(&state);

    ctrl.UpdateSession(3, 5, 2, {3, 1});
    /*Facts written (rows normalized: pruned + sorted).*/
    REQUIRE(state.GetFrameCount() == 3);
    REQUIRE(state.GetModelCount() == 5);
    REQUIRE(state.GetCurrentFrame() == 2);
    REQUIRE(state.GetSelectedModels() == std::vector<int>({1, 3}));
    /*Mirrors NOT touched by the sync — the view's save-last-pose runs
     * between the two phases and must still read the pre-change selection.*/
    REQUIRE(state.GetPreviousFrame() == -1);
    REQUIRE(state.GetPreviousModelRows().empty());

    ctrl.CommitSelection();
    /*Mirrors advance to the now-current selection (H2 steady state).*/
    REQUIRE(state.GetPreviousFrame() == 2);
    REQUIRE(state.GetPreviousModelRows() == std::vector<int>({1, 3}));
    REQUIRE(state.HasPreviousSelection());
}

TEST_CASE(
    "SessionStateController: the two-phase contract keeps the pre-change "
    "selection readable",
    "[session_state_controller]") {
    SessionState state;
    SessionStateController ctrl(&state);

    ctrl.UpdateSession(10, 5, 3, {0, 1});
    ctrl.CommitSelection();

    /*Between UpdateSession and CommitSelection the mirrors still name the
     * pre-change selection (the widgets handler order: sync -> SaveLastPose
     * -> mirrors).*/
    ctrl.UpdateSession(10, 5, 7, {2});
    REQUIRE(state.GetPreviousFrame() == 3);
    REQUIRE(state.GetPreviousModelRows() == std::vector<int>({0, 1}));

    ctrl.CommitSelection();
    REQUIRE(state.GetPreviousFrame() == 7);
    REQUIRE(state.GetPreviousModelRows() == std::vector<int>({2}));
}

TEST_CASE(
    "SessionStateController: out-of-range frame and rows normalize before "
    "the diff",
    "[session_state_controller]") {
    SessionState state;
    SessionStateController ctrl(&state);

    ctrl.UpdateSession(10, 3, 99, {0, 2, 7, -1});
    REQUIRE(state.GetCurrentFrame() == -1);  // out-of-range -> none
    REQUIRE(state.GetSelectedModels() == std::vector<int>({0, 2}));

    /*Identical raw-but-normalizing input is a no-change (the diff compares
     * normalized-to-normalized).*/
    ctrl.UpdateSession(10, 3, -1, {2, 0, 7});
    REQUIRE(state.GetCurrentFrame() == -1);
    REQUIRE(state.GetSelectedModels() == std::vector<int>({0, 2}));
}

TEST_CASE(
    "SessionStateController: repeated identical syncs are silent (diff "
    "emission)",
    "[session_state_controller]") {
    SessionState state;
    SessionStateController ctrl(&state);
    int dataset_changes = 0;
    int selection_changes = 0;
    QObject::connect(
        &ctrl, &SessionStateController::datasetChanged,
        [&] { ++dataset_changes; });
    QObject::connect(
        &ctrl, &SessionStateController::selectionChanged,
        [&](int, int, const std::vector<int>&, int,
            const std::vector<int>&) { ++selection_changes; });

    ctrl.UpdateSession(10, 5, 3, {0, 1});
    REQUIRE(dataset_changes == 1);
    REQUIRE(selection_changes == 0);  // deferred to CommitSelection (M9)
    ctrl.CommitSelection();
    REQUIRE(selection_changes == 1);

    /*Repeated syncs: no writes, no spam.*/
    ctrl.UpdateSession(10, 5, 3, {0, 1});
    ctrl.CommitSelection();
    ctrl.UpdateSession(10, 5, 3, {0, 1});
    REQUIRE(dataset_changes == 1);
    REQUIRE(selection_changes == 1);
}

TEST_CASE(
    "SessionStateController: shrinking counts prune selection and invalidate "
    "an out-of-range current frame",
    "[session_state_controller]") {
    SessionState state;
    SessionStateController ctrl(&state);

    ctrl.UpdateSession(10, 5, 4, {0, 1, 4});
    ctrl.CommitSelection();

    /*Frame count shrinks below the current frame: the normalized diff sees
     * the current as a change and resolves it to -1 (like the domain
     * setter).*/
    ctrl.UpdateSession(3, 5, 4, {0, 1});
    REQUIRE(state.GetFrameCount() == 3);
    REQUIRE(state.GetCurrentFrame() == -1);

    /*Model count shrinks: rows pruned to the new range.*/
    ctrl.UpdateSession(3, 2, -1, {0, 1});
    REQUIRE(state.GetModelCount() == 2);
    REQUIRE(state.GetSelectedModels() == std::vector<int>({0, 1}));

    /*Out-of-range rows against the narrowed list normalize away.*/
    ctrl.UpdateSession(3, 2, -1, {0, 1, 9});
    REQUIRE(state.GetSelectedModels() == std::vector<int>({0, 1}));
}

TEST_CASE(
    "SessionStateController: ResetForDatasetClear resets mirrors and invokes "
    "the seed-clear",
    "[session_state_controller]") {
    SessionState state;
    int seed_clears = 0;
    SessionStateController ctrl(&state, {}, [&] { ++seed_clears; });

    ctrl.UpdateSession(10, 5, 3, {0, 1});
    ctrl.CommitSelection();
    REQUIRE(state.HasPreviousSelection());

    ctrl.ResetForDatasetClear();
    REQUIRE(state.GetPreviousFrame() == -1);
    REQUIRE(state.GetPreviousModelRows().empty());
    REQUIRE_FALSE(state.HasPreviousSelection());
    /*H5/M10b: the optimizer's pending seed is dropped (via the callback the
     * composition root wires).*/
    REQUIRE(seed_clears == 1);
}

TEST_CASE(
    "SessionStateController: runInFlight reflects the injected probe",
    "[session_state_controller]") {
    SessionState state;
    bool in_flight = false;
    SessionStateController ctrl(&state, [&] { return in_flight; });

    REQUIRE_FALSE(ctrl.runInFlight());
    in_flight = true;
    REQUIRE(ctrl.runInFlight());
    in_flight = false;
    REQUIRE_FALSE(ctrl.runInFlight());
}

TEST_CASE(
    "SessionStateController: sessionState() exposes the wrapped state",
    "[session_state_controller]") {
    SessionState state;
    SessionStateController ctrl(&state);

    ctrl.UpdateSession(4, 2, 1, {1, 0});
    ctrl.CommitSelection();
    REQUIRE(ctrl.sessionState().GetCurrentFrame() == 1);
    REQUIRE(ctrl.sessionState().GetPrimaryModelIndex() == 0);
    REQUIRE(ctrl.sessionState().GetPreviousFrame() == 1);
    REQUIRE(
        ctrl.sessionState().GetPreviousModelRows() ==
        std::vector<int>({0, 1}));
}
