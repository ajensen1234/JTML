// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// App-state / session orchestration unit tests (plan U7, R8/R9):
// selecting a model/frame updates session state with no widget, matching the
// original MainScreen rule that the primary model is the first selected row.

#include <catch2/catch_test_macros.hpp>

#include "domain/session_state.h"

using jta::SessionState;

TEST_CASE("SessionState: model list count", "[session_state]") {
    SessionState s;
    REQUIRE(s.GetModelCount() == 0);
    s.SetModelCount(5);
    REQUIRE(s.GetModelCount() == 5);
    s.SetModelCount(-1);  // negative -> 0
    REQUIRE(s.GetModelCount() == 0);
}

TEST_CASE("SessionState: primary model is the first selected row",
          "[session_state]") {
    SessionState s;
    s.SetModelCount(5);
    REQUIRE(s.GetPrimaryModelIndex() == -1);  // nothing selected yet

    s.SetSelectedModels({2, 4});
    REQUIRE(s.GetPrimaryModelIndex() == 2);
    REQUIRE_FALSE(s.IsSingleSelection());

    s.SetSelectedModels({4, 1, 3});  // unsorted input
    REQUIRE(s.GetPrimaryModelIndex() == 1);  // primary = first (sorted)
    REQUIRE(s.GetSelectedModels() == std::vector<int>({1, 3, 4}));

    s.SetSelectedModels({3});  // single selection
    REQUIRE(s.IsSingleSelection());
    REQUIRE(s.GetPrimaryModelIndex() == 3);

    s.SetSelectedModels({});  // clear
    REQUIRE(s.GetPrimaryModelIndex() == -1);
    REQUIRE_FALSE(s.IsSingleSelection());
}

TEST_CASE("SessionState: out-of-range selection is pruned", "[session_state]") {
    SessionState s;
    s.SetModelCount(3);
    s.SetSelectedModels({0, 2, 7, -1});  // 7 and -1 are invalid
    REQUIRE(s.GetSelectedModels() == std::vector<int>({0, 2}));
    REQUIRE(s.GetPrimaryModelIndex() == 0);

    s.SetModelCount(1);  // narrowing prunes the now-out-of-range row 2
    REQUIRE(s.GetSelectedModels() == std::vector<int>({0}));
}

TEST_CASE("SessionState: frame navigation (no widget)", "[session_state]") {
    SessionState s;
    s.SetFrameCount(10);
    REQUIRE(s.GetFrameCount() == 10);

    REQUIRE(s.GetCurrentFrame() == -1);  // none until set
    s.SetCurrentFrame(3);
    REQUIRE(s.GetCurrentFrame() == 3);

    s.SetCurrentFrame(99);   // out of range -> none
    REQUIRE(s.GetCurrentFrame() == -1);
    s.SetCurrentFrame(-1);
    REQUIRE(s.GetCurrentFrame() == -1);

    s.SetCurrentFrame(5);
    s.SetFrameCount(3);      // narrowing clears a now-invalid frame
    REQUIRE(s.GetCurrentFrame() == -1);
}

TEST_CASE("SessionState: model and frame counts are independent of selection",
          "[session_state]") {
    SessionState s;
    s.SetModelCount(4);
    s.SetFrameCount(7);
    s.SetSelectedModels({1});
    s.SetCurrentFrame(6);
    REQUIRE(s.GetModelCount() == 4);
    REQUIRE(s.GetFrameCount() == 7);
    REQUIRE(s.GetPrimaryModelIndex() == 1);
    REQUIRE(s.GetCurrentFrame() == 6);
}

TEST_CASE("SessionState: previous-selection mirrors default to none",
          "[session_state]") {
    SessionState s;
    REQUIRE(s.GetPreviousFrame() == -1);
    REQUIRE(s.GetPreviousModelRows().empty());
    REQUIRE_FALSE(s.HasPreviousSelection());
}

TEST_CASE("SessionState: previous-selection mirror set/get round-trip",
          "[session_state]") {
    SessionState s;
    s.SetModelCount(5);
    s.SetFrameCount(10);

    s.SetPreviousFrame(3);
    REQUIRE(s.GetPreviousFrame() == 3);
    s.SetPreviousModelRows({2, 4});
    REQUIRE(s.GetPreviousModelRows() == std::vector<int>({2, 4}));

    // Negative frame resolves to -1 (none), matching SetCurrentFrame.
    s.SetPreviousFrame(-4);
    REQUIRE(s.GetPreviousFrame() == -1);

    // The two mirrors are independent stores.
    s.SetPreviousFrame(1);
    REQUIRE(s.GetPreviousFrame() == 1);
    REQUIRE(s.GetPreviousModelRows() == std::vector<int>({2, 4}));
    s.SetPreviousModelRows({});
    REQUIRE(s.GetPreviousFrame() == 1);
    REQUIRE(s.GetPreviousModelRows().empty());
}

TEST_CASE("SessionState: previous model rows stay sorted after arbitrary writes",
          "[session_state]") {
    SessionState s;
    s.SetModelCount(5);

    s.SetPreviousModelRows({4, 1, 3});  // unsorted input
    REQUIRE(s.GetPreviousModelRows() == std::vector<int>({1, 3, 4}));

    s.SetPreviousModelRows({3, 1, 1, 2});  // duplicates preserved, sorted
    REQUIRE(s.GetPreviousModelRows() == std::vector<int>({1, 1, 2, 3}));

    s.SetPreviousModelRows({0, 2, 7, -1});  // invalid rows pruned
    REQUIRE(s.GetPreviousModelRows() == std::vector<int>({0, 2}));

    s.SetPreviousModelRows({});  // clear
    REQUIRE(s.GetPreviousModelRows().empty());
}

TEST_CASE("SessionState: HasPreviousSelection requires both mirrors set",
          "[session_state]") {
    SessionState s;
    s.SetModelCount(5);
    REQUIRE_FALSE(s.HasPreviousSelection());

    s.SetPreviousFrame(2);  // rows still empty
    REQUIRE_FALSE(s.HasPreviousSelection());

    s.SetPreviousModelRows({1});  // both set
    REQUIRE(s.HasPreviousSelection());

    s.SetPreviousModelRows({});  // rows cleared
    REQUIRE_FALSE(s.HasPreviousSelection());

    s.SetPreviousModelRows({1});
    s.SetPreviousFrame(-1);  // frame cleared
    REQUIRE_FALSE(s.HasPreviousSelection());
}

TEST_CASE(
    "SessionState: previous-selection mirrors are independent of the current selection",
    "[session_state]") {
    SessionState s;
    s.SetModelCount(5);
    s.SetFrameCount(10);
    s.SetSelectedModels({2});
    s.SetCurrentFrame(4);
    REQUIRE(s.GetPreviousFrame() == -1);  // current selection does not touch mirrors
    REQUIRE(s.GetPreviousModelRows().empty());

    s.SetPreviousFrame(4);
    s.SetPreviousModelRows({2});
    // Writing the mirrors does not disturb the current selection.
    REQUIRE(s.GetPrimaryModelIndex() == 2);
    REQUIRE(s.GetCurrentFrame() == 4);
}
