// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel PBT for the pure SessionState app-state holder (plan U7).
// Complements the deterministic unit test by probing the selection
// well-formedness invariant across many random row sets:
//   - after SetSelectedModels, the selection is sorted non-decreasing and
//     every row is in [0, model_count) (out-of-range rows are dropped),
//   - the primary model is the first selected row (-1 when empty),
//   - IsSingleSelection() agrees with the selection cardinality, and
//   - SetModelCount on a narrower list prunes any selection row it can no
//     longer hold.
// Plus the previous-selection mirror (plan-006 U2) invariants: the mirror
// stays sorted/in-range after arbitrary writes, round-trips valid row sets
// exactly, and HasPreviousSelection() agrees with the stored mirrors.
// R2-safe: purely structural (order/bounds/count), no re-derivation.

#include <algorithm>
#include <vector>

#include <hegel/hegel.h>

#include <catch2/catch_test_macros.hpp>

#include "domain/session_state.h"

namespace gs = hegel::generators;

TEST_CASE("session_state[PBT]: selection is well-formed after SetSelectedModels",
          "[session_state][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto model_count =
                tc.draw(gs::integers<int>({.min_value = 0, .max_value = 20}));
            // Draw rows from a range that includes negatives and >model_count
            // values, so out-of-range indices are frequent (forces pruning).
            auto rows = tc.draw(gs::vectors(
                gs::integers<int>({.min_value = -5,
                                   .max_value = model_count + 5}),
                {.min_size = 0, .max_size = 12}));

            jta::SessionState s;
            s.SetModelCount(model_count);
            s.SetSelectedModels(rows);

            const auto& sel = s.GetSelectedModels();
            REQUIRE(std::is_sorted(sel.begin(), sel.end()));
            for (int r : sel) {
                REQUIRE(r >= 0);
                REQUIRE(r < model_count);
            }
            REQUIRE(s.GetPrimaryModelIndex() == (sel.empty() ? -1 : sel.front()));
            REQUIRE(s.IsSingleSelection() == (sel.size() == 1));

            // Determinism: re-applying the same rows yields the same selection.
            s.SetSelectedModels(rows);
            REQUIRE(s.GetSelectedModels() == sel);
        },
        hegel::Settings{.test_cases = 400});
}

TEST_CASE("session_state[PBT]: SetModelCount prunes out-of-range selection",
          "[session_state][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto big = tc.draw(gs::integers<int>({.min_value = 5, .max_value = 20}));
            auto rows = tc.draw(gs::vectors(
                gs::integers<int>({.min_value = 0, .max_value = 20}),
                {.min_size = 0, .max_size = 10}));
            auto shrink = tc.draw(gs::integers<int>({.min_value = 0,
                                                     .max_value = big}));

            jta::SessionState s;
            s.SetModelCount(big);
            s.SetSelectedModels(rows);
            s.SetModelCount(shrink);
            for (int r : s.GetSelectedModels()) {
                REQUIRE(r < shrink);
            }
        },
        hegel::Settings{.test_cases = 300});
}

TEST_CASE("session_state[PBT]: previous-model mirror is well-formed after "
          "SetPreviousModelRows",
          "[session_state][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto model_count =
                tc.draw(gs::integers<int>({.min_value = 0, .max_value = 20}));
            // Draw rows from a range that includes negatives and >model_count
            // values, so out-of-range indices are frequent (forces pruning).
            auto rows = tc.draw(gs::vectors(
                gs::integers<int>({.min_value = -5,
                                   .max_value = model_count + 5}),
                {.min_size = 0, .max_size = 12}));

            jta::SessionState s;
            s.SetModelCount(model_count);
            s.SetPreviousModelRows(rows);

            const auto& prev = s.GetPreviousModelRows();
            REQUIRE(std::is_sorted(prev.begin(), prev.end()));
            for (int r : prev) {
                REQUIRE(r >= 0);
                REQUIRE(r < model_count);
            }

            // Determinism: re-applying the same rows yields the same mirror.
            s.SetPreviousModelRows(rows);
            REQUIRE(s.GetPreviousModelRows() == prev);
        },
        hegel::Settings{.test_cases = 400});
}

TEST_CASE("session_state[PBT]: previous-model mirror round-trips valid row sets "
          "and HasPreviousSelection agrees",
          "[session_state][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto model_count =
                tc.draw(gs::integers<int>({.min_value = 1, .max_value = 20}));
            // Rows drawn from the valid range: the mirror round-trips exactly
            // (sorted, nothing pruned).
            auto rows = tc.draw(gs::vectors(
                gs::integers<int>({.min_value = 0, .max_value = model_count - 1}),
                {.min_size = 0, .max_size = 12}));
            auto have_frame = tc.draw(gs::booleans());

            jta::SessionState s;
            s.SetModelCount(model_count);
            s.SetPreviousFrame(have_frame ? 0 : -1);
            s.SetPreviousModelRows(rows);

            auto expected = rows;
            std::sort(expected.begin(), expected.end());
            REQUIRE(s.GetPreviousModelRows() == expected);
            REQUIRE(s.HasPreviousSelection() == (have_frame && !rows.empty()));
        },
        hegel::Settings{.test_cases = 400});
}
