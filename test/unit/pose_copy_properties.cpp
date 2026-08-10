// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Hegel property-based tests for the pure pose-copy seam (plan 004 U4,
// R13/R15). PBT complements the deterministic cases in pose_copy_test.cpp by
// locking the index/boundary invariants the copy slots depend on:
//   - the read/write index split (read frame = current +- 1 at the primary
//     model; write frame = current at the CURRENT model row) holds for every
//     input,
//   - boundary_fallback is exactly "the read index is outside
//     [0, frame_count)" (row 0 / last row / no-frames all flagged),
//   - +1/-1 round-trips return to the original row,
//   - the guard is exactly the preserved decision table (selection check
//     first, radio check second, Ok otherwise),
//   - determinism: same input -> same plan.

#include <hegel/hegel.h>

#include <catch2/catch_test_macros.hpp>

#include "domain/pose_copy.h"

namespace gs = hegel::generators;
namespace pc = jta::pose_copy;

namespace {

// Frame rows: -1 (no frame selected) .. 12; model rows and frame count are
// non-negative. Small ranges keep the boundary cases dense.
auto FrameRow() { return gs::integers<int>({.min_value = -1, .max_value = 12}); }
auto ModelRow() { return gs::integers<int>({.min_value = 0, .max_value = 5}); }
auto FrameCount() {
    return gs::integers<int>({.min_value = 0, .max_value = 12});
}

bool SamePlan(const pc::CopyPlan& a, const pc::CopyPlan& b) {
    return a.read_frame == b.read_frame && a.read_model == b.read_model &&
           a.write_frame == b.write_frame && a.write_model == b.write_model &&
           a.boundary_fallback == b.boundary_fallback;
}

}  // namespace

TEST_CASE("pose_copy[PBT]: read/write index split holds for every input",
          "[pose_copy][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto current = tc.draw("current_frame_row", FrameRow());
            auto model = tc.draw("current_model_row", ModelRow());
            auto primary = tc.draw("primary_model_row", ModelRow());
            auto count = tc.draw("frame_count", FrameCount());

            auto prev = pc::PreviousPose(current, model, primary, count);
            // READ at row - 1, WRITE at the current row (never aligned).
            REQUIRE(prev.read_frame == current - 1);
            REQUIRE(prev.write_frame == current);
            // READ at the primary (first-selected) model, WRITE at the
            // CURRENT model row -- the R13 split.
            REQUIRE(prev.read_model == primary);
            REQUIRE(prev.write_model == model);

            auto next = pc::NextPose(current, model, primary, count);
            REQUIRE(next.read_frame == current + 1);
            REQUIRE(next.write_frame == current);
            REQUIRE(next.read_model == primary);
            REQUIRE(next.write_model == model);
        },
        hegel::Settings{.test_cases = 400});
}

TEST_CASE("pose_copy[PBT]: boundary fallback exactly when the read is out of "
          "range (row 0 / last row / no frames)",
          "[pose_copy][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto current = tc.draw("current_frame_row", FrameRow());
            auto model = tc.draw("current_model_row", ModelRow());
            auto primary = tc.draw("primary_model_row", ModelRow());
            auto count = tc.draw("frame_count", FrameCount());

            auto prev = pc::PreviousPose(current, model, primary, count);
            REQUIRE(prev.boundary_fallback ==
                    (prev.read_frame < 0 || prev.read_frame >= count));

            auto next = pc::NextPose(current, model, primary, count);
            REQUIRE(next.boundary_fallback ==
                    (next.read_frame < 0 || next.read_frame >= count));

            // The named boundary cases, explicitly:
            if (current == 0) {
                REQUIRE(prev.boundary_fallback);  // read is -1
            }
            if (count > 0 && current == count - 1) {
                REQUIRE(next.boundary_fallback);  // read is frame_count
            }
            if (count > 0 && current > 0 && current < count - 1) {
                REQUIRE_FALSE(prev.boundary_fallback);
                REQUIRE_FALSE(next.boundary_fallback);
            }
        },
        hegel::Settings{.test_cases = 400});
}

TEST_CASE("pose_copy[PBT]: +1/-1 round-trip returns to the original row",
          "[pose_copy][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto current = tc.draw("current_frame_row", FrameRow());
            auto model = tc.draw("current_model_row", ModelRow());
            auto primary = tc.draw("primary_model_row", ModelRow());
            auto count = tc.draw("frame_count", FrameCount());

            auto prev = pc::PreviousPose(current, model, primary, count);
            auto back = pc::NextPose(prev.read_frame, prev.write_model,
                                     prev.read_model, count);
            REQUIRE(back.read_frame == current);
            REQUIRE(back.write_frame == prev.read_frame);

            auto next = pc::NextPose(current, model, primary, count);
            auto back2 = pc::PreviousPose(next.read_frame, next.write_model,
                                          next.read_model, count);
            REQUIRE(back2.read_frame == current);
        },
        hegel::Settings{.test_cases = 400});
}

TEST_CASE("pose_copy[PBT]: guard is exactly the preserved decision table",
          "[pose_copy][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto current = tc.draw("current_frame_row", FrameRow());
            auto selected = tc.draw("selected_model_count",
                                    gs::integers<int>(
                                        {.min_value = 0, .max_value = 8}));
            auto multi_radio =
                tc.draw("multi_model_radio_checked", gs::booleans());

            auto guard = pc::CheckSelection(current, selected, multi_radio);
            if (current < 0 || selected == 0) {
                // Selection check runs FIRST: it wins over the radio.
                REQUIRE(guard == pc::SelectionGuard::NoFrameOrModel);
            } else if (multi_radio) {
                // Radio check, not selection-mode check (R13).
                REQUIRE(guard == pc::SelectionGuard::MultiModelMode);
            } else {
                REQUIRE(guard == pc::SelectionGuard::Ok);
            }
        },
        hegel::Settings{.test_cases = 400});
}

TEST_CASE("pose_copy[PBT]: deterministic given the same input",
          "[pose_copy][pbt]") {
    hegel::test(
        [&](hegel::TestCase& tc) {
            auto current = tc.draw("current_frame_row", FrameRow());
            auto model = tc.draw("current_model_row", ModelRow());
            auto primary = tc.draw("primary_model_row", ModelRow());
            auto count = tc.draw("frame_count", FrameCount());

            auto prev1 = pc::PreviousPose(current, model, primary, count);
            auto prev2 = pc::PreviousPose(current, model, primary, count);
            REQUIRE(SamePlan(prev1, prev2));

            auto next1 = pc::NextPose(current, model, primary, count);
            auto next2 = pc::NextPose(current, model, primary, count);
            REQUIRE(SamePlan(next1, next2));
        },
        hegel::Settings{.test_cases = 400});
}
