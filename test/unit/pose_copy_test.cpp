// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Deterministic Catch2 tests for the pure pose-copy seam (plan 004 U4,
// R8/R13/R14). Pins the named boundary cases the seam must preserve
// byte-identical:
//   - copy-previous at row 0 / copy-next at the last row fall back to the
//     no-image default pose (the model's initial pose) and overwrite frame 0 /
//     the last frame,
//   - the primary-vs-current index split (read uses the first-selected model,
//     write uses the CURRENT model row),
//   - the guard decision (selection check first; multi-model RADIO check,
//     not selection mode),
//   - the camera-B no-conversion property (the copy chain stores the pose
//     read at the plan's cell verbatim -- the seam carries no calibration
//     hook).
// The LocationStorage service is compiled in so the boundary fallback chain is
// pinned against the real storage (GetPose(-1, ...) -> no-image vector).

#include <catch2/catch_test_macros.hpp>

#include "domain/data_structures_6D.h"
#include "domain/pose_copy.h"
#include "services/location_storage.h"

namespace pc = jta::pose_copy;

namespace {

// 2 models x 3 frames; model default pose (0, 0, -2500, 0, 0, 0) from
// LoadNewModel(1000.0, 0.1).
LocationStorage MakeStorage() {
    LocationStorage ls;
    ls.LoadNewModel(1000.0, 0.1);
    ls.LoadNewModel(1000.0, 0.1);
    ls.LoadNewFrame();
    ls.LoadNewFrame();
    ls.LoadNewFrame();
    return ls;
}

bool SamePose(const Point6D& a, const Point6D& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.xa == b.xa &&
           a.ya == b.ya && a.za == b.za;
}

// The exact copy chain the view runs for a Copy_Previous/Copy_Next slot:
// GetPose at the plan's read cell, SavePose at the plan's write cell, with
// the RAW read index (no clamping, no conversion).
void RunCopyChain(LocationStorage& ls, const pc::CopyPlan& plan) {
    ls.SavePose(plan.write_frame, plan.write_model,
                ls.GetPose(plan.read_frame, plan.read_model));
}

}  // namespace

TEST_CASE("pose_copy: copy-previous at an interior frame moves the pose",
          "[pose_copy]") {
    LocationStorage ls = MakeStorage();
    Point6D A(1.5, -2.25, 30.0, 10.0, -5.0, 90.0);
    ls.SavePose(1, 0, A);

    pc::CopyPlan plan = pc::PreviousPose(2, 0, 0, ls.GetFrameCount());
    REQUIRE(plan.read_frame == 1);
    REQUIRE(plan.read_model == 0);
    REQUIRE(plan.write_frame == 2);
    REQUIRE(plan.write_model == 0);
    REQUIRE_FALSE(plan.boundary_fallback);

    RunCopyChain(ls, plan);
    REQUIRE(SamePose(ls.GetPose(2, 0), A));
}

TEST_CASE("pose_copy: copy-next at an interior frame moves the pose",
          "[pose_copy]") {
    LocationStorage ls = MakeStorage();
    Point6D A(1.5, -2.25, 30.0, 10.0, -5.0, 90.0);
    ls.SavePose(1, 0, A);

    pc::CopyPlan plan = pc::NextPose(0, 0, 0, ls.GetFrameCount());
    REQUIRE(plan.read_frame == 1);
    REQUIRE(plan.write_frame == 0);
    REQUIRE_FALSE(plan.boundary_fallback);

    RunCopyChain(ls, plan);
    REQUIRE(SamePose(ls.GetPose(0, 0), A));
}

TEST_CASE("pose_copy: copy-previous at row 0 falls back to the no-image "
          "default pose and overwrites frame 0 (R13)",
          "[pose_copy]") {
    LocationStorage ls = MakeStorage();
    // The no-image default is a VALID pose -- the model's initial pose, NOT
    // frame 0's stored pose.
    Point6D default_pose = ls.GetPose(-1, 0);
    REQUIRE(default_pose.z == -2500.0);
    REQUIRE(default_pose.x == 0.0);

    // Frame 0 holds a real (non-default) pose; copying previous onto it must
    // replace it with the DEFAULT, not with frame 0's own pose.
    Point6D X(9.0, 9.0, 9.0, 9.0, 9.0, 9.0);
    ls.SavePose(0, 0, X);

    pc::CopyPlan plan = pc::PreviousPose(0, 0, 0, ls.GetFrameCount());
    REQUIRE(plan.read_frame == -1);
    REQUIRE(plan.boundary_fallback);

    // Raw index read: GetPose(-1, ...) resolves to the no-image default.
    Point6D read = ls.GetPose(plan.read_frame, plan.read_model);
    REQUIRE(SamePose(read, default_pose));
    REQUIRE_FALSE(SamePose(read, X));

    RunCopyChain(ls, plan);
    REQUIRE(SamePose(ls.GetPose(0, 0), default_pose));
}

TEST_CASE("pose_copy: copy-next at the last row falls back to the no-image "
          "default pose and overwrites the last frame (R13)",
          "[pose_copy]") {
    LocationStorage ls = MakeStorage();
    Point6D default_pose = ls.GetPose(-1, 0);

    Point6D X(9.0, 9.0, 9.0, 9.0, 9.0, 9.0);
    ls.SavePose(2, 0, X);  // last frame holds a real pose

    pc::CopyPlan plan = pc::NextPose(2, 0, 0, ls.GetFrameCount());
    REQUIRE(plan.read_frame == 3);
    REQUIRE(plan.boundary_fallback);

    Point6D read = ls.GetPose(plan.read_frame, plan.read_model);
    REQUIRE(SamePose(read, default_pose));
    REQUIRE_FALSE(SamePose(read, X));

    RunCopyChain(ls, plan);
    REQUIRE(SamePose(ls.GetPose(2, 0), default_pose));
}

TEST_CASE("pose_copy: primary-vs-current index split is preserved (R13)",
          "[pose_copy]") {
    // Multi-selection scenario: primary (first-selected) model is 0, the
    // CURRENT model row is 3. The read must use the primary, the write the
    // current row -- never aligned by the seam.
    pc::CopyPlan prev = pc::PreviousPose(5, 3, 0, 10);
    REQUIRE(prev.read_model == 0);   // primary (first-selected)
    REQUIRE(prev.write_model == 3);  // current row
    REQUIRE(prev.read_frame == 4);
    REQUIRE(prev.write_frame == 5);

    pc::CopyPlan next = pc::NextPose(5, 3, 0, 10);
    REQUIRE(next.read_model == 0);
    REQUIRE(next.write_model == 3);
    REQUIRE(next.read_frame == 6);
    REQUIRE(next.write_frame == 5);
}

TEST_CASE("pose_copy: copy chain stores the pose verbatim -- no A<->B "
          "conversion (R13)",
          "[pose_copy]") {
    // The seam's plan carries indices + a boundary flag only; there is no
    // calibration/conversion surface anywhere in the API. A camera-B copy
    // therefore stores the A-coordinate pose un-converted (the model jumps in
    // the B viewport) -- preserved.
    LocationStorage ls = MakeStorage();
    Point6D A(1.5, -2.0, 30.0, 10.0, -5.0, 90.0);
    ls.SavePose(1, 0, A);

    pc::CopyPlan plan = pc::PreviousPose(2, 0, 0, ls.GetFrameCount());
    Point6D raw = ls.GetPose(plan.read_frame, plan.read_model);
    REQUIRE(SamePose(raw, A));  // read back verbatim

    RunCopyChain(ls, plan);
    Point6D stored = ls.GetPose(2, 0);
    // Bit-identical transfer: no transform was applied between read and write.
    REQUIRE(stored.x == A.x);
    REQUIRE(stored.y == A.y);
    REQUIRE(stored.z == A.z);
    REQUIRE(stored.xa == A.xa);
    REQUIRE(stored.ya == A.ya);
    REQUIRE(stored.za == A.za);
}

TEST_CASE("pose_copy: selection guard decision table (R13)", "[pose_copy]") {
    // Happy paths.
    REQUIRE(pc::CheckSelection(0, 1, false) == pc::SelectionGuard::Ok);
    REQUIRE(pc::CheckSelection(3, 2, false) == pc::SelectionGuard::Ok);

    // No frame (current row < 0) or no model selection.
    REQUIRE(pc::CheckSelection(-1, 1, false) ==
            pc::SelectionGuard::NoFrameOrModel);
    REQUIRE(pc::CheckSelection(0, 0, false) ==
            pc::SelectionGuard::NoFrameOrModel);

    // Multi-model guard tests the RADIO, not the selection mode: even a
    // single selected model errors out when the radio is checked.
    REQUIRE(pc::CheckSelection(0, 1, true) ==
            pc::SelectionGuard::MultiModelMode);
    REQUIRE(pc::CheckSelection(0, 2, true) ==
            pc::SelectionGuard::MultiModelMode);

    // The selection check runs FIRST: a missing selection reports
    // NoFrameOrModel even when the multi-model radio is also checked.
    REQUIRE(pc::CheckSelection(-1, 1, true) ==
            pc::SelectionGuard::NoFrameOrModel);
    REQUIRE(pc::CheckSelection(0, 0, true) ==
            pc::SelectionGuard::NoFrameOrModel);
}

TEST_CASE("pose_copy: no-frames state (frame_count == 0)", "[pose_copy]") {
    // The guard blocks copy slots before the plan is ever computed (current
    // row is -1 with no frames), but the plan math stays consistent: any
    // read is out of range and flagged as a boundary fallback.
    pc::CopyPlan plan = pc::PreviousPose(0, 0, 0, 0);
    REQUIRE(plan.read_frame == -1);
    REQUIRE(plan.boundary_fallback);
}
