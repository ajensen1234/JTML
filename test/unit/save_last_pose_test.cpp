// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Save-last-pose shared core tests (plan 006 U3, R10 part / R13, H4). The
// parameterized jta::SaveLastPoseToStorage (services layer) is the ONE core
// the four divergent save-last-pose call sites drive — the widgets
// MainScreen::SaveLastPose, the camera-A and camera-B slot inline copies
// (converge in U9), and the QML OptimizerBridge::
// saveScenePosesForCurrentSelection mirror. Each behavior is correct for its
// source's coordinate frame, so the call-site TABLE is pinned, not unified
// (the conversion-divergence unification is the Deferred follow-up cut):
// every row below reproduces a call site's parameterization and asserts the
// exact LocationStorage::SavePose arguments the core writes. The pre-cut
// oracle (verbatim old bodies) at the bottom proves the core writes
// storage-identical contents for equivalent inputs.
//
// Pure logic: no Qt, no GPU, no widgets — headless Catch2.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <vector>

#include "domain/data_structures_6D.h"
#include "services/calibration.h"
#include "services/location_storage.h"
#include "services/save_last_pose.h"

using Catch::Approx;
using jta::SaveLastPoseToStorage;
using jta::SavePoseConvertRule;

namespace {

/*F × M storage with the default loaded poses (z = -.25*pd/pp, non-zero).*/
LocationStorage MakeStorage(int frames, int models) {
    LocationStorage storage;
    for (int m = 0; m < models; ++m) {
        storage.LoadNewModel(1198.0, 0.373);
    }
    for (int f = 0; f < frames; ++f) {
        storage.LoadNewFrame();
    }
    return storage;
}

/*Biplane calibration with identity B axes + a (10, 20, 30) mm origin
 * offset: the B→A conversion translates position by the offset and (identity
 * axes, ZXY recovery) reproduces the orientation — predictable expected
 * values. Identity axes make `R_B = axes_B_ * R = R`, and the ZXY
 * extraction of R = Rz*Rx*Ry recovers the input angles exactly.*/
Calibration MakeBiplaneCalibration() {
    return Calibration(
        CameraCalibration(1198, 0, 0, 0.373),
        CameraCalibration(1198, 0, 0, 0.373),
        Vect_3(10.0f, 20.0f, 30.0f),
        Matrix_3_3(1, 0, 0, 0, 1, 0, 0, 0, 1));
}

/*Pose equality with tolerance — the B→A conversion runs float math.*/
void RequirePose(
    LocationStorage& storage, int frame, int row, const Point6D& expected) {
    const Point6D pose = storage.GetPose(frame, row);
    REQUIRE(pose.x == Approx(expected.x).margin(1e-3));
    REQUIRE(pose.y == Approx(expected.y).margin(1e-3));
    REQUIRE(pose.z == Approx(expected.z).margin(1e-3));
    REQUIRE(pose.xa == Approx(expected.xa).margin(1e-3));
    REQUIRE(pose.ya == Approx(expected.ya).margin(1e-3));
    REQUIRE(pose.za == Approx(expected.za).margin(1e-3));
}

/*Viewer-source functor shape (vw->get_model_position_at_index /
 * get_model_orientation_at_index).*/
std::function<Point6D(int)> ViewerSource(const std::vector<Point6D>& poses) {
    return [&poses](int row) { return poses[static_cast<size_t>(row)]; };
}

/*Full-storage dump compare. Bit-exact: the core and the pre-cut oracle
 * compute every converted pose through the SAME convert_Pose_B_to_Pose_A
 * call on the same input, and raw poses are double copies.*/
void RequireSameStorage(LocationStorage& a, LocationStorage& b) {
    REQUIRE(a.GetFrameCount() == b.GetFrameCount());
    REQUIRE(a.GetModelCount() == b.GetModelCount());
    for (int f = 0; f < a.GetFrameCount(); ++f) {
        for (int m = 0; m < a.GetModelCount(); ++m) {
            const Point6D pa = a.GetPose(f, m);
            const Point6D pb = b.GetPose(f, m);
            REQUIRE(pa.x == pb.x);
            REQUIRE(pa.y == pb.y);
            REQUIRE(pa.z == pb.z);
            REQUIRE(pa.xa == pb.xa);
            REQUIRE(pa.ya == pb.ya);
            REQUIRE(pa.za == pb.za);
        }
    }
}

/*Pre-cut characterization oracle: the four call-site bodies VERBATIM as
 * they existed before U3 (mainscreen.cpp:4101-4129 / :2641-2662 /
 * :2801-2818, OptimizerBridge.cpp:466-481), with the QModelIndexList /
 * QVariantList plumbing already reduced to plain rows (the signature
 * adaptation). `currently_optimizing_` (view-side, camera slots) is out of
 * scope of the storage-write contract and excluded from both sides.*/
namespace precut {

/*Row 1: widgets MainScreen::SaveLastPose.*/
void WidgetsSaveLastPose(
    int previous_frame_index,
    const std::vector<int>& previous_model_indices,
    const std::vector<Point6D>& viewer_poses,
    bool camera_a_checked,
    Calibration& calibration,
    LocationStorage& storage) {
    if (previous_model_indices.size() > 0 && previous_frame_index != -1) {
        for (size_t i = 0; i < previous_model_indices.size(); i++) {
            Point6D last_pose = viewer_poses[previous_model_indices[i]];
            /*If Camera B View, Save in Camera A coordinates*/
            if (camera_a_checked) {
                storage.SavePose(
                    previous_frame_index, previous_model_indices[i], last_pose);
            } else {
                storage.SavePose(
                    previous_frame_index,
                    previous_model_indices[i],
                    calibration.convert_Pose_B_to_Pose_A(last_pose));
            }
        }
    }
}

/*Row 2: camera-A slot inline (actor-list source, always convert).*/
void CameraASlotInline(
    int previous_frame_index,
    const std::vector<int>& selected,
    const std::vector<Point6D>& actor_poses,
    Calibration& calibration,
    LocationStorage& storage) {
    for (size_t r = 0; r < selected.size(); r++) {
        if (selected.size() != 0 && previous_frame_index != -1) {
            Point6D last_pose = actor_poses[selected[r]];
            /*Camera A View, Save in Camera A coordinates by converting
             * camera B*/
            storage.SavePose(
                previous_frame_index,
                selected[r],
                calibration.convert_Pose_B_to_Pose_A(last_pose));
        }
    }
}

/*Row 3: camera-B slot inline (viewer source, never convert).*/
void CameraBSlotInline(
    int previous_frame_index,
    const std::vector<int>& selected,
    const std::vector<Point6D>& viewer_poses,
    LocationStorage& storage) {
    for (size_t r = 0; r < selected.size(); r++) {
        if (selected.size() != 0 && previous_frame_index != -1) {
            Point6D last_pose = viewer_poses[selected[r]];
            storage.SavePose(previous_frame_index, selected[r], last_pose);
        }
    }
}

/*Row 4: QML OptimizerBridge::saveScenePosesForCurrentSelection.*/
void QmlSaveScenePoses(
    int frame,
    const std::vector<int>& selected,
    const std::vector<Point6D>& scene_poses,
    LocationStorage& storage) {
    if (frame < 0) {
        return;
    }
    for (size_t i = 0; i < selected.size(); i++) {
        const int model_row = selected[i];
        if (model_row >= 0 &&
            model_row < static_cast<int>(scene_poses.size())) {
            storage.SavePose(
                frame, model_row, scene_poses[static_cast<size_t>(model_row)]);
        }
    }
}

} // namespace precut

} // namespace

/*Scenario (a): the call-site table — each of the four rows writes the
 * expected SavePose arguments given (source, selection, frame, convert
 * rule).*/
TEST_CASE(
    "U3 table: four call sites write their pinned SavePose arguments",
    "[save_last_pose]") {
    Calibration biplane = MakeBiplaneCalibration();

    /*Per-source pose sets (the three live pose sources: viewer, actor list,
     * QML scene). All poses non-zero so the sentinel skip never triggers.*/
    const std::vector<Point6D> viewer_poses = {
        Point6D(1, 2, 3, 10, 20, 30), // row 0
        Point6D(-4, 5, -6, 0, 0, 0),  // row 1
        Point6D(7, 8, 9, 0, 0, 0),    // row 2
    };
    const std::vector<Point6D> actor_poses = {
        Point6D(7, 8, 9, 0, 0, 0),    // row 0 (unused)
        Point6D(11, 12, 13, 0, 0, 0), // row 1
        Point6D(15, 16, 17, 0, 0, 0), // row 2
    };
    const std::vector<Point6D> scene_poses = {
        Point6D(0, 0, 0, 0, 0, 0),    // row 0 (unused, zero)
        Point6D(21, 22, 23, 0, 0, 0), // row 1
        Point6D(25, 26, 27, 0, 0, 0), // row 2
    };

    SECTION(
        "row 1: widgets SaveLastPose — previous selection, previous "
        "frame, viewer source, convert iff camera B") {
        LocationStorage storage = MakeStorage(2, 3);
        const int written = SaveLastPoseToStorage(
            0 /* previous frame */,
            {0, 1} /* previous selection */,
            ViewerSource(viewer_poses),
            /*camera_is_a=*/true,
            SavePoseConvertRule::ConvertWhenCameraB,
            biplane,
            storage);
        REQUIRE(written == 2);
        /*Camera A checked -> raw save.*/
        RequirePose(storage, 0, 0, viewer_poses[0]);
        RequirePose(storage, 0, 1, viewer_poses[1]);
    }

    SECTION(
        "row 2: camera-A slot inline — CURRENT selection, previous "
        "frame, actor-list source, ALWAYS convert B->A") {
        LocationStorage storage = MakeStorage(2, 3);
        const int written = SaveLastPoseToStorage(
            0 /* previous frame */,
            {1, 2} /* current selection */,
            ViewerSource(actor_poses),
            /*camera_is_a=*/true, /* unused by ConvertBToA */
            SavePoseConvertRule::ConvertBToA,
            biplane,
            storage);
        REQUIRE(written == 2);
        /*+origin offset (10, 20, 30); identity axes keep orientation.*/
        RequirePose(storage, 0, 1, Point6D(21, 32, 43, 0, 0, 0));
        RequirePose(storage, 0, 2, Point6D(25, 36, 47, 0, 0, 0));
    }

    SECTION(
        "row 3: camera-B slot inline — current selection, previous "
        "frame, viewer source, NEVER convert (raw save is correct: the "
        "source is A-coords)") {
        LocationStorage storage = MakeStorage(2, 3);
        const int written = SaveLastPoseToStorage(
            0 /* previous frame */,
            {0, 2} /* current selection */,
            ViewerSource(viewer_poses),
            /*camera_is_a=*/false, /* unused by NeverConvert */
            SavePoseConvertRule::NeverConvert,
            biplane,
            storage);
        REQUIRE(written == 2);
        RequirePose(storage, 0, 0, viewer_poses[0]);
        RequirePose(storage, 0, 2, viewer_poses[2]);
    }

    SECTION(
        "row 4: QML mirror — current selection, CURRENT frame, scene "
        "source, never convert") {
        LocationStorage storage = MakeStorage(3, 3);
        const int written = SaveLastPoseToStorage(
            2 /* current frame */,
            {1} /* current selection */,
            ViewerSource(scene_poses),
            /*camera_is_a=*/true, /* unused by NeverConvert */
            SavePoseConvertRule::NeverConvert,
            biplane,
            storage);
        REQUIRE(written == 1);
        RequirePose(storage, 2, 1, scene_poses[1]);
        /*No cross-frame writes.*/
        RequirePose(
            storage,
            0,
            1,
            Point6D(0, 0, -0.25 * 1198.0 / 0.373, 0, 0, 0)); // untouched
    }
}

/*Scenario (b): no previous selection -> no-op, no write.*/
TEST_CASE(
    "U3 no previous selection: empty rows and -1 frame are no-ops",
    "[save_last_pose]") {
    Calibration calibration(CameraCalibration(1198, 0, 0, 0.373));
    LocationStorage storage = MakeStorage(2, 2);
    /*Sentinel poses prove nothing is touched.*/
    storage.SavePose(0, 0, Point6D(9, 9, 9, 9, 9, 9));
    storage.SavePose(1, 1, Point6D(8, 8, 8, 8, 8, 8));
    const std::vector<Point6D> poses = {
        Point6D(1, 2, 3, 0, 0, 0), Point6D(4, 5, 6, 0, 0, 0)};

    REQUIRE(
        SaveLastPoseToStorage(
            -1 /* no previous frame */,
            {0},
            ViewerSource(poses),
            true,
            SavePoseConvertRule::ConvertWhenCameraB,
            calibration,
            storage) == 0);
    REQUIRE(
        SaveLastPoseToStorage(
            0,
            {} /* no previous selection */,
            ViewerSource(poses),
            true,
            SavePoseConvertRule::ConvertWhenCameraB,
            calibration,
            storage) == 0);
    RequirePose(storage, 0, 0, Point6D(9, 9, 9, 9, 9, 9));
    RequirePose(storage, 1, 1, Point6D(8, 8, 8, 8, 8, 8));
}

/*Scenario (c): convert-rule matrix — the canonical row over every
 * rule × camera_is_a combination.*/
TEST_CASE(
    "U3 convert-rule matrix: A checked raw, B checked converts B->A",
    "[save_last_pose]") {
    Calibration biplane = MakeBiplaneCalibration();
    const Point6D pose(1, 2, 3, 0, 0, 0); // zero orientation: predictable
    const Point6D converted(11, 22, 33, 0, 0, 0); // + origin offset
    auto source = [&pose](int) { return pose; };

    struct Case {
        SavePoseConvertRule rule;
        bool camera_is_a;
        Point6D expected;
    };
    const std::vector<Case> cases = {
        {SavePoseConvertRule::NeverConvert, true, pose},
        {SavePoseConvertRule::NeverConvert, false, pose},
        {SavePoseConvertRule::ConvertBToA, true, converted},
        {SavePoseConvertRule::ConvertBToA, false, converted},
        {SavePoseConvertRule::ConvertWhenCameraB, true, pose},
        {SavePoseConvertRule::ConvertWhenCameraB, false, converted},
    };
    for (const Case& c : cases) {
        LocationStorage storage = MakeStorage(1, 1);
        const int written = SaveLastPoseToStorage(
            0, {0}, source, c.camera_is_a, c.rule, biplane, storage);
        REQUIRE(written == 1);
        RequirePose(storage, 0, 0, c.expected);
    }
}

/*Scenario (d): pose-source functor returns out-of-range/zero for a row ->
 * skipped without corrupting the other rows.*/
TEST_CASE(
    "U3 out-of-range/zero pose source: row skipped, others written",
    "[save_last_pose]") {
    Calibration calibration(CameraCalibration(1198, 0, 0, 0.373));
    LocationStorage storage = MakeStorage(1, 3);
    const std::vector<Point6D> poses = {
        Point6D(1, 2, 3, 0, 0, 0),
        Point6D(4, 5, 6, 0, 0, 0),
        Point6D(7, 8, 9, 0, 0, 0),
    };
    /*Row 1's source has "no valid pose" (out-of-range -> all-zero
     * sentinel).*/
    auto source = [&poses](int row) {
        if (row == 1) {
            return Point6D(); // no valid pose
        }
        return poses[static_cast<size_t>(row)];
    };

    const int written = SaveLastPoseToStorage(
        0,
        {0, 1, 2},
        source,
        true,
        SavePoseConvertRule::NeverConvert,
        calibration,
        storage);
    REQUIRE(written == 2);
    RequirePose(storage, 0, 0, poses[0]);
    RequirePose(storage, 0, 2, poses[2]);
    /*Row 1 untouched — keeps the storage's pre-existing default pose (a
     * bogus all-zero write would corrupt it).*/
    RequirePose(storage, 0, 1, Point6D(0, 0, -0.25 * 1198.0 / 0.373, 0, 0, 0));
}

/*Scenario (e): integration — the core produces storage-identical contents
 * to the pre-cut call-site bodies for equivalent inputs (the headless part
 * of the pre-cut storage-dump compare).*/
TEST_CASE(
    "U3 integration: core storage dump == pre-cut call-site bodies",
    "[save_last_pose]") {
    Calibration biplane = MakeBiplaneCalibration();
    const std::vector<Point6D> viewer_poses = {
        Point6D(1, 2, 3, 10, 20, 30),
        Point6D(-4, 5, -6, 0, 0, 0),
        Point6D(7, 8, 9, 0, 0, 0),
    };
    const std::vector<Point6D> actor_poses = {
        Point6D(7, 8, 9, 0, 0, 0),
        Point6D(11, 12, 13, 0, 0, 0),
        Point6D(15, 16, 17, 0, 0, 0),
    };
    const std::vector<Point6D> scene_poses = {
        Point6D(21, 22, 23, 0, 0, 0),
        Point6D(25, 26, 27, 0, 0, 0),
        Point6D(29, 30, 31, 0, 0, 0),
    };

    SECTION("widgets SaveLastPose, camera A checked (raw)") {
        LocationStorage a = MakeStorage(2, 3);
        LocationStorage b = MakeStorage(2, 3);
        precut::WidgetsSaveLastPose(0, {0, 1}, viewer_poses, true, biplane, a);
        SaveLastPoseToStorage(
            0,
            {0, 1},
            ViewerSource(viewer_poses),
            true,
            SavePoseConvertRule::ConvertWhenCameraB,
            biplane,
            b);
        RequireSameStorage(a, b);
    }
    SECTION("widgets SaveLastPose, camera B checked (converted)") {
        LocationStorage a = MakeStorage(2, 3);
        LocationStorage b = MakeStorage(2, 3);
        precut::WidgetsSaveLastPose(0, {0, 1}, viewer_poses, false, biplane, a);
        SaveLastPoseToStorage(
            0,
            {0, 1},
            ViewerSource(viewer_poses),
            false,
            SavePoseConvertRule::ConvertWhenCameraB,
            biplane,
            b);
        RequireSameStorage(a, b);
    }
    SECTION("camera-A slot inline (actor source, always convert)") {
        LocationStorage a = MakeStorage(2, 3);
        LocationStorage b = MakeStorage(2, 3);
        precut::CameraASlotInline(0, {1, 2}, actor_poses, biplane, a);
        SaveLastPoseToStorage(
            0,
            {1, 2},
            ViewerSource(actor_poses),
            true,
            SavePoseConvertRule::ConvertBToA,
            biplane,
            b);
        RequireSameStorage(a, b);
    }
    SECTION("camera-B slot inline (viewer source, never convert)") {
        LocationStorage a = MakeStorage(2, 3);
        LocationStorage b = MakeStorage(2, 3);
        precut::CameraBSlotInline(0, {0, 2}, viewer_poses, a);
        SaveLastPoseToStorage(
            0,
            {0, 2},
            ViewerSource(viewer_poses),
            false,
            SavePoseConvertRule::NeverConvert,
            biplane,
            b);
        RequireSameStorage(a, b);
    }
    SECTION(
        "QML mirror (scene source, current frame, never convert, incl. "
        "out-of-range row)") {
        LocationStorage a = MakeStorage(3, 3);
        LocationStorage b = MakeStorage(3, 3);
        precut::QmlSaveScenePoses(2, {0, 1, 5}, scene_poses, a);
        /*The bridge's scene lambda keeps the row-range guard as the
         * all-zero sentinel the core skips (row 5 out of range).*/
        auto scene_source = [&scene_poses](int model_row) {
            if (model_row >= 0 &&
                model_row < static_cast<int>(scene_poses.size())) {
                return scene_poses[static_cast<size_t>(model_row)];
            }
            return Point6D(); // out-of-range -> core skips
        };
        SaveLastPoseToStorage(
            2,
            {0, 1, 5},
            scene_source,
            true,
            SavePoseConvertRule::NeverConvert,
            biplane,
            b);
        RequireSameStorage(a, b);
    }
}
