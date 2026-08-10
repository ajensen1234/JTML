// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// SessionController tests (plan 004 U6 / R6 + R10 / R13 / R14). Deterministic
// Catch2 twin pinning the load-path extraction: calibration parsing (against
// the real test/golden/calibration.txt fixture), image parsing + dataset
// population with the partial-load (goto stop / stop_biplane) semantics,
// model parsing + population, the PoseMatrixDimensionMismatch invariant
// (LocationStorage dims == frame/model counts), the camera radio
// enable/disable decision matrix, and the active-camera + dataset-count
// mirrors.
//
// The controller .cpp compiles directly against the headless Frame twin
// (test/unit/frame_headless.cpp, pure OpenCV), the real LocationStorage /
// Model / stl_reader / ModelListBuilder services, Qt6::Core (QString / QFile /
// QTextStream) and VTK (Model's vtkSTLReader -- no render window, no GPU).
// Image fixtures are written as temp PNGs at runtime (cv::imwrite), so no
// checked-in image fixtures are needed; the calibration fixture is the golden
// file (repo-root WORKING_DIRECTORY, like jtml.pose_file_io).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <QString>
#include <QStringList>
#include <QTemporaryDir>

#include <opencv2/imgcodecs.hpp>

#include "services/calibration.h"
#include "services/location_storage.h"
#include "services/model.h"
#include "services/session_controller.h"

using Catch::Approx;

namespace {

using jta::ActiveCamera;
using jta::CalibrationParseResult;
using jta::CameraRadioEvent;
using jta::ImageLoadParams;
using jta::ImageLoadStatus;
using jta::SessionController;

ImageLoadParams default_params() {
    return ImageLoadParams{3, 40, 120, 0};
}

/*Write a calibration file of the given lines into dir and return its path.*/
QString write_calibration(const QTemporaryDir& dir,
                          const QStringList& lines) {
    const QString path = dir.filePath("calib.txt");
    QFile file(path);
    REQUIRE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    for (const QString& line : lines) {
        file.write(line.toUtf8());
        file.write("\n");
    }
    file.close();
    return path;
}

/*Write a solid-gray PNG of the given size into dir and return its path.*/
QString write_png(const QTemporaryDir& dir, const QString& name, int w, int h) {
    cv::Mat img(h, w, CV_8UC1, cv::Scalar(64));
    const QString path = dir.filePath(name);
    REQUIRE(cv::imwrite(path.toStdString(), img));
    return path;
}

}  // namespace

TEST_CASE(
    "session_controller: golden calibration fixture parses to the same values",
    "[session_controller]") {
    /*The golden fixture: JT_INTCALIB / 1198 / 0 / 0 / 0.373 -- the values the
     * MainScreen slot produced before the extraction.*/
    const CalibrationParseResult result =
        SessionController::ParseCalibration("test/golden/calibration.txt");
    REQUIRE(result.ok);
    REQUIRE(result.error == CalibrationParseResult::Error::None);
    REQUIRE(result.kind == CalibrationParseResult::Kind::Monoplane);
    REQUIRE(result.calibrated_for_monoplane_viewport);
    REQUIRE_FALSE(result.calibrated_for_biplane_viewport);

    REQUIRE(result.calibration.biplane_calibration == false);
    REQUIRE(result.calibration.type_ == "UF");
    /*Negatives for the offsets (to make consistent with JointTrack): the
     * zero offsets parse to -0.0, which compares equal to 0.0.*/
    REQUIRE(result.calibration.camera_A_principal_.principal_distance_ ==
            Approx(1198.0));
    REQUIRE(result.calibration.camera_A_principal_.principal_x_ == Approx(0.0));
    REQUIRE(result.calibration.camera_A_principal_.principal_y_ == Approx(0.0));
    REQUIRE(result.calibration.camera_A_principal_.pixel_pitch_ ==
            Approx(0.373));
}

TEST_CASE("session_controller: calibration parse error paths",
          "[session_controller]") {
    QTemporaryDir dir;
    REQUIRE(dir.isValid());

    /*Pixel size zero (monoplane code) -- the error box path.*/
    const CalibrationParseResult zero =
        SessionController::ParseCalibration(write_calibration(
            dir, {"JT_INTCALIB", "100", "0", "0", "0"}));
    REQUIRE_FALSE(zero.ok);
    REQUIRE(zero.error == CalibrationParseResult::Error::PixelSizeZero);
    REQUIRE_FALSE(zero.calibrated_for_monoplane_viewport);
    REQUIRE_FALSE(zero.calibrated_for_biplane_viewport);

    /*Pixel size zero (biplane code) -- either A or B pixel pitch zero.*/
    const QStringList biplane_zero = {
        "JTA_INTCALIB_BIPLANE", "1", "2", "3", "4", "5", "6", "7", "0"};
    REQUIRE_FALSE(
        SessionController::ParseCalibration(write_calibration(dir, biplane_zero))
            .ok);
    REQUIRE(
        SessionController::ParseCalibration(write_calibration(dir, biplane_zero))
            .error == CalibrationParseResult::Error::PixelSizeZero);

    /*Invalid code.*/
    const CalibrationParseResult invalid =
        SessionController::ParseCalibration(
            write_calibration(dir, {"NOT_A_CALIBRATION_CODE", "1"}));
    REQUIRE_FALSE(invalid.ok);
    REQUIRE(invalid.error == CalibrationParseResult::Error::InvalidCode);

    /*File open failure: silent, nothing happened (the slot's open guard).*/
    const CalibrationParseResult missing = SessionController::ParseCalibration(
        dir.filePath("does_not_exist.txt"));
    REQUIRE_FALSE(missing.ok);
    REQUIRE(missing.error == CalibrationParseResult::Error::FileOpenFailed);
}

TEST_CASE("session_controller: biplane and Denver calibration parses",
          "[session_controller]") {
    QTemporaryDir dir;
    REQUIRE(dir.isValid());

    /*JTA_INTCALIB_BIPLANE: 20 numeric tokens; A=(1,-2,-3,4), B=(5,-6,-7,8),
     * origin_B=(9,10,11), axes_B = 12..20.*/
    const QStringList lines = {
        "JTA_INTCALIB_BIPLANE", "1", "2", "3", "4", "5", "6", "7", "8",
        "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
        "20"};
    const CalibrationParseResult bi =
        SessionController::ParseCalibration(write_calibration(dir, lines));
    REQUIRE(bi.ok);
    REQUIRE(bi.kind == CalibrationParseResult::Kind::Biplane);
    REQUIRE_FALSE(bi.calibrated_for_monoplane_viewport);
    REQUIRE(bi.calibrated_for_biplane_viewport);
    REQUIRE(bi.calibration.biplane_calibration == true);
    REQUIRE(bi.calibration.camera_A_principal_.principal_distance_ ==
            Approx(1.0));
    REQUIRE(bi.calibration.camera_A_principal_.principal_x_ == Approx(-2.0));
    REQUIRE(bi.calibration.camera_A_principal_.principal_y_ == Approx(-3.0));
    REQUIRE(bi.calibration.camera_A_principal_.pixel_pitch_ == Approx(4.0));
    REQUIRE(bi.calibration.camera_B_principal_.principal_distance_ ==
            Approx(5.0));
    REQUIRE(bi.calibration.camera_B_principal_.principal_x_ == Approx(-6.0));
    REQUIRE(bi.calibration.camera_B_principal_.principal_y_ == Approx(-7.0));
    REQUIRE(bi.calibration.camera_B_principal_.pixel_pitch_ == Approx(8.0));
    REQUIRE(bi.calibration.origin_B_.v_1_ == Approx(9.0));
    REQUIRE(bi.calibration.origin_B_.v_2_ == Approx(10.0));
    REQUIRE(bi.calibration.origin_B_.v_3_ == Approx(11.0));
    REQUIRE(bi.calibration.axes_B_.A_11_ == Approx(12.0));
    REQUIRE(bi.calibration.axes_B_.A_12_ == Approx(13.0));
    REQUIRE(bi.calibration.axes_B_.A_33_ == Approx(20.0));

    /*Denver "image" code: token 9 is skipped (fx=100, sc=1, cx=200, fy=300,
     * cy=1); no pixel-size error check exists for this branch (preserved).*/
    CalibrationParseResult denver = SessionController::ParseCalibration(
        write_calibration(dir, {"image", "0", "0", "0", "0", "0", "100", "1",
                                "200", "0", "300", "1"}));
    REQUIRE(denver.ok);
    REQUIRE(denver.kind == CalibrationParseResult::Kind::Denver);
    REQUIRE(denver.calibrated_for_monoplane_viewport);
    REQUIRE_FALSE(denver.calibrated_for_biplane_viewport);
    REQUIRE(denver.calibration.biplane_calibration == false);
    REQUIRE(denver.calibration.type_ == "Denver");
    REQUIRE(denver.calibration.camera_A_principal_.fx() == Approx(100.0));
    REQUIRE(denver.calibration.camera_A_principal_.cx() == Approx(200.0));
    REQUIRE(denver.calibration.camera_A_principal_.fy() == Approx(300.0));
}

TEST_CASE("session_controller: monoplane image parse populates frames + storage",
          "[session_controller]") {
    QTemporaryDir dir;
    REQUIRE(dir.isValid());
    const QString f1 = write_png(dir, "frame_one.png", 8, 8);
    const QString f2 = write_png(dir, "frame_two.png", 8, 8);

    SessionController controller;
    std::vector<Frame> frames;
    LocationStorage locations;
    const auto result = controller.ParseImages(
        {f1, f2}, default_params(), frames, locations);

    REQUIRE(result.status == ImageLoadStatus::Completed);
    REQUIRE(result.frame_names == QStringList({"frame_one", "frame_two"}));
    REQUIRE(frames.size() == 2);
    REQUIRE(locations.GetFrameCount() == 2);
    REQUIRE(controller.GetFrameCount() == 2);
    REQUIRE(controller.GetModelCount() == 0);
}

TEST_CASE(
    "session_controller: partial load aborts mid-list and persists the frames "
    "appended so far (goto stop semantics)",
    "[session_controller]") {
    QTemporaryDir dir;
    REQUIRE(dir.isValid());
    const QString ok = write_png(dir, "ok.png", 8, 8);
    const QString bad = write_png(dir, "bad.png", 16, 16);

    SessionController controller;
    std::vector<Frame> frames;
    LocationStorage locations;
    const auto result =
        controller.ParseImages({ok, bad}, default_params(), frames, locations);

    REQUIRE(result.status == ImageLoadStatus::SizeMismatchAborted);
    /*The first frame was appended and sized before the mismatch; the second
     * was not -- the goto stop partial-load semantics, pinned.*/
    REQUIRE(result.frame_names == QStringList({"ok"}));
    REQUIRE(frames.size() == 1);
    REQUIRE(locations.GetFrameCount() == 1);
    REQUIRE(controller.GetFrameCount() == 1);

    /*Loading the same list again reproduces today's behavior: the appended
     * "ok" frame is now part of the loaded set, and the second file still
     * mismatches (16x16 vs 8x8).*/
    const auto again =
        controller.ParseImages({ok, bad}, default_params(), frames, locations);
    REQUIRE(again.status == ImageLoadStatus::SizeMismatchAborted);
    REQUIRE(again.frame_names == QStringList({"ok"}));
    REQUIRE(frames.size() == 2);
    REQUIRE(locations.GetFrameCount() == 2);
}

TEST_CASE("session_controller: empty/cancelled image selection changes nothing",
          "[session_controller]") {
    SessionController controller;
    std::vector<Frame> frames;
    LocationStorage locations;
    const auto result =
        controller.ParseImages({}, default_params(), frames, locations);
    REQUIRE(result.status == ImageLoadStatus::Completed);
    REQUIRE(result.frame_names.isEmpty());
    REQUIRE(frames.empty());
    REQUIRE(locations.GetFrameCount() == 0);
    REQUIRE(controller.GetFrameCount() == 0);
}

TEST_CASE("session_controller: biplane image parse (happy, count mismatch, "
          "B-size abort)",
          "[session_controller]") {
    QTemporaryDir dir;
    REQUIRE(dir.isValid());
    const QString a1 = write_png(dir, "a1.png", 8, 8);
    const QString a2 = write_png(dir, "a2.png", 8, 8);
    const QString b1 = write_png(dir, "b1.png", 8, 8);
    const QString b2_big = write_png(dir, "b2_big.png", 16, 16);

    /*Happy path: A and B lists of the same length and sizes.*/
    {
        SessionController controller;
        std::vector<Frame> frames_a, frames_b;
        LocationStorage locations;
        const auto result = controller.ParseBiplaneImages(
            {a1, a2}, {b1, b1}, default_params(), frames_a, frames_b,
            locations);
        REQUIRE(result.status == ImageLoadStatus::Completed);
        REQUIRE(result.frame_names ==
                QStringList({"A: a1\nB: b1", "A: a2\nB: b1"}));
        REQUIRE(frames_a.size() == 2);
        REQUIRE(frames_b.size() == 2);
        REQUIRE(locations.GetFrameCount() == 2);
        REQUIRE(controller.GetFrameCount() == 2);
    }

    /*A/B count mismatch: nothing is appended, storage untouched.*/
    {
        SessionController controller;
        std::vector<Frame> frames_a, frames_b;
        LocationStorage locations;
        const auto result = controller.ParseBiplaneImages(
            {a1, a2}, {b1}, default_params(), frames_a, frames_b, locations);
        REQUIRE(result.status == ImageLoadStatus::CameraCountMismatch);
        REQUIRE(result.frame_names.isEmpty());
        REQUIRE(frames_a.empty());
        REQUIRE(frames_b.empty());
        REQUIRE(locations.GetFrameCount() == 0);
    }

    /*B-size mismatch on the second pair: both lists keep the first pair
     * (per-iteration atomicity + goto stop_biplane semantics).*/
    {
        SessionController controller;
        std::vector<Frame> frames_a, frames_b;
        LocationStorage locations;
        const auto result = controller.ParseBiplaneImages(
            {a1, a2}, {b1, b2_big}, default_params(), frames_a, frames_b,
            locations);
        REQUIRE(result.status == ImageLoadStatus::SizeMismatchAborted);
        REQUIRE(result.frame_names == QStringList({"A: a1\nB: b1"}));
        REQUIRE(frames_a.size() == 1);
        REQUIRE(frames_b.size() == 1);
        REQUIRE(locations.GetFrameCount() == 1);
    }
}

TEST_CASE("session_controller: model paths parse and populate the dataset",
          "[session_controller]") {
    /*Missing STL files are the point: Model's STL parse degrades to
     * initialized_correctly_ == false without crashing (the slot's warning
     * path), so no STL fixtures are needed.*/
    const QStringList paths = {"no_such_femur.stl", "no_such_tibia.stl"};

    const std::vector<jta::ParsedModel> parsed =
        SessionController::ParseModels(paths);
    REQUIRE(parsed.size() == 2);
    REQUIRE(parsed[0].file_path == "no_such_femur.stl");
    REQUIRE(parsed[0].base_name == "no_such_femur");
    REQUIRE(parsed[1].file_path == "no_such_tibia.stl");
    REQUIRE(parsed[1].base_name == "no_such_tibia");

    SessionController controller;
    std::vector<Model> models;
    LocationStorage locations;
    const Calibration calibration(CameraCalibration(1198, 0, 0, 0.373));
    /*Unique display names come from the view's AppendModels dedup; here the
     * dedup output is fed in directly (R5 boundary).*/
    controller.PopulateModels(
        parsed, {"femur", "tibia"}, calibration, models, locations);

    REQUIRE(models.size() == 2);
    REQUIRE(models[0].file_location_ == "no_such_femur.stl");
    REQUIRE(models[0].model_name_ == "femur");
    REQUIRE(models[0].model_type_ == "BLANK");
    REQUIRE_FALSE(models[0].initialized_correctly_);
    REQUIRE(models[1].model_name_ == "tibia");
    /*With no frames loaded, LocationStorage::GetModelCount() is 0 -- the
     * models live in the no-image vector, which GetModelCount() does not
     * count (existing LocationStorage semantics, preserved). The sizing is
     * pinned via the GetPose(-1, ...) boundary fallback instead.*/
    REQUIRE(locations.GetModelCount() == 0);
    REQUIRE(locations.GetPose(-1, 0).z == Approx(-0.25 * 1198.0 / 0.373));
    REQUIRE(locations.GetPose(-1, 1).z == Approx(-0.25 * 1198.0 / 0.373));
    REQUIRE(controller.GetModelCount() == 2);

    /*Loading the same list twice appends two more rows (duplicate names go
     * through the view's dedup, which is not the controller's job).*/
    controller.PopulateModels(
        parsed, {"femur", "tibia"}, calibration, models, locations);
    REQUIRE(models.size() == 4);
    REQUIRE(locations.GetModelCount() == 0);
    REQUIRE(locations.GetPose(-1, 3).z == Approx(-0.25 * 1198.0 / 0.373));
    REQUIRE(controller.GetModelCount() == 4);
}

TEST_CASE(
    "session_controller: pose-matrix dimensions stay consistent with "
    "frame/model counts after load (PoseMatrixDimensionMismatch invariant)",
    "[session_controller]") {
    QTemporaryDir dir;
    REQUIRE(dir.isValid());
    const QString f1 = write_png(dir, "f1.png", 8, 8);
    const QString f2 = write_png(dir, "f2.png", 8, 8);

    SessionController controller;
    std::vector<Frame> frames;
    std::vector<Model> models;
    LocationStorage locations;
    const Calibration calibration(CameraCalibration(1198, 0, 0, 0.373));

    const auto images = controller.ParseImages(
        {f1, f2}, default_params(), frames, locations);
    REQUIRE(images.status == ImageLoadStatus::Completed);
    controller.PopulateModels({{"no_such_a.stl", "no_such_a"},
                               {"no_such_b.stl", "no_such_b"}},
                              {"a", "b"}, calibration, models, locations);

    /*The invariant behind PoseMatrixDimensionMismatch: storage dims ==
     * dataset counts, and the controller's mirrors agree.*/
    REQUIRE(locations.GetFrameCount() == static_cast<int>(frames.size()));
    REQUIRE(locations.GetModelCount() == static_cast<int>(models.size()));
    REQUIRE(controller.GetFrameCount() == 2);
    REQUIRE(controller.GetModelCount() == 2);

    /*UF default pose sizing: (0, 0, -0.25 * principal_distance / pixel_pitch)
     * for every (frame, model) cell.*/
    const Point6D pose = locations.GetPose(1, 1);
    REQUIRE(pose.x == Approx(0.0));
    REQUIRE(pose.y == Approx(0.0));
    REQUIRE(pose.z == Approx(-0.25 * 1198.0 / 0.373));
}

TEST_CASE("session_controller: camera radio enable/disable decision matrix",
          "[session_controller]") {
    /*CalibrationLoaded: monoplane disables both radios (A stays checked);
     * biplane enables both; neither flag (invalid load) is not applied.*/
    auto actions = SessionController::DecideCameraRadios(
        CameraRadioEvent::CalibrationLoaded, true, false);
    REQUIRE_FALSE(actions.enable_camera_a);
    REQUIRE_FALSE(actions.enable_camera_b);

    actions = SessionController::DecideCameraRadios(
        CameraRadioEvent::CalibrationLoaded, false, true);
    REQUIRE(actions.enable_camera_a);
    REQUIRE(actions.enable_camera_b);

    actions = SessionController::DecideCameraRadios(
        CameraRadioEvent::CalibrationLoaded, false, false);
    REQUIRE_FALSE(actions.enable_camera_a);
    REQUIRE_FALSE(actions.enable_camera_b);

    /*SwitchToCameraA: A disabled, B enabled under the biplane guard; a
     * no-op for monoplane (the caller does not apply it, as today).*/
    actions = SessionController::DecideCameraRadios(
        CameraRadioEvent::SwitchToCameraA, false, true);
    REQUIRE_FALSE(actions.enable_camera_a);
    REQUIRE(actions.enable_camera_b);

    actions = SessionController::DecideCameraRadios(
        CameraRadioEvent::SwitchToCameraA, true, false);
    REQUIRE_FALSE(actions.enable_camera_a);
    REQUIRE_FALSE(actions.enable_camera_b);

    /*SwitchToCameraB: B disabled, A enabled unconditionally.*/
    actions = SessionController::DecideCameraRadios(
        CameraRadioEvent::SwitchToCameraB, false, true);
    REQUIRE(actions.enable_camera_a);
    REQUIRE_FALSE(actions.enable_camera_b);

    actions = SessionController::DecideCameraRadios(
        CameraRadioEvent::SwitchToCameraB, true, false);
    REQUIRE(actions.enable_camera_a);
    REQUIRE_FALSE(actions.enable_camera_b);
}

TEST_CASE("session_controller: active-camera mirror tracks the radios",
          "[session_controller]") {
    SessionController controller;
    /*Default: camera A (the radio checked after any calibration load).*/
    REQUIRE(controller.GetActiveCamera() == ActiveCamera::CameraA);

    controller.SetActiveCamera(ActiveCamera::CameraB);
    REQUIRE(controller.GetActiveCamera() == ActiveCamera::CameraB);

    controller.SetActiveCamera(ActiveCamera::CameraA);
    REQUIRE(controller.GetActiveCamera() == ActiveCamera::CameraA);
}
