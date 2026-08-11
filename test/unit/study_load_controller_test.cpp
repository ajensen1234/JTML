// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U7: StudyLoadController tests (R11 + R13). Deterministic Catch2
// headless tests that direct-compile the shared study-load controller + the
// real SessionController + LocationStorage/Model/stl_reader/ModelListBuilder
// services + the direct-compiled ModelListModel (the dedup seam BOTH views
// wire — the widgets' model_list_model_ and StudyBridge's model_list_model_)
// against the headless Frame twin (test/unit/frame_headless.cpp, pure
// OpenCV).
//
// Pins (plan 006 U7 test scenarios):
//  - (a) happy path: full study load (calibration -> images -> models) →
//    containers populated, counts set (the result facts + the shared
//    SessionController mirrors), session mirror updated (active camera A);
//  - (b) dataset replace: wiping the caller-owned containers (the QML
//    clearDataset tail) + reloading a second study keeps the calibration;
//  - (c) calibration one-use: a second calibration load is rejected
//    (CalibrationAlreadyLoaded — widgets parity), the calibration container
//    is not overwritten;
//  - (d) error paths: missing calibration file / bad calibration → clear
//    error status with no dataset writes; image size mismatch → the partial
//    load persists (goto stop semantics) with a clear status; biplane count
//    mismatch → nothing appended;
//  - (e) the injected run-in-flight probe (L17) rejects all three loads
//    before any parse/populate (nothing changed, dedup seam untouched).
//
// Fixtures: image fixtures are temp PNGs written at runtime (cv::imwrite);
// calibration fixtures are temp files (the golden test/golden/calibration.txt
// values); models are missing STL files — Model's STL parse degrades to
// initialized_correctly_ == false without crashing, exactly like the
// jtml.session_controller tests.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <QFile>
#include <QString>
#include <QStringList>
#include <QTemporaryDir>
#include <QVector>

#include <opencv2/imgcodecs.hpp>

#include "services/calibration.h"
#include "services/location_storage.h"
#include "services/model.h"
#include "services/session_controller.h"
#include "services/study_load_controller.h"
#include "view/model_list_model.h"

using Catch::Approx;

namespace {

using jta::ActiveCamera;
using jta::CalibrationParseResult;
using jta::ImageLoadParams;
using jta::StudyLoadStatus;

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

/*The golden calibration values (test/golden/calibration.txt): JT_INTCALIB /
 * 1198 / 0 / 0 / 0.373.*/
QString write_golden_calibration(const QTemporaryDir& dir,
                                 const QString& name) {
    return write_calibration(
        dir, {name, "1198", "0", "0", "0.373"});
}

/*Write a solid-gray PNG of the given size into dir and return its path.*/
QString write_png(const QTemporaryDir& dir, const QString& name, int w, int h) {
    cv::Mat img(h, w, CV_8UC1, cv::Scalar(64));
    const QString path = dir.filePath(name);
    REQUIRE(cv::imwrite(path.toStdString(), img));
    return path;
}

/*The U7 fixture: the load controller wrapped around its shared
 * SessionController + the caller-owned containers the views would pass
 * (calibration + flags + frames/models/LocationStorage) + the ModelListModel
 * dedup seam (the AppendModels both views wire).*/
struct LoadFixture {
    jta::SessionController session_controller;
    jta::StudyLoadController loads{&session_controller};
    Calibration calibration;
    bool calibrated_mono = false;
    bool calibrated_bi = false;
    std::vector<Frame> frames;
    std::vector<Frame> frames_b;
    std::vector<Model> models;
    LocationStorage locations;
    ModelListModel model_names;

    /*The widgets/StudyBridge dedup seam: ModelListModel::AppendModels.*/
    QVector<QString> AppendModels(const QVector<QString>& base_names) {
        return model_names.AppendModels(base_names);
    }

    /*The Kneel_1 three-action load (golden calibration values + temp PNGs +
     * missing STLs — the session_controller_test recipe).*/
    void LoadStudyOne(const QTemporaryDir& dir, const QString& cal_path) {
        REQUIRE(loads.LoadCalibration(cal_path, calibration, calibrated_mono,
                                      calibrated_bi)
                    .status == StudyLoadStatus::Ok);
        const QString f1 = write_png(dir, "one_a.png", 8, 8);
        const QString f2 = write_png(dir, "one_b.png", 8, 8);
        REQUIRE(loads.LoadImages({f1, f2}, default_params(), frames, locations)
                    .status == StudyLoadStatus::Ok);
        REQUIRE(loads.LoadModels({"no_such_femur.stl", "no_such_tibia.stl"},
                                 calibration, models, locations,
                                 [this](const QVector<QString>& base) {
                                     return AppendModels(base);
                                 })
                    .status == StudyLoadStatus::Ok);
    }
};

}  // namespace

TEST_CASE(
    "study_load_controller: full study load populates the containers, counts "
    "and mirrors",
    "[study_load_controller]") {
    QTemporaryDir dir;
    REQUIRE(dir.isValid());
    LoadFixture f;

    /*Calibration: the caller-owned calibration + flags written; the shared
     * active-camera mirror advances to Camera A (the camera-radio check
     * after any valid calibration, R10).*/
    const jta::StudyCalibrationLoadResult cal =
        f.loads.LoadCalibration(write_golden_calibration(dir, "JT_INTCALIB"),
                                f.calibration, f.calibrated_mono,
                                f.calibrated_bi);
    REQUIRE(cal.status == StudyLoadStatus::Ok);
    REQUIRE(cal.parse.ok);
    REQUIRE(f.calibrated_mono);
    REQUIRE_FALSE(f.calibrated_bi);
    REQUIRE(f.calibration.type_ == "UF");
    REQUIRE(f.calibration.camera_A_principal_.principal_distance_ ==
            Approx(1198.0));
    REQUIRE(f.session_controller.GetActiveCamera() == ActiveCamera::CameraA);

    /*Images: 2 frames appended + storage sized; the result carries the
     * display names + the post-load count facts (the sync-tail surface).*/
    const QString f1 = write_png(dir, "frame_one.png", 8, 8);
    const QString f2 = write_png(dir, "frame_two.png", 8, 8);
    const jta::StudyImageLoadResult images = f.loads.LoadImages(
        {f1, f2}, default_params(), f.frames, f.locations);
    REQUIRE(images.status == StudyLoadStatus::Ok);
    REQUIRE(images.frame_names == QStringList({"frame_one", "frame_two"}));
    REQUIRE(f.frames.size() == 2);
    REQUIRE(f.locations.GetFrameCount() == 2);
    REQUIRE(images.frame_count == 2);
    REQUIRE(images.model_count == 0);
    REQUIRE(f.session_controller.GetFrameCount() == 2);

    /*Models: the dedup seam returns the unique display names (also appended
     * to the view-model); the Model objects + storage sizing land in the
     * caller-owned containers; the count facts + shared mirrors agree.*/
    const jta::StudyModelLoadResult models = f.loads.LoadModels(
        {"no_such_femur.stl", "no_such_tibia.stl"}, f.calibration, f.models,
        f.locations, [&](const QVector<QString>& base) {
            return f.AppendModels(base);
        });
    REQUIRE(models.status == StudyLoadStatus::Ok);
    REQUIRE(models.file_paths ==
            QStringList({"no_such_femur.stl", "no_such_tibia.stl"}));
    REQUIRE(models.unique_names ==
            QStringList({"no_such_femur", "no_such_tibia"}));
    REQUIRE(f.models.size() == 2);
    REQUIRE(f.models[0].model_name_ == "no_such_femur");
    REQUIRE(f.models[0].model_type_ == "BLANK");
    REQUIRE_FALSE(f.models[0].initialized_correctly_);  // missing STL fixture
    REQUIRE(f.model_names.rowCount() == 2);
    REQUIRE(models.model_count == 2);
    REQUIRE(f.session_controller.GetModelCount() == 2);
    /*With 2 frames loaded, the models live in the per-frame pose matrix:
     * LocationStorage::GetModelCount() counts them (unlike the no-frames
     * case, where they live in the no-image vector and are not counted —
     * existing semantics, preserved). The no-image fallback is still sized
     * (the GetPose(-1, ...) boundary, pinned below).*/
    REQUIRE(f.locations.GetModelCount() == 2);
    REQUIRE(f.locations.GetPose(-1, 0).z == Approx(-0.25 * 1198.0 / 0.373));
}

TEST_CASE(
    "study_load_controller: biplane image load (happy path + count mismatch)",
    "[study_load_controller]") {
    QTemporaryDir dir;
    REQUIRE(dir.isValid());
    LoadFixture f;
    const QString a1 = write_png(dir, "a1.png", 8, 8);
    const QString a2 = write_png(dir, "a2.png", 8, 8);
    const QString b1 = write_png(dir, "b1.png", 8, 8);

    /*Happy path: A and B lists of the same length and sizes.*/
    const jta::StudyImageLoadResult bi = f.loads.LoadBiplaneImages(
        {a1, a2}, {b1, b1}, default_params(), f.frames, f.frames_b,
        f.locations);
    REQUIRE(bi.status == StudyLoadStatus::Ok);
    REQUIRE(bi.frame_names ==
            QStringList({"A: a1\nB: b1", "A: a2\nB: b1"}));
    REQUIRE(f.frames.size() == 2);
    REQUIRE(f.frames_b.size() == 2);
    REQUIRE(f.locations.GetFrameCount() == 2);
    REQUIRE(bi.frame_count == 2);

    /*A/B count mismatch: nothing appended, storage untouched, clear status.*/
    LoadFixture g;
    const jta::StudyImageLoadResult mismatch = g.loads.LoadBiplaneImages(
        {a1, a2}, {b1}, default_params(), g.frames, g.frames_b, g.locations);
    REQUIRE(mismatch.status == StudyLoadStatus::CameraCountMismatch);
    REQUIRE(mismatch.frame_names.isEmpty());
    REQUIRE(g.frames.empty());
    REQUIRE(g.frames_b.empty());
    REQUIRE(g.locations.GetFrameCount() == 0);
    REQUIRE(mismatch.frame_count == 0);
}

TEST_CASE(
    "study_load_controller: dataset replace keeps the calibration; one-use "
    "rejects a second calibration load",
    "[study_load_controller]") {
    QTemporaryDir dir;
    REQUIRE(dir.isValid());
    LoadFixture f;
    f.LoadStudyOne(dir, write_golden_calibration(dir, "JT_INTCALIB"));

    /*Calibration one-use per session (widgets parity: the load-calibration
     * button disables after a successful load): a second load is rejected
     * BEFORE parsing and the calibration container is not overwritten.*/
    const jta::StudyCalibrationLoadResult again =
        f.loads.LoadCalibration(
            write_golden_calibration(dir, "OTHER_CAL"), f.calibration,
            f.calibrated_mono, f.calibrated_bi);
    REQUIRE(again.status == StudyLoadStatus::CalibrationAlreadyLoaded);
    REQUIRE(again.parse.error == CalibrationParseResult::Error::None);
    REQUIRE(f.calibrated_mono);  // kept
    REQUIRE(f.calibration.type_ == "UF");
    REQUIRE(f.calibration.camera_A_principal_.principal_distance_ ==
            Approx(1198.0));  // untouched

    /*Dataset replace (the QML clearDataset tail): the view wipes the
     * caller-owned containers + swaps in fresh list models; the calibration
     * + flags are deliberately KEPT (one-use per session).*/
    f.frames.clear();
    f.frames_b.clear();
    f.models.clear();
    f.locations = LocationStorage();
    ModelListModel fresh_names;  // the QML's fresh ModelListModel instance

    /*A second study loads cleanly onto the wiped containers.*/
    const QString s2a = write_png(dir, "two_a.png", 8, 8);
    const QString s2b = write_png(dir, "two_b.png", 8, 8);
    const jta::StudyImageLoadResult images = f.loads.LoadImages(
        {s2a, s2b}, default_params(), f.frames, f.locations);
    REQUIRE(images.status == StudyLoadStatus::Ok);
    REQUIRE(f.frames.size() == 2);
    REQUIRE(f.locations.GetFrameCount() == 2);
    REQUIRE(images.frame_count == 2);

    const jta::StudyModelLoadResult models = f.loads.LoadModels(
        {"no_such_femur.stl"}, f.calibration, f.models, f.locations,
        [&](const QVector<QString>& base) {
            return fresh_names.AppendModels(base);
        });
    REQUIRE(models.status == StudyLoadStatus::Ok);
    REQUIRE(f.models.size() == 1);
    REQUIRE(fresh_names.rowCount() == 1);
    REQUIRE(models.model_count == 1);

    /*The kept calibration is still the FIRST study's (dataset-replace
     * semantics, pinned).*/
    REQUIRE(f.calibrated_mono);
    REQUIRE(f.calibration.camera_A_principal_.principal_distance_ ==
            Approx(1198.0));
}

TEST_CASE(
    "study_load_controller: parse failures surface a clear error status with "
    "no dataset writes",
    "[study_load_controller]") {
    QTemporaryDir dir;
    REQUIRE(dir.isValid());

    /*Missing calibration file: the silent open-guard kind, nothing written.*/
    {
        LoadFixture f;
        const jta::StudyCalibrationLoadResult missing =
            f.loads.LoadCalibration(dir.filePath("does_not_exist.txt"),
                                    f.calibration, f.calibrated_mono,
                                    f.calibrated_bi);
        REQUIRE(missing.status == StudyLoadStatus::CalibrationParseError);
        REQUIRE(missing.parse.error ==
                CalibrationParseResult::Error::FileOpenFailed);
        REQUIRE_FALSE(f.calibrated_mono);
        REQUIRE_FALSE(f.calibrated_bi);
        REQUIRE_FALSE(f.calibration.biplane_calibration);
    }

    /*Pixel size zero: the typed error kind + the widgets' error-path flag
     * writes (both flags false — no-ops here, the widgets' literal
     * behavior).*/
    {
        LoadFixture f;
        const jta::StudyCalibrationLoadResult zero =
            f.loads.LoadCalibration(
                write_calibration(dir, {"JT_INTCALIB", "100", "0", "0", "0"}),
                f.calibration, f.calibrated_mono, f.calibrated_bi);
        REQUIRE(zero.status == StudyLoadStatus::CalibrationParseError);
        REQUIRE(zero.parse.error ==
                CalibrationParseResult::Error::PixelSizeZero);
        REQUIRE_FALSE(f.calibrated_mono);
        REQUIRE_FALSE(f.calibrated_bi);
    }

    /*Invalid code: same surface.*/
    {
        LoadFixture f;
        const jta::StudyCalibrationLoadResult invalid =
            f.loads.LoadCalibration(
                write_calibration(dir, {"NOT_A_CALIBRATION_CODE", "1"}),
                f.calibration, f.calibrated_mono, f.calibrated_bi);
        REQUIRE(invalid.status == StudyLoadStatus::CalibrationParseError);
        REQUIRE(invalid.parse.error ==
                CalibrationParseResult::Error::InvalidCode);
        REQUIRE_FALSE(f.calibrated_mono);
        REQUIRE_FALSE(f.calibrated_bi);
    }

    /*Image size mismatch: the frames appended so far persist (goto stop
     * semantics), the status is unambiguous, the count facts match the
     * partial set.*/
    {
        LoadFixture f;
        const QString ok = write_png(dir, "ok.png", 8, 8);
        const QString bad = write_png(dir, "bad.png", 16, 16);
        const jta::StudyImageLoadResult images = f.loads.LoadImages(
            {ok, bad}, default_params(), f.frames, f.locations);
        REQUIRE(images.status == StudyLoadStatus::SizeMismatchAborted);
        REQUIRE(images.frame_names == QStringList({"ok"}));
        REQUIRE(f.frames.size() == 1);
        REQUIRE(f.locations.GetFrameCount() == 1);
        REQUIRE(images.frame_count == 1);
    }
}

TEST_CASE(
    "study_load_controller: run-in-flight probe rejects every load before "
    "any parse/populate (L17)",
    "[study_load_controller]") {
    QTemporaryDir dir;
    REQUIRE(dir.isValid());
    bool in_flight = true;
    jta::SessionController session_controller;
    jta::StudyLoadController loads(&session_controller,
                                   [&] { return in_flight; });

    Calibration calibration;
    bool calibrated_mono = false;
    bool calibrated_bi = false;
    std::vector<Frame> frames;
    std::vector<Model> models;
    LocationStorage locations;
    ModelListModel model_names;
    bool dedup_called = false;

    /*Calibration: rejected, nothing written.*/
    const jta::StudyCalibrationLoadResult cal =
        loads.LoadCalibration(write_golden_calibration(dir, "JT_INTCALIB"),
                              calibration, calibrated_mono, calibrated_bi);
    REQUIRE(cal.status == StudyLoadStatus::RunInFlight);
    REQUIRE_FALSE(calibrated_mono);
    REQUIRE_FALSE(calibration.biplane_calibration);

    /*Images: rejected, no frames, no storage sizing.*/
    const QString f1 = write_png(dir, "f1.png", 8, 8);
    const jta::StudyImageLoadResult images = loads.LoadImages(
        {f1}, default_params(), frames, locations);
    REQUIRE(images.status == StudyLoadStatus::RunInFlight);
    REQUIRE(frames.empty());
    REQUIRE(locations.GetFrameCount() == 0);

    /*Models: rejected BEFORE the dedup seam is consulted (nothing parsed,
     * nothing appended).*/
    const jta::StudyModelLoadResult models_result = loads.LoadModels(
        {"no_such_femur.stl"}, calibration, models, locations,
        [&](const QVector<QString>& base) {
            dedup_called = true;
            return model_names.AppendModels(base);
        });
    REQUIRE(models_result.status == StudyLoadStatus::RunInFlight);
    REQUIRE_FALSE(dedup_called);
    REQUIRE(models.empty());
    REQUIRE(model_names.rowCount() == 0);

    /*The probe is consulted at each load (lazily): the same controller
     * proceeds once the run is over.*/
    in_flight = false;
    const jta::StudyCalibrationLoadResult later =
        loads.LoadCalibration(write_golden_calibration(dir, "JT_INTCALIB"),
                              calibration, calibrated_mono, calibrated_bi);
    REQUIRE(later.status == StudyLoadStatus::Ok);
    REQUIRE(calibrated_mono);
}
