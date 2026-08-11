// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 005 U4: StudyBridge + delegate-selection contract pins (R3, R17).
// Deterministic Catch2 headless tests that direct-compile the bridge
// (AppBridge/StudyBridge/DelegateSelection/ExperimentalScene) against the
// headless Frame twin (test/unit/frame_headless.cpp, pure OpenCV) + the
// real SessionController + LocationStorage/Model/stl_reader/ModelListBuilder
// services + the direct-compiled FrameListModel/ModelListModel.
//
// Pins (plan 005 U4 test scenarios):
//  - happy path: the three-action load flow (calibration -> images -> models)
//    on the REAL Kneel_1 fixtures populates frames/models/locations/scene
//    with the counts SessionController's own tests establish;
//  - edge: partial-load semantics (a size mismatch keeps the frames appended
//    so far); dataset-replace semantics (clearDataset wipes the dataset but
//    keeps the calibration; reload lands on fresh list models);
//  - edge: empty selection states are well-defined (no crash, -1 primary,
//    empty scene background, out-of-range rows ignored);
//  - error paths: bad calibration files surface the typed error kinds as
//    messageRequested (PixelSizeZero/InvalidCode), FileOpenFailed is silent;
//    models/images before calibration hit the "Load Calibration First!"
//    guard prompt and change nothing;
//  - the delegate selection contract: current frame, multi-select set,
//    primary = first selected, and the SessionState mirror.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <QFile>
#include <QFileInfo>
#include <QSettings>
#include <QStringList>
#include <QTemporaryDir>
#include <QUrl>

#include <opencv2/imgcodecs.hpp>

#include "AppBridge.h"
#include "DelegateSelection.h"
#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "StudyBridge.h"
#include "services/settings_service.h"
#include "view/frame_list_model.h"
#include "view/model_list_model.h"

using Catch::Approx;

namespace {

/*Kneel_1 fixtures (repo-root WORKING_DIRECTORY, like jtml.session_controller
 * and jtml.pose_file_io).*/
const char* kCalibrationPath = "test/golden/calibration.txt";
const QStringList kImagePaths = {
    "example_studies/Kneel_1/AT_K1_V1_0160.tif",
    "example_studies/Kneel_1/AT_K1_V1_0170.tif",
    "example_studies/Kneel_1/AT_K1_V1_0180.tif"};
const QStringList kModelPaths = {"example_studies/Kneel_1/KR_right_6_tib.stl"};

/*Write a solid-gray PNG of the given size into dir and return its path.*/
QString write_png(const QTemporaryDir& dir, const QString& name, int w, int h) {
    cv::Mat img(h, w, CV_8UC1, cv::Scalar(64));
    const QString path = dir.filePath(name);
    REQUIRE(cv::imwrite(path.toStdString(), img));
    return path;
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

/*Records messageRequested emissions (the QML Dialog analog).*/
struct MessageRecorder {
    QStringList titles;
    QStringList texts;
};

void connect_messages(StudyBridge* bridge, MessageRecorder* recorder) {
    QObject::connect(
        bridge, &StudyBridge::messageRequested,
        [recorder](const QString& title, const QString& message) {
            recorder->titles.push_back(title);
            recorder->texts.push_back(message);
        });
}

/*The U4 fixture: a bridge wired to an app-owned scene + hub, with the
 * message recorder connected. The hub's U5 settings surface is injected
 * with an ini-backed SettingsService (isolated to a temp dir — the real
 * registry is never touched by headless tests).*/
struct BridgeFixture {
    QTemporaryDir dir;
    jta::SettingsService settings_service{
        dir.filePath("settings.ini"), QSettings::IniFormat};
    ExperimentalScene scene;
    AppBridge hub{&scene, &settings_service};
    MessageRecorder messages;

    BridgeFixture() {
        REQUIRE(dir.isValid());
        connect_messages(hub.studyBridge(), &messages);
    }

    StudyBridge* bridge() { return hub.studyBridge(); }
    ExperimentalSession* session() { return hub.session(); }
};

/*The Kneel_1 three-action load (calibration -> images -> models).*/
void load_kneel_1(StudyBridge* bridge) {
    bridge->loadCalibration(kCalibrationPath);
    bridge->loadImages(kImagePaths);
    bridge->loadModels(kModelPaths);
}

}  // namespace

TEST_CASE("study_bridge: Kneel_1 three-action load populates the dataset",
          "[study_bridge]") {
    BridgeFixture f;
    StudyBridge* bridge = f.bridge();

    REQUIRE_FALSE(bridge->hasCalibration());
    REQUIRE_FALSE(bridge->hasDataset());
    REQUIRE(bridge->frameCount() == 0);
    REQUIRE(bridge->modelCount() == 0);
    REQUIRE(f.scene.backgroundImage().empty());
    REQUIRE(f.scene.models().empty());

    /*Calibration (the golden fixture: JT_INTCALIB monoplane).*/
    bridge->loadCalibration(kCalibrationPath);
    REQUIRE(bridge->hasCalibration());
    REQUIRE(bridge->calibratedForMonoplane());
    REQUIRE_FALSE(bridge->calibratedForBiplane());
    REQUIRE(f.session()->calibration_file.type_ == "UF");
    REQUIRE(f.messages.titles.isEmpty());  // no error surfaced
    REQUIRE(f.scene.focalLengthPx() == Approx(1198.0));  // golden principal distance

    /*Images: 3 frames, first frame default-selected, scene background +
     * camera set from frame 0.*/
    bridge->loadImages(kImagePaths);
    REQUIRE(bridge->frameCount() == 3);
    REQUIRE(bridge->frameListModel()->rowCount() == 3);
    REQUIRE(bridge->currentFrame() == 0);
    REQUIRE(f.hub.frameCount() == 3);
    REQUIRE(f.session()->loaded_frames.size() == 3);
    REQUIRE(f.session()->model_locations.GetFrameCount() == 3);
    REQUIRE_FALSE(f.scene.backgroundImage().empty());
    REQUIRE(f.scene.cameraViewAngle() > 0.0);
    REQUIRE(f.scene.focalLengthPx() == Approx(1198.0));
    REQUIRE(bridge->frameListModel()
                ->data(bridge->frameListModel()->index(0, 0))
                .toString() == "AT_K1_V1_0160");

    /*Models: 1 STL, scene model at the stored default pose.*/
    bridge->loadModels(kModelPaths);
    REQUIRE(bridge->modelCount() == 1);
    REQUIRE(bridge->modelListModel()->rowCount() == 1);
    REQUIRE(f.hub.modelCount() == 1);
    REQUIRE(f.session()->loaded_models.size() == 1);
    REQUIRE(f.session()->loaded_models[0].initialized_correctly_);
    REQUIRE(f.session()->model_locations.GetModelCount() == 1);
    REQUIRE(f.scene.models().size() == 1);
    REQUIRE(f.scene.models()[0].name == "KR_right_6_tib");
    REQUIRE(f.scene.models()[0].pose.z ==
            Approx(-0.25 * 1198.0 / 0.373));  // LoadNewModel default pose
    REQUIRE(f.messages.titles.isEmpty());

    /*SessionState mirror (the widgets SyncSessionState tail).*/
    REQUIRE(f.session()->session_state.GetFrameCount() == 3);
    REQUIRE(f.session()->session_state.GetModelCount() == 1);
    REQUIRE(f.session()->session_state.GetCurrentFrame() == 0);
    REQUIRE(f.session()->session_state.GetSelectedModels().empty());
    REQUIRE(f.session()->session_state.GetPrimaryModelIndex() == -1);
}

TEST_CASE("study_bridge: calibration error mapping (typed, widgets texts)",
          "[study_bridge]") {
    BridgeFixture f;
    StudyBridge* bridge = f.bridge();
    QTemporaryDir dir;
    REQUIRE(dir.isValid());

    /*PixelSizeZero -> message, nothing calibrated.*/
    bridge->loadCalibration(write_calibration(
        dir, {"JT_INTCALIB", "100", "0", "0", "0"}));
    REQUIRE_FALSE(bridge->hasCalibration());
    REQUIRE(f.messages.texts.size() == 1);
    REQUIRE(f.messages.texts[0].contains("0"));

    /*InvalidCode -> message, nothing calibrated.*/
    bridge->loadCalibration(write_calibration(dir, {"GARBAGE"}));
    REQUIRE_FALSE(bridge->hasCalibration());
    REQUIRE(f.messages.texts.size() == 2);
    REQUIRE(f.messages.texts[1] == "Invalid Configuration File!");

    /*FileOpenFailed -> silent, state unchanged (the widgets open guard).*/
    bridge->loadCalibration(dir.filePath("nonexistent.txt"));
    REQUIRE_FALSE(bridge->hasCalibration());
    REQUIRE(f.messages.texts.size() == 2);  // no new message
}

TEST_CASE("study_bridge: models/images before calibration hit the guard",
          "[study_bridge]") {
    BridgeFixture f;
    StudyBridge* bridge = f.bridge();

    bridge->loadModels(kModelPaths);
    REQUIRE(bridge->modelCount() == 0);
    REQUIRE(f.session()->loaded_models.empty());
    REQUIRE(f.messages.texts.size() == 1);
    REQUIRE(f.messages.texts[0] == "Load Calibration First!");

    bridge->loadImages(kImagePaths);
    REQUIRE(bridge->frameCount() == 0);
    REQUIRE(f.session()->loaded_frames.empty());
    REQUIRE(f.scene.backgroundImage().empty());
    REQUIRE(f.messages.texts.size() == 2);
    REQUIRE(f.messages.texts[1] == "Load Calibration First!");

    /*Calibration is still loadable afterwards, and the load then works.*/
    bridge->loadCalibration(kCalibrationPath);
    REQUIRE(bridge->hasCalibration());
    bridge->loadModels(kModelPaths);
    REQUIRE(bridge->modelCount() == 1);
}

TEST_CASE("study_bridge: partial-load semantics (size mismatch keeps the "
          "frames appended so far)", "[study_bridge]") {
    BridgeFixture f;
    StudyBridge* bridge = f.bridge();
    bridge->loadCalibration(kCalibrationPath);

    QTemporaryDir dir;
    REQUIRE(dir.isValid());
    /*goto-stop semantics: two same-size frames append, the mismatched third
     * aborts the list — the appended frames persist (SessionController
     * verbatim).*/
    bridge->loadImages({write_png(dir, "a.png", 64, 64),
                        write_png(dir, "b.png", 64, 64),
                        write_png(dir, "c.png", 32, 32)});
    REQUIRE(bridge->frameCount() == 2);
    REQUIRE(bridge->currentFrame() == 0);
    REQUIRE(f.messages.texts.size() == 1);
    REQUIRE(f.messages.texts[0] == "Images Loaded Must Be The Same Size!");
}

TEST_CASE("study_bridge: dataset replace semantics (confirm tail + reload)",
          "[study_bridge]") {
    BridgeFixture f;
    StudyBridge* bridge = f.bridge();
    load_kneel_1(bridge);
    REQUIRE(bridge->frameCount() == 3);
    REQUIRE(bridge->modelCount() == 1);

    /*The QML replace-confirm calls clearDataset() before re-loading. The
     * dataset (frames/models/locations/lists/scene/selection) is wiped; the
     * calibration is deliberately KEPT (one-use per session).*/
    FrameListModel* old_frame_model = bridge->frameListModel();
    ModelListModel* old_model_model = bridge->modelListModel();
    bridge->clearDataset();
    REQUIRE(bridge->frameCount() == 0);
    REQUIRE(bridge->modelCount() == 0);
    REQUIRE(bridge->currentFrame() == -1);
    REQUIRE(bridge->primaryModelIndex() == -1);
    REQUIRE(bridge->hasCalibration());  // kept
    REQUIRE(f.session()->loaded_frames.empty());
    REQUIRE(f.session()->loaded_models.empty());
    REQUIRE(f.session()->model_locations.GetFrameCount() == 0);
    REQUIRE(f.session()->session_state.GetFrameCount() == 0);
    REQUIRE(f.session()->session_state.GetCurrentFrame() == -1);
    REQUIRE(f.scene.models().empty());
    REQUIRE(f.scene.backgroundImage().empty());
    /*Fresh list model instances (the models have no reset API).*/
    REQUIRE(bridge->frameListModel() != old_frame_model);
    REQUIRE(bridge->modelListModel() != old_model_model);

    /*A fresh study loads cleanly onto the fresh instances.*/
    bridge->loadImages(kImagePaths);
    REQUIRE(bridge->frameCount() == 3);
    REQUIRE(bridge->currentFrame() == 0);
    bridge->loadModels(kModelPaths);
    REQUIRE(bridge->modelCount() == 1);
    REQUIRE(f.scene.models().size() == 1);

    /*Documented append parity (the widgets behavior): without the confirm
     * flow, loadImages appends — QML guarantees the clear first.*/
    bridge->loadImages(kImagePaths);
    REQUIRE(bridge->frameCount() == 6);
    REQUIRE(bridge->currentFrame() == 0);
}

TEST_CASE("delegate_selection: contract pins (current frame, multi-select, "
          "primary = first selected, empty states)", "[delegate_selection]") {
    /*Pure helper: well-defined empty states.*/
    DelegateSelection sel;
    REQUIRE(sel.GetCurrentFrame() == -1);
    REQUIRE(sel.GetSelectedModelCount() == 0);
    REQUIRE(sel.GetPrimaryModelIndex() == -1);
    REQUIRE_FALSE(sel.HasModelSelection());
    REQUIRE(sel.GetSelectedModelRows().empty());

    sel.SetCurrentFrame(2);
    REQUIRE(sel.GetCurrentFrame() == 2);

    /*Multi-select set with primary = first selected row (ascending).*/
    sel.SetModelSelected(3, true);
    sel.SetModelSelected(1, true);
    sel.ToggleModel(0);
    REQUIRE(sel.GetSelectedModelCount() == 3);
    REQUIRE(sel.GetSelectedModelRows() == std::vector<int>({0, 1, 3}));
    REQUIRE(sel.GetPrimaryModelIndex() == 0);
    REQUIRE(sel.IsModelSelected(1));
    REQUIRE_FALSE(sel.IsModelSelected(2));

    /*Unselecting the primary promotes the next selected row.*/
    sel.ToggleModel(0);
    REQUIRE(sel.GetPrimaryModelIndex() == 1);
    sel.SetModelSelected(3, false);
    REQUIRE(sel.GetPrimaryModelIndex() == 1);

    sel.ClearModelSelection();
    REQUIRE(sel.GetSelectedModelCount() == 0);
    REQUIRE(sel.GetPrimaryModelIndex() == -1);
    REQUIRE(sel.GetCurrentFrame() == 2);  // frame state untouched
}

TEST_CASE("study_bridge: QML FileDialog file:// URLs normalize to local "
          "paths", "[study_bridge]") {
    BridgeFixture f;
    StudyBridge* bridge = f.bridge();

    /*The QML FileDialog yields absolute file:// URLs (the widgets
     * QFileDialog returned plain paths); the bridge normalizes them for the
     * seams. (Absolute inputs: QUrl::fromLocalFile on a relative path would
     * produce a root-relative URL — real dialogs never do that.)*/
    const auto to_file_url = [](const QString& path) {
        return QUrl::fromLocalFile(QFileInfo(path).absoluteFilePath())
            .toString();
    };
    bridge->loadCalibration(to_file_url(kCalibrationPath));
    REQUIRE(bridge->hasCalibration());

    QStringList url_images;
    for (const QString& path : kImagePaths) {
        url_images.push_back(to_file_url(path));
    }
    bridge->loadImages(url_images);
    REQUIRE(bridge->frameCount() == 3);
    REQUIRE(bridge->frameListModel()
                ->data(bridge->frameListModel()->index(0, 0))
                .toString() == "AT_K1_V1_0160");

    QStringList url_models;
    for (const QString& path : kModelPaths) {
        url_models.push_back(to_file_url(path));
    }
    bridge->loadModels(url_models);
    REQUIRE(bridge->modelCount() == 1);
    REQUIRE(f.session()->loaded_models[0].initialized_correctly_);
    REQUIRE(f.scene.models().size() == 1);
}

TEST_CASE("study_bridge: selection contract through the bridge + "
          "SessionState mirror", "[study_bridge]") {
    BridgeFixture f;
    StudyBridge* bridge = f.bridge();
    load_kneel_1(bridge);

    /*Empty selection states are well-defined (render-without-selection).*/
    REQUIRE(bridge->selectedModelCount() == 0);
    REQUIRE(bridge->primaryModelIndex() == -1);
    REQUIRE(bridge->selectedModels().isEmpty());
    REQUIRE(bridge->isModelSelected(0) == false);

    /*Out-of-range rows are ignored (the bridge guard).*/
    bridge->toggleModelSelected(-1);
    bridge->toggleModelSelected(99);
    REQUIRE(bridge->selectedModelCount() == 0);

    /*Multi-select with primary = first selected (widgets rule).*/
    bridge->toggleModelSelected(0);
    REQUIRE(bridge->selectedModelCount() == 1);
    REQUIRE(bridge->primaryModelIndex() == 0);
    REQUIRE(bridge->selectedModels().size() == 1);
    REQUIRE(bridge->selectedModels()[0].toInt() == 0);
    REQUIRE(f.session()->session_state.GetSelectedModels() ==
            std::vector<int>({0}));
    REQUIRE(f.session()->session_state.GetPrimaryModelIndex() == 0);

    /*A second model (same STL re-loaded: ModelListBuilder dedup names it
     * "KR_right_6_tib(2)" — two models from one file).*/
    bridge->loadModels(kModelPaths);
    REQUIRE(bridge->modelCount() == 2);
    bridge->toggleModelSelected(1);
    REQUIRE(bridge->selectedModelCount() == 2);
    REQUIRE(bridge->primaryModelIndex() == 0);  // first selected stays primary
    REQUIRE(bridge->selectedModels() == QVariantList{0, 1});
    REQUIRE(f.session()->session_state.GetSelectedModels() ==
            std::vector<int>({0, 1}));

    /*Unselecting the primary promotes the next selected row.*/
    bridge->toggleModelSelected(0);
    REQUIRE(bridge->selectedModelCount() == 1);
    REQUIRE(bridge->primaryModelIndex() == 1);
    REQUIRE(f.session()->session_state.GetPrimaryModelIndex() == 1);

    /*Frame selection: current index drives the scene background + mirror;
     * -1 (no current frame) is a well-defined empty state.*/
    bridge->setCurrentFrame(2);
    REQUIRE(bridge->currentFrame() == 2);
    REQUIRE(f.session()->session_state.GetCurrentFrame() == 2);
    REQUIRE_FALSE(f.scene.backgroundImage().empty());
    bridge->setCurrentFrame(-1);
    REQUIRE(bridge->currentFrame() == -1);
    REQUIRE(f.session()->session_state.GetCurrentFrame() == -1);
    REQUIRE(f.scene.backgroundImage().empty());

    /*clearModelSelection leaves the frame selection alone.*/
    bridge->toggleModelSelected(1);
    bridge->clearModelSelection();
    REQUIRE(bridge->selectedModelCount() == 0);
    REQUIRE(bridge->primaryModelIndex() == -1);
    REQUIRE(bridge->currentFrame() == -1);
    REQUIRE(f.session()->session_state.GetSelectedModels().empty());
}
