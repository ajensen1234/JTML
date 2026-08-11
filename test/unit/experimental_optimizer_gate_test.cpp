// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 005 U6 / 006 U5: OptimizerBridge QML-policy pins (R6, R7, R13).
// Deterministic Catch2 headless tests that direct-compile the bridge
// (OptimizerBridge + AppBridge/StudyBridge/SettingsBridge/DelegateSelection/
// ExperimentalScene) and link jtml_coordinator for the REAL seams
// (OptimizerRunController + OptimizerManager symbols) — but never touch the
// GPU: every rejection happens BEFORE any driver/manager/thread is created,
// so run()'s rejection paths are headless-safe.
//
// The shared-core cases (gate evaluation, run-state machine, progress
// mapping, OptimizedFrame handling) moved to
// test/unit/optimizer_run_controller_core_test.cpp +
// test/lifecycle/optimizer_run_controller_test.cpp (plan 006 U5 port).
// What stays here is the BRIDGE policy surface: the SingleModelOnly
// pre-check (after the shared gate's first guard — a QML policy, NOT in
// the controller), the run() rejection message paths against the real
// study fixture (state untouched, no driver created), the stop() no-op,
// and the buildGateInput assembly (previous == current, H2).

#include <catch2/catch_test_macros.hpp>

#include <QFile>
#include <QFileInfo>
#include <QSettings>
#include <QStringList>
#include <QTemporaryDir>
#include <QUrl>

#include <opencv2/imgcodecs.hpp>

#include "AppBridge.h"
#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "OptimizerBridge.h"
#include "StudyBridge.h"
#include "services/settings_service.h"
#include "view/frame_list_model.h"
#include "view/model_list_model.h"

using jta::OptimizeIntentController;

namespace {

/*Kneel_1 fixtures (repo-root WORKING_DIRECTORY, like jtml.session_controller
 * and jtml.experimental_selection).*/
const char* kCalibrationPath = "test/golden/calibration.txt";
const QStringList kImagePaths = {
    "example_studies/Kneel_1/AT_K1_V1_0160.tif",
    "example_studies/Kneel_1/AT_K1_V1_0170.tif",
    "example_studies/Kneel_1/AT_K1_V1_0180.tif"};
const QStringList kModelPaths = {
    "example_studies/Kneel_1/KR_right_6_tib.stl",
    "example_studies/Kneel_1/KR_right_7_fem.stl"};

/*The core gate test's runnable input shape (shared by the bridge gate).*/
jta::OptimizerRunControllerCore::GateInput RunnableInput() {
    jta::OptimizerRunControllerCore::GateInput in;
    in.selected_model_rows = {0};
    in.current_frame = 0;
    in.frame_count = 3;
    in.model_current_index = 0;
    in.model_count = 2;
    in.pose_frame_count = 3;
    in.pose_model_count = 2;
    return in;
}

/*Records messageRequested emissions (the single QML Dialog analog).*/
struct MessageRecorder {
    QStringList titles;
    QStringList texts;
};

void connect_messages(OptimizerBridge* bridge, MessageRecorder* recorder) {
    QObject::connect(
        bridge, &OptimizerBridge::messageRequested,
        [recorder](const QString& title, const QString& message) {
            recorder->titles.push_back(title);
            recorder->texts.push_back(message);
        });
}

/*The U6 fixture: a bridge wired to an app-owned scene + hub with an
 * ini-backed SettingsService (isolated — the real registry is never touched
 * by headless tests).*/
struct BridgeFixture {
    QTemporaryDir dir;
    jta::SettingsService settings_service{
        dir.filePath("settings.ini"), QSettings::IniFormat};
    ExperimentalScene scene;
    AppBridge hub{&scene, &settings_service};
    MessageRecorder messages;

    BridgeFixture() {
        REQUIRE(dir.isValid());
        connect_messages(hub.optimizerBridge(), &messages);
    }

    StudyBridge* study() { return hub.studyBridge(); }
    ExperimentalSession* session() { return hub.session(); }
    OptimizerBridge* bridge() { return hub.optimizerBridge(); }

    /*The Kneel_1 three-action load + a runnable selection (frame 0, model 0).*/
    void loadWithSelection() {
        study()->loadCalibration(kCalibrationPath);
        study()->loadImages(kImagePaths);
        study()->loadModels(kModelPaths);
        study()->setCurrentFrame(0);
        study()->toggleModelSelected(0);
    }
};

}  // namespace

TEST_CASE("optimizer_bridge: multi-mode (multi-select) rejects — v1 "
          "single-model rule",
          "[optimizer_bridge]") {
    /*The shared core itself Oks multi-select (primary = first); the bridge
     * adds the v1 single-model-mode rejection on top (plan 005 U6 review
     * fix — pose ops are primary-model-only in v1). A bridge policy, NOT in
     * the controller (widgets supports multi-run directives).*/
    auto in = RunnableInput();
    in.selected_model_rows = {0, 1};
    REQUIRE(OptimizerBridge::EvaluateGate(in).status ==
            OptimizerBridge::GateStatus::SingleModelOnly);
}

TEST_CASE("optimizer_bridge: single-select runnable input passes the "
          "bridge gate",
          "[optimizer_bridge]") {
    /*The shared gate passes + the v1 rule passes: Ok with the primary +
     * frame packed through.*/
    const auto result = OptimizerBridge::EvaluateGate(RunnableInput());
    REQUIRE(result.status == OptimizerBridge::GateStatus::Ok);
    REQUIRE(result.intent.status == OptimizeIntentController::Status::Ok);
    REQUIRE(result.intent.primary_model_index == 0);
    REQUIRE(result.intent.current_frame == 0);
}

TEST_CASE("optimizer_bridge: buildGateInput mirrors the study state",
          "[optimizer_bridge]") {
    /*Integration: after the real Kneel_1 load + a runnable selection, the
     * bridge's gate Input assembly matches the core-test input shape for
     * the same state (frames, models, selection rows, pose-matrix counts);
     * previous == current by construction (H2).*/
    BridgeFixture f;
    f.loadWithSelection();

    const auto in = f.bridge()->buildGateInput();
    REQUIRE(in.selected_model_rows == std::vector<int>{0});
    REQUIRE(in.current_frame == 0);
    REQUIRE(in.frame_count == 3);
    REQUIRE(in.model_current_index == 0);
    REQUIRE(in.model_count == 2);
    REQUIRE(in.pose_frame_count == 3);
    REQUIRE(in.pose_model_count == 2);
    REQUIRE(
        jta::OptimizerRunControllerCore::BuildGateInput(in)
            .previous_frame_index == in.current_frame);

    /*And the assembled input passes the bridge gate (the runnable path).*/
    const auto result = OptimizerBridge::EvaluateGate(in);
    REQUIRE(result.status == OptimizerBridge::GateStatus::Ok);
    REQUIRE(result.intent.primary_model_index == 0);
}

TEST_CASE("optimizer_bridge: run() with no selection rejects headlessly, "
          "state stays idle",
          "[optimizer_bridge]") {
    /*Error path (plan 005 U6 scenario b): no selection -> typed message,
     * no driver/thread starts (the rejection happens before the controller
     * creates anything), state machine untouched and re-runnable.*/
    BridgeFixture f;
    f.study()->loadCalibration(kCalibrationPath);
    f.study()->loadImages(kImagePaths);
    f.study()->loadModels(kModelPaths);
    f.study()->setCurrentFrame(0);
    /*No model selection.*/

    f.bridge()->run();

    REQUIRE(f.messages.titles.size() == 1);
    REQUIRE(f.messages.titles.front() == QStringLiteral("Error!"));
    REQUIRE(
        f.messages.texts.front() ==
        QStringLiteral("Select Frame and Model First!"));
    REQUIRE(f.bridge()->runState() == OptimizerBridge::RunState::Idle);
    REQUIRE(f.bridge()->canRun());
    REQUIRE(!f.bridge()->running());
}

TEST_CASE("optimizer_bridge: run() with a multi-select rejects headlessly, "
          "state stays idle",
          "[optimizer_bridge]") {
    /*Error path (plan 005 U6 scenario c): multi-mode rejection surfaces as
     * a message; no run starts.*/
    BridgeFixture f;
    f.loadWithSelection();
    f.study()->toggleModelSelected(1);  // second model -> multi-select

    f.bridge()->run();

    REQUIRE(f.messages.titles.size() == 1);
    REQUIRE(
        f.messages.texts.front().startsWith(
            QStringLiteral("Single-model mode")));
    REQUIRE(f.bridge()->runState() == OptimizerBridge::RunState::Idle);
    REQUIRE(f.bridge()->canRun());
}

TEST_CASE("optimizer_bridge: run() with a pose-dimension mismatch rejects "
          "headlessly",
          "[optimizer_bridge]") {
    /*Error path: pose matrix out of sync with the loaded lists -> the typed
     * critical-error message; no run starts.*/
    BridgeFixture f;
    f.loadWithSelection();
    /*Drift the storage out of sync (a plain service call, like a partial
     * second load): one extra frame slot in the pose matrix.*/
    f.session()->model_locations.LoadNewFrame();

    f.bridge()->run();

    REQUIRE(f.messages.titles.size() == 1);
    REQUIRE(
        f.messages.titles.front() == QStringLiteral("Critical Error!"));
    REQUIRE(f.bridge()->runState() == OptimizerBridge::RunState::Idle);
}

TEST_CASE("optimizer_bridge: stop() outside a run is a no-op",
          "[optimizer_bridge]") {
    BridgeFixture f;
    f.loadWithSelection();
    f.bridge()->stop();
    REQUIRE(f.bridge()->runState() == OptimizerBridge::RunState::Idle);
    REQUIRE(f.bridge()->canRun());
}
