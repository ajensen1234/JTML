// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 005 U6: OptimizerBridge gate + state-core pins (R6, R7, R13).
// Deterministic Catch2 headless tests that direct-compile the bridge
// (OptimizerBridge + AppBridge/StudyBridge/SettingsBridge/DelegateSelection/
// ExperimentalScene) and link jtml_coordinator for the REAL OptimizerManager
// symbols — but never touch the GPU: the gate rejects BEFORE any manager or
// thread is created, so run()'s rejection paths are headless-safe; the state
// + persistence core (applyOptimizedFrame) is exercised directly. The GPU
// run itself is manual-visual under xcb / oracle-arbitrated by U9.
//
// Pins (plan 005 U6 test scenarios a-e, headless subset):
//  - the bridge's gate INPUTS mirror OptimizeIntentController's tested
//    semantics (reusing the intent test's RunnableInput shape): the
//    buildGateInput() assembly from a real loaded study, and the
//    EvaluateGate classification — no-selection rejection, multi-mode
//    (single-model v1 rule) rejection, pose-dimension-mismatch rejection;
//  - run() rejection paths surface the typed messages through the single
//    Dialog channel and leave the state machine untouched (idle, re-runnable
//    — no thread starts, the gate is before the manager creation);
//  - integration: OptimizedFrame handling (applyOptimizedFrame) lands the
//    pose in LocationStorage + the scene, transitions to completed, and
//    leaves the bridge re-runnable (stop() no-op outside a run).

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
#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "OptimizerBridge.h"
#include "StudyBridge.h"
#include "services/settings_service.h"
#include "view/frame_list_model.h"
#include "view/model_list_model.h"

using Catch::Approx;
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

/*The OptimizeIntentController test's runnable input shape (reused verbatim
 * for the bridge gate — the same plain values the intent tests pin).*/
OptimizeIntentController::Input RunnableInput() {
    OptimizeIntentController::Input in;
    in.selected_model_rows = {0};
    in.previous_frame_index = 0;
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

TEST_CASE("optimizer_bridge: gate accepts the controller-test runnable input",
          "[optimizer_bridge]") {
    /*The bridge gate mirrors OptimizeIntentController's tested semantics:
     * the intent test's RunnableInput shape passes and the primary + frame
     * pack through unchanged.*/
    const auto result = OptimizerBridge::EvaluateGate(RunnableInput());
    REQUIRE(result.status == OptimizerBridge::GateStatus::Ok);
    REQUIRE(result.intent.status == OptimizeIntentController::Status::Ok);
    REQUIRE(result.intent.primary_model_index == 0);
    REQUIRE(result.intent.current_frame == 0);
}

TEST_CASE("optimizer_bridge: no selection rejects before any run",
          "[optimizer_bridge]") {
    /*Mirror of the intent test's empty-selection case at the bridge level:
     * the controller's SelectFrameAndModel wins (checked before the
     * single-model rule).*/
    auto in = RunnableInput();
    in.selected_model_rows = {};
    in.model_current_index = -1;
    REQUIRE(OptimizerBridge::EvaluateGate(in).status ==
            OptimizerBridge::GateStatus::SelectFrameAndModel);
}

TEST_CASE("optimizer_bridge: multi-mode (multi-select) rejects — v1 "
          "single-model rule",
          "[optimizer_bridge]") {
    /*The controller itself Oks multi-select (primary = first); the bridge
     * adds the v1 single-model-mode rejection on top (plan 005 U6 review
     * fix — pose ops are primary-model-only in v1).*/
    auto in = RunnableInput();
    in.selected_model_rows = {0, 1};
    REQUIRE(OptimizerBridge::EvaluateGate(in).status ==
            OptimizerBridge::GateStatus::SingleModelOnly);
}

TEST_CASE("optimizer_bridge: pose-dimension mismatch rejects",
          "[optimizer_bridge]") {
    /*Mirror of the intent test's PoseMatrixDimensionMismatch case.*/
    auto in = RunnableInput();
    in.pose_frame_count = 2;  // != frame_count
    REQUIRE(OptimizerBridge::EvaluateGate(in).status ==
            OptimizerBridge::GateStatus::PoseMatrixDimensionMismatch);
}

TEST_CASE("optimizer_bridge: no current frame rejects", "[optimizer_bridge]") {
    auto in = RunnableInput();
    in.previous_frame_index = -1;
    in.current_frame = -1;
    REQUIRE(OptimizerBridge::EvaluateGate(in).status ==
            OptimizerBridge::GateStatus::SelectFrameAndModel);
}

TEST_CASE("optimizer_bridge: buildGateInput mirrors the study state",
          "[optimizer_bridge]") {
    /*Integration: after the real Kneel_1 load + a runnable selection, the
     * bridge's Input assembly matches the controller-test input shape for
     * the same state (frames, models, selection rows, pose-matrix counts).*/
    BridgeFixture f;
    f.loadWithSelection();

    const auto in = f.bridge()->buildGateInput();
    REQUIRE(in.selected_model_rows == std::vector<int>{0});
    REQUIRE(in.previous_frame_index == 0);
    REQUIRE(in.current_frame == 0);
    REQUIRE(in.frame_count == 3);
    REQUIRE(in.model_current_index == 0);
    REQUIRE(in.model_count == 2);
    REQUIRE(in.pose_frame_count == 3);
    REQUIRE(in.pose_model_count == 2);

    /*And the assembled input passes the bridge gate (the runnable path).*/
    const auto result = OptimizerBridge::EvaluateGate(in);
    REQUIRE(result.status == OptimizerBridge::GateStatus::Ok);
    REQUIRE(result.intent.primary_model_index == 0);
}

TEST_CASE("optimizer_bridge: run() with no selection rejects headlessly, "
          "state stays idle",
          "[optimizer_bridge]") {
    /*Error path (plan 005 U6 scenario b): no selection -> typed message,
     * no thread starts (the gate runs before any manager/thread is
     * created), state machine untouched and re-runnable.*/
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

TEST_CASE("optimizer_bridge: OptimizedFrame handling saves the pose and "
          "completes the run",
          "[optimizer_bridge]") {
    /*Integration (plan 005 U6 scenario e): the OptimizedFrame core lands the
     * result pose in LocationStorage + the scene, emits the frameOptimized
     * relay, and transitions to a re-runnable completed state. The
     * signal-relay slot and the core are the same code path the GPU run
     * uses.*/
    BridgeFixture f;
    f.loadWithSelection();
    REQUIRE(f.bridge()->runState() == OptimizerBridge::RunState::Idle);

    bool relayed = false;
    int relayed_model = -1;
    QObject::connect(
        f.bridge(), &OptimizerBridge::frameOptimized,
        [&relayed, &relayed_model](int, int modelIndex) {
            relayed = true;
            relayed_model = modelIndex;
        });

    f.bridge()->applyOptimizedFrame(
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        false /* move_next_frame: Single directive, v1 */,
        0 /* primary_model_index */, false /* error_occurred */,
        QStringLiteral("Single"));

    /*The pose landed in the session storage (the manager worked on a copy)
     * and in the scene (viewport re-render path).*/
    const Point6D stored =
        f.session()->model_locations.GetPose(0, 0);
    REQUIRE(stored.x == Approx(1.0));
    REQUIRE(stored.y == Approx(2.0));
    REQUIRE(stored.z == Approx(3.0));
    REQUIRE(stored.xa == Approx(4.0));
    REQUIRE(stored.ya == Approx(5.0));
    REQUIRE(stored.za == Approx(6.0));
    REQUIRE(f.scene.models().size() == 2);
    REQUIRE(f.scene.models()[0].pose.x == Approx(1.0));
    REQUIRE(f.scene.models()[0].pose.za == Approx(6.0));

    /*The relay fired for the QML viewport glue, and the run completed
     * re-runnable.*/
    REQUIRE(relayed);
    REQUIRE(relayed_model == 0);
    REQUIRE(f.bridge()->runState() == OptimizerBridge::RunState::Completed);
    REQUIRE(f.bridge()->canRun());
    REQUIRE(!f.bridge()->running());
}

TEST_CASE("optimizer_bridge: OptimizedFrame keeps a completed run at "
          "completed on a duplicate trailing frame",
          "[optimizer_bridge]") {
    /*The headless-reachable ordering pin: the trailing-frame path the GPU
     * run emits (OptimizedFrame -> finished) must not disturb a finished
     * run's state — a duplicate/trailing frame still saves (widgets
     * SavePose parity) but the state stays completed and re-runnable.
     * (The error-state branch itself is set by the OptimizerError relay,
     * which is GPU-run-only and thus manual/oracle.)*/
    BridgeFixture f;
    f.loadWithSelection();

    f.bridge()->applyOptimizedFrame(
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, false,
        QStringLiteral("Single"));
    REQUIRE(f.bridge()->runState() == OptimizerBridge::RunState::Completed);
    f.bridge()->applyOptimizedFrame(
        2.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, false,
        QStringLiteral("Single"));
    REQUIRE(f.bridge()->runState() == OptimizerBridge::RunState::Completed);
    REQUIRE(f.bridge()->canRun());
    /*The trailing frame still saved (widgets SavePose parity).*/
    REQUIRE(f.session()->model_locations.GetPose(0, 0).x == Approx(2.0));
}

TEST_CASE("optimizer_bridge: progress surface mirrors the widgets stage "
          "math",
          "[optimizer_bridge]") {
    /*UpdateDisplay bind: the stage/calls/min/progress surface mirrors the
     * widgets onUpdateDisplay arithmetic (mainscreen.cpp:4521-4550) against
     * the same cumulative budgets — Trunk while calls < trunk budget, then
     * Branch n per branch-budget block, then Extra Z-Translation, then
     * Finished at the cumulative cap; progress = calls / cumulative.
     * (The values come from the session's OptimizerSettings, so the
     * assertions reuse the same arithmetic instead of hardcoding
     * constants.)*/
    BridgeFixture f;
    f.loadWithSelection();
    const OptimizerSettings& s = f.hub.settingsBridge()->optimizerSettings();
    const int cumulative =
        s.trunk_budget +
        (s.enable_branch_ ? s.number_branches * s.branch_budget : 0) +
        (s.enable_leaf_ ? s.leaf_budget : 0);
    REQUIRE(cumulative > 0);

    /*Mid-trunk: stage Trunk, progress calls/cumulative.*/
    const int mid_trunk = s.trunk_budget / 2;
    f.bridge()->refreshProgress(mid_trunk, -0.25);
    REQUIRE(f.bridge()->stageText() == QStringLiteral("Trunk"));
    REQUIRE(f.bridge()->costCalls() == mid_trunk);
    REQUIRE(f.bridge()->currentMinimum() == Approx(-0.25));
    REQUIRE(
        f.bridge()->progress() ==
        Approx(double(mid_trunk) / double(cumulative)));

    /*First branch block (when enabled): Branch 1.*/
    if (s.enable_branch_) {
        const int first_branch = s.trunk_budget + 1;
        f.bridge()->refreshProgress(first_branch, 0.0);
        REQUIRE(f.bridge()->stageText() == QStringLiteral("Branch 1"));
    }

    /*At/over the cumulative cap: Finished, progress pinned at 1.*/
    f.bridge()->refreshProgress(cumulative, -1.0);
    REQUIRE(f.bridge()->stageText() == QStringLiteral("Finished"));
    REQUIRE(f.bridge()->progress() == Approx(1.0));
    f.bridge()->refreshProgress(cumulative * 2, -1.0);
    REQUIRE(f.bridge()->progress() == Approx(1.0));
}
