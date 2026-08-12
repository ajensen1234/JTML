// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 005 U7: MlBridge degradation + seed pins (R8, R16, AE4). Deterministic
// Catch2 headless tests that direct-compile the bridge (MlBridge +
// AppBridge/StudyBridge/SettingsBridge/OptimizerBridge/DelegateSelection/
// ExperimentalScene) and link jtml_coordinator for the REAL seam symbols —
// every torch/GPU entry in MlBridge is guarded behind the availability flags,
// so the missing-model paths never invoke torch (the torch/GPU paths
// themselves are manual-visual under xcb / oracle-arbitrated by U9).
//
// Pins (plan 005 U7 test scenario b, headless subset):
//  - the .pt path surface: env fallback (JTML_SEG_PT / JTML_FEM_ESTIMATE_PT),
//    file:// URL normalization, and the availability flags;
//  - the AE4 degradation state machine: missing .pt -> typed messages + hint
//    status, no crash, no frame mutation, the estimate display stays empty,
//    and the plain-optimize path is untouched (the optimizer bridge still
//    gates + rejects identically — no ML state wedges it);
//  - the estimate -> optimizer seed (test scenario c, headless subset):
//    setSeedPose/applySeedPose lands the pose in LocationStorage + the
//    scene, is one-shot, drops on clearSeedPose, and never applies a
//    stale-frame seed (selection-change guard);
//  - the kind-preferred segment-model rule (femur/tibia pickers, fallback).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <QSettings>
#include <QTemporaryDir>

#include <opencv2/core.hpp>

#include "AppBridge.h"
#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "MlBridge.h"
#include "OptimizerBridge.h"
#include "StudyBridge.h"
#include "services/settings_service.h"

using Catch::Approx;

namespace {

/*Kneel_1 fixtures (repo-root WORKING_DIRECTORY, like jtml.session_controller
 * and the other experimental tests).*/
const char* kCalibrationPath = "test/golden/calibration.txt";
const QStringList kImagePaths = {
    "example_studies/Kneel_1/AT_K1_V1_0160.tif",
    "example_studies/Kneel_1/AT_K1_V1_0170.tif",
    "example_studies/Kneel_1/AT_K1_V1_0180.tif"};
const QStringList kModelPaths = {
    "example_studies/Kneel_1/KR_right_6_tib.stl",
    "example_studies/Kneel_1/KR_right_7_fem.stl"};

/*Records messageRequested emissions (the single QML Dialog analog) from
 * both bridges that use it (MlBridge + OptimizerBridge).*/
struct MessageRecorder {
    QStringList titles;
    QStringList texts;
};

void connect_messages(MlBridge* bridge, MessageRecorder* recorder) {
    QObject::connect(
        bridge, &MlBridge::messageRequested,
        [recorder](const QString& title, const QString& message) {
            recorder->titles.push_back(title);
            recorder->texts.push_back(message);
        });
}

void connect_messages(OptimizerBridge* bridge, MessageRecorder* recorder) {
    QObject::connect(
        bridge, &OptimizerBridge::messageRequested,
        [recorder](const QString& title, const QString& message) {
            recorder->titles.push_back(title);
            recorder->texts.push_back(message);
        });
}

/*Clears the env fallback BEFORE the hub member initializes (member init
 * order): the AppBridge/MlBridge constructor reads JTML_SEG_PT /
 * JTML_FEM_ESTIMATE_PT, so a previous test case that set them (the env
 * fallback pin) must never leak into a later fixture. Declared before
 * `hub` in the member list.*/
struct EnvIsolator {
    EnvIsolator() {
        qputenv("JTML_SEG_PT", "");
        qputenv("JTML_FEM_ESTIMATE_PT", "");
    }
};

/*The U7 fixture: a hub wired to an app-owned scene with an ini-backed
 * SettingsService (isolated — the real registry is never touched by headless
 * tests). The env fallback is isolated (see EnvIsolator) so a developer's
 * real .pt paths can never leak into the degradation pins.*/
struct MlFixture {
    QTemporaryDir dir;
    jta::SettingsService settings_service{
        dir.filePath("settings.ini"), QSettings::IniFormat};
    EnvIsolator env_isolator;  // must precede `hub` (member init order)
    ExperimentalScene scene;
    AppBridge hub{&scene, &settings_service};
    MessageRecorder messages;

    MlFixture() {
        REQUIRE(dir.isValid());
        connect_messages(hub.mlBridge(), &messages);
        connect_messages(hub.optimizerBridge(), &messages);
    }

    StudyBridge* study() { return hub.studyBridge(); }
    ExperimentalSession* session() { return hub.session(); }
    OptimizerBridge* optimizer() { return hub.optimizerBridge(); }
    MlBridge* ml() { return hub.mlBridge(); }

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

TEST_CASE("ml_bridge: env fallback + path normalization + availability flags",
          "[ml_bridge]") {
    /*Env fallback: JTML_SEG_PT -> segmentFemPt, JTML_FEM_ESTIMATE_PT ->
     * estimatePt (the oracle's user-provided fixture vars; the tibia picker
     * has no env var — picker-only). Set BEFORE the hub is constructed.*/
    qputenv("JTML_SEG_PT", "/models/seg_fem.pt");
    qputenv("JTML_FEM_ESTIMATE_PT", "/models/est_fem.pt");

    QTemporaryDir dir;
    REQUIRE(dir.isValid());
    jta::SettingsService settings_service{
        dir.filePath("settings.ini"), QSettings::IniFormat};
    ExperimentalScene scene;
    AppBridge hub{&scene, &settings_service};
    MlBridge* ml = hub.mlBridge();

    REQUIRE(ml->segmentFemPt() == QStringLiteral("/models/seg_fem.pt"));
    REQUIRE(ml->estimatePt() == QStringLiteral("/models/est_fem.pt"));
    REQUIRE(ml->segmentTibPt().isEmpty());
    REQUIRE(ml->hasSegmentModel());
    REQUIRE(ml->hasEstimateModel());

    /*file:// URL normalization (the QML FileDialog yields URLs; the torch
     * loader wants local paths).*/
    ml->setSegmentTibPt(QStringLiteral("file:///models/seg_tib.pt"));
    REQUIRE(ml->segmentTibPt() == QStringLiteral("/models/seg_tib.pt"));
    /*Clearing a path flips the flag.*/
    ml->setEstimatePt(QStringLiteral(""));
    REQUIRE(!ml->hasEstimateModel());
    /*Plain paths pass through untouched (headless-test call style).*/
    ml->setEstimatePt(QStringLiteral("/models/other.pt"));
    REQUIRE(ml->estimatePt() == QStringLiteral("/models/other.pt"));
}

TEST_CASE("ml_bridge: no study -> typed message, nothing changes",
          "[ml_bridge]") {
    /*Degradation guard order: the running guard, then the frames guard —
     * with no study the actions report and touch nothing.*/
    MlFixture f;

    f.ml()->segmentCurrentFrame();
    REQUIRE(f.messages.titles.size() == 1);
    REQUIRE(f.messages.titles.front() == QStringLiteral("Error!"));
    REQUIRE(f.messages.texts.front() == QStringLiteral("Load images first!"));
    REQUIRE(f.ml()->statusText() == QStringLiteral("No frames loaded."));
    REQUIRE(!f.ml()->hasEstimate());

    f.ml()->estimateCurrentFrame();
    REQUIRE(f.messages.titles.size() == 2);
    REQUIRE(f.messages.texts[1] == QStringLiteral("Load images first!"));
    REQUIRE(!f.ml()->hasEstimate());
}

TEST_CASE("ml_bridge: missing models degrade cleanly — clear messages, no "
          "crash, optimizer untouched",
          "[ml_bridge]") {
    /*AE4 (plan 005 U7 test scenario b): with a loaded study but NO .pt
     * models, the segment/estimate actions report typed messages, mutate
     * nothing (the frame's inverted image stays the plain inversion), and
     * never touch torch.*/
    MlFixture f;
    f.loadWithSelection();

    REQUIRE(!f.ml()->hasSegmentModel());
    REQUIRE(!f.ml()->hasEstimateModel());

    /*Capture the pre-segment inverted image (the plain 255 - original
     * inversion the Frame constructor made).*/
    const cv::Mat inverted_before =
        f.session()->loaded_frames[0].GetInvertedImage();
    const cv::Mat expected =
        255 - f.session()->loaded_frames[0].GetOriginalImage();

    f.ml()->segmentCurrentFrame();
    REQUIRE(f.messages.titles.size() == 1);
    REQUIRE(f.messages.titles.front() == QStringLiteral("Error!"));
    REQUIRE(
        f.messages.texts.front().startsWith(
            QStringLiteral("No segmentation model loaded")));
    REQUIRE(
        f.ml()->statusText() ==
        QStringLiteral("No segmentation model (.pt) loaded."));
    /*No torch, no crash, no frame mutation.*/
    cv::Mat diff;
    cv::absdiff(
        f.session()->loaded_frames[0].GetInvertedImage(), expected, diff);
    REQUIRE(cv::sum(diff)[0] == 0);

    f.ml()->estimateCurrentFrame();
    REQUIRE(f.messages.titles.size() == 2);
    REQUIRE(
        f.messages.texts[1].startsWith(
            QStringLiteral("No pose-estimation model loaded")));
    REQUIRE(!f.ml()->hasEstimate());
    REQUIRE(f.ml()->estimateText().isEmpty());

    /*The plain-optimize path is untouched: the optimizer bridge still gates
     * identically (WITHOUT a selection -> the typed rejection, state Idle)
     * and stays re-runnable — the ML degradation wedges nothing. The
     * selection is cleared first: with a valid selection the gate would
     * pass and a REAL GPU run would start (not headless-safe; the GPU run
     * itself is manual-visual / oracle).*/
    f.study()->clearModelSelection();
    REQUIRE(f.optimizer()->canRun());
    REQUIRE(!f.optimizer()->running());
    f.optimizer()->run();
    REQUIRE(f.messages.titles.size() == 3);
    REQUIRE(
        f.messages.texts[2] ==
        QStringLiteral("Select Frame and Model First!"));
    REQUIRE(f.optimizer()->runState() == OptimizerBridge::RunState::Idle);
    REQUIRE(f.optimizer()->canRun());
    /*And the gate itself still accepts the controller-test runnable shape
     * (identical semantics to the U6 pins — the ML surface adds no gate
     * coupling; the shared core's GateInput shape, plan 006 U5).*/
    jta::OptimizerRunControllerCore::GateInput in;
    in.selected_model_rows = {0};
    in.current_frame = 0;
    in.frame_count = 3;
    in.model_current_index = 0;
    in.model_count = 2;
    in.pose_frame_count = 3;
    in.pose_model_count = 2;
    REQUIRE(
        OptimizerBridge::EvaluateGate(in).status ==
        OptimizerBridge::GateStatus::Ok);
}

TEST_CASE("ml_bridge: estimate requires the segmentation model too",
          "[ml_bridge]") {
    /*The widgets estimate actions segment FIRST and require the segment
     * model; the bridge mirrors that (a clear message, no torch work) even
     * when the estimate .pt IS set.*/
    MlFixture f;
    f.loadWithSelection();
    f.ml()->setEstimatePt(QStringLiteral("/models/est_fem.pt"));
    REQUIRE(f.ml()->hasEstimateModel());
    REQUIRE(!f.ml()->hasSegmentModel());

    f.ml()->estimateCurrentFrame();
    REQUIRE(f.messages.titles.size() == 1);
    REQUIRE(
        f.messages.texts.front().startsWith(
            QStringLiteral("Estimate needs a segmentation model too")));
    REQUIRE(!f.ml()->hasEstimate());
    REQUIRE(!f.ml()->statusText().isEmpty());
}

TEST_CASE("ml_bridge: estimate without a selected model reports, no torch",
          "[ml_bridge]") {
    /*The v1 single-model rule: pose ops need a primary model. With the .pt
     * paths set but no selection, the estimate reports and touches nothing
     * (the seg-model guard is ordered after the selection guard — mirror of
     * the widgets "Must Be in Single Model Selection Mode" checks).*/
    MlFixture f;
    f.loadWithSelection();
    f.study()->clearModelSelection();
    f.ml()->setSegmentFemPt(QStringLiteral("/models/seg_fem.pt"));
    f.ml()->setEstimatePt(QStringLiteral("/models/est_fem.pt"));

    f.ml()->estimateCurrentFrame();
    REQUIRE(f.messages.titles.size() == 1);
    REQUIRE(
        f.messages.texts.front() ==
        QStringLiteral("Select one model to estimate."));
    REQUIRE(!f.ml()->hasEstimate());
}

TEST_CASE("ml_bridge: segment model follows the implant kind with fallback",
          "[ml_bridge]") {
    /*The kind-preferred rule (one Segment button mirroring the two widgets
     * actions): Femur -> segmentFemPt, Tibia -> segmentTibPt, and the other
     * picker as fallback when the preferred one is unset.*/
    MlFixture f;
    f.ml()->setSegmentFemPt(QStringLiteral("/models/seg_fem.pt"));
    f.ml()->setSegmentTibPt(QStringLiteral("/models/seg_tib.pt"));

    REQUIRE(f.ml()->implantKind() == 0);  // default Femur
    REQUIRE(
        f.ml()->activeSegmentModelPath() ==
        QStringLiteral("/models/seg_fem.pt"));

    f.ml()->setImplantKind(1);
    REQUIRE(
        f.ml()->activeSegmentModelPath() ==
        QStringLiteral("/models/seg_tib.pt"));

    /*Fallback: Tibia preferred but only the femur picker set.*/
    f.ml()->setSegmentTibPt(QStringLiteral(""));
    REQUIRE(
        f.ml()->activeSegmentModelPath() ==
        QStringLiteral("/models/seg_fem.pt"));
    REQUIRE(f.ml()->hasSegmentModel());

    /*Out-of-range kind values are ignored (property guard).*/
    f.ml()->setImplantKind(7);
    REQUIRE(f.ml()->implantKind() == 1);
}

TEST_CASE("ml_bridge: the estimate seed lands in storage + scene, one-shot",
          "[ml_bridge]") {
    /*Integration (plan 005 U7 test scenario c, headless subset): the
     * OptimizerBridge seed API — what estimateCurrentFrame() calls after a
     * successful estimate — persists the pose into the session storage (the
     * by-value matrix Initialize copies — the widgets LaunchOptimizer seed
     * path) + the scene, exactly once, and re-runs are unaffected.*/
    MlFixture f;
    f.loadWithSelection();

    f.optimizer()->setSeedPose(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    f.optimizer()->applySeedPose();

    const Point6D stored = f.session()->model_locations.GetPose(0, 0);
    REQUIRE(stored.x == Approx(1.0));
    REQUIRE(stored.y == Approx(2.0));
    REQUIRE(stored.z == Approx(3.0));
    REQUIRE(stored.xa == Approx(4.0));
    REQUIRE(stored.ya == Approx(5.0));
    REQUIRE(stored.za == Approx(6.0));
    REQUIRE(f.scene.models().size() == 2);
    REQUIRE(f.scene.models()[0].pose.x == Approx(1.0));
    REQUIRE(f.scene.models()[0].pose.za == Approx(6.0));

    /*One-shot: a second apply with no new seed changes nothing.*/
    f.optimizer()->applySeedPose();
    REQUIRE(f.session()->model_locations.GetPose(0, 0).x == Approx(1.0));

    /*clearSeedPose (MlBridge's stale-estimate cleanup): the next apply is a
     * no-op, so a NEW seed still applies normally (re-seed works).*/
    f.optimizer()->clearSeedPose();
    f.optimizer()->applySeedPose();
    REQUIRE(f.session()->model_locations.GetPose(0, 0).x == Approx(1.0));
    f.optimizer()->setSeedPose(9.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    f.optimizer()->applySeedPose();
    REQUIRE(f.session()->model_locations.GetPose(0, 0).x == Approx(9.0));
}

TEST_CASE("ml_bridge: a stale-frame seed never overrides another frame",
          "[ml_bridge]") {
    /*Stale guard: the seed was estimated for frame 0; switching the current
     * frame (which also fires MlBridge's selection-change cleanup) must drop
     * it — a stale-frame estimate must never override a different frame's
     * pose. Observable contract: applying after the switch leaves frame 1
     * at its pre-seed pose.*/
    MlFixture f;
    f.loadWithSelection();

    f.optimizer()->setSeedPose(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    const Point6D frame1_before = f.session()->model_locations.GetPose(1, 0);

    f.study()->setCurrentFrame(1);
    f.optimizer()->applySeedPose();

    const Point6D frame1_after = f.session()->model_locations.GetPose(1, 0);
    REQUIRE(frame1_after.x == Approx(frame1_before.x));
    REQUIRE(frame1_after.y == Approx(frame1_before.y));
    REQUIRE(frame1_after.z == Approx(frame1_before.z));
    /*And the seeded frame's stored pose is NOT the seed either (the seed
     * was dropped, not re-targeted).*/
    REQUIRE(f.session()->model_locations.GetPose(0, 0).x != Approx(1.0));
}

TEST_CASE("ml_bridge: a failed segment aborts the estimate (P2-3)",
          "[ml_bridge]") {
    /*Review fix P2-3: the estimate segments FIRST; a segment failure must
     * abort the estimate BEFORE the estimate-model load — the regression
     * would otherwise run on the stale inverted image (saving a bogus
     * pose + seeding the optimizer) and overwrite the 'Segmentation
     * failed.' status. Headless-reachable failure leg: the torch load of
     * a bogus .pt path (the throw/empty-result legs are pinned at the
     * orchestrator level in ml_orchestrator_test.cpp). Observable pins:
     * ONE message (the segment load error, no second estimate-model
     * message), the segment-failure status SURVIVES, no estimate display,
     * no estimateChanged emission, no seed set on the optimizer.*/
    MlFixture f;
    f.loadWithSelection();
    f.ml()->setSegmentFemPt(QStringLiteral("/nonexistent/seg_fem.pt"));
    f.ml()->setEstimatePt(QStringLiteral("/nonexistent/est_fem.pt"));
    int estimate_changed = 0;
    QObject::connect(
        f.ml(), &MlBridge::estimateChanged,
        [&estimate_changed]() { ++estimate_changed; });

    f.ml()->estimateCurrentFrame();

    /*Only the segment failure surfaced (the OLD flow continued into the
     * estimate-model load and emitted a second message + overwrote the
     * status).*/
    REQUIRE(f.messages.titles.size() == 1);
    REQUIRE(
        f.messages.texts.front() ==
        QStringLiteral("Cannot load PyTorch Torch Script model at: "
                       "/nonexistent/seg_fem.pt"));
    REQUIRE(
        f.ml()->statusText() ==
        QStringLiteral("Segmentation model failed to load."));
    REQUIRE(!f.ml()->hasEstimate());
    REQUIRE(f.ml()->estimateText().isEmpty());
    REQUIRE(estimate_changed == 0);
    /*The pending optimizer seed stays unset (the estimate never
     * completed): applySeedPose is a no-op and the storage pose is
     * untouched.*/
    const Point6D pose_before = f.session()->model_locations.GetPose(0, 0);
    f.optimizer()->applySeedPose();
    const Point6D pose_after = f.session()->model_locations.GetPose(0, 0);
    REQUIRE(pose_after.x == Approx(pose_before.x));
    REQUIRE(pose_after.za == Approx(pose_before.za));
}

TEST_CASE("ml_bridge: clearEstimate is a safe no-op without an estimate",
          "[ml_bridge]") {
    /*The stale-display cleanup (selection-change slot + QML button): with
     * nothing estimated it changes nothing and emits nothing crashy.*/
    MlFixture f;
    f.loadWithSelection();
    f.ml()->clearEstimate();
    REQUIRE(!f.ml()->hasEstimate());
    REQUIRE(f.ml()->estimateText().isEmpty());
    REQUIRE(f.ml()->statusText().isEmpty());
}

/*Plan 007 U3 (D4): a pose-table edit on the seeded frame+model drops the
 * pending ML seed — wired in the hub (poseTableChanged ->
 * OptimizerBridge::clearSeedPose). Integration via the REAL Kneel_1 load
 * path (the pose suite covers the synthetic-dataset variants of the same
 * wiring). Without this the next run() would silently apply the estimate
 * over the user's arrangement (I3).*/
TEST_CASE("ml_bridge: a pose-table edit drops the pending seed (D4 wiring)",
          "[ml_bridge]") {
    MlFixture f;
    f.loadWithSelection();

    f.optimizer()->setSeedPose(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    REQUIRE(f.optimizer()->hasSeedPose());

    /*The pose table's per-cell commit (frame 0, primary model 0).*/
    REQUIRE(f.hub.poseBridge()->setPoseValue(
        0, 0, 2, QStringLiteral("-4.0")));
    REQUIRE_FALSE(f.optimizer()->hasSeedPose());

    /*The seed is really gone: a subsequent apply is a no-op — the edited
     * pose stays, the estimate pose does not arrive.*/
    f.optimizer()->applySeedPose();
    REQUIRE(f.session()->model_locations.GetPose(0, 0).z == Approx(-4.0));
}
