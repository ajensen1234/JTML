// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Tier-2 parity gate (plan 005 U9, R13): the app's OWN optimizer run path
// (OptimizerBridge — the real wiring the QML app drives: OptimizeIntent
// Controller gate -> OptimizerManager + QThread -> Initialize with the app's
// containers by value + a QModelIndexList -> the 7 binds -> OptimizedFrame)
// on Kneel_1 frame 0 WITH THE ORACLE CONFIG must land within the oracle's
// band: rendered-silhouette IoU >= 0.85 vs the empirically pinned label
// (frame 0 <-> Labels/fem/AT_K1_V1_0160_label_fem.tif — NOT name-aligned in
// the filesystem; pinned by start-pose IoU == 1.0 in test/oracle/
// oracle_test.cpp).
//
// This is NOT a QML window and NOT a hand-rolled DirectOptimizer run (that
// is what jtml.oracle already does). The parity gate's job is the APP's
// wiring against the R2-validated band: the binary direct-compiles the real
// bridge stack (AppBridge/StudyBridge/SettingsBridge/OptimizerBridge), loads
// the study through StudyBridge's three-action flow, seeds the start pose
// from example_studies/Kneel_1/fem.jts via pose_file_io (the oracle's
// StartPoses values — a fresh-load default pose would false-fail the gate),
// configures the parity OptimizerSettings through SettingsBridge, calls
// OptimizerBridge::run(), waits for the Completed state, then renders at the
// final pose via the compute render path (GPUModel, backface culling OFF —
// the oracle config) and computes IoU vs the pinned label with the SAME
// mechanics as oracle_test.cpp (including the label TIFF vertical flip).
//
// Parity protocol (pinned by the plan's review fix + baseline.json):
//  - base frame: example_studies/Kneel_1/1024/2806.tif (oracle frame 0);
//  - edge config: Canny 3/0/150, dilation 6 (the oracle's measured config —
//    StudyBridge's hard-coded load params {3,40,120,0} are replaced by the
//    oracle-config Frame after loadImages; documented in the report);
//  - OptimizerSettings: trunk budget 3000, branch/leaf budgets 0 (the plan's
//    primary config; RunDirectStage at budget 0 = exactly one cost eval, no
//    search iterations, no error, running optimum preserved — verified
//    against direct_optimizer.cpp's `(calls + offset) < budget_` loop
//    guard), trunk range (12,12,15,15,15,15), DIRECT_DILATION, dilation 6,
//    directive Single (v1 run scope = the current frame);
//  - start pose: fem.jts frame 0 via pose_file::ReadKinematicsFile -> set
//    into LocationStorage BEFORE the run + the scene mirror (the widgets
//    Load_Pose then optimize equivalent; keeps run()'s SaveLastPose mirror
//    from clobbering the seed with the fresh-load default pose);
//  - curvature heatmaps: setCurvatureHeatmaps() on the loaded frame (the
//    widgets segment post-processing / oracle BuildFramePipeline parity —
//    the Frame ctor leaves num_curvature_keypoints_ uninitialized and
//    OptimizerManager::Initialize reads it unconditionally);
//  - comparator: GPUModel(backface OFF) + GPUMetrics::IOU vs the vertically
//    flipped label TIFF, exactly like oracle_test.cpp;
//  - gate: recovered-pose IoU >= 0.85; start-pose IoU > 0.50 pins the
//    frame<->label pairing empirically (oracle parity); pose gap vs fem.jts
//    is informational.
//
// Harness assertions beyond the gate (plan 005 U9 test scenario b): the run
// reached the Completed state (wiring correctness) and SavePose landed in
// LocationStorage (applyOptimizedFrame wrote the session storage + scene).
//
// GPU-only: NEVER in the headless default. Run explicitly on the GPU box:
//   pixi run cmake --build .build --target jtml_test_qml_parity_check
//   ctest --test-dir .build -R qml_parity_check --output-on-failure
// (registered LABELS "oracle", TIMEOUT 3600, repo-root WORKING_DIRECTORY,
// forced xcb env — the jtml.oracle recipe).

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <QCoreApplication>
#include <QEventLoop>
#include <QSettings>
#include <QTemporaryDir>
#include <QTimer>

#include <opencv2/imgcodecs.hpp>

#include "compute/camera_calibration.h"
#include "compute/frame.h"
#include "compute/gpu_image.cuh"
#include "compute/gpu_metrics.cuh"
#include "compute/gpu_model.cuh"
#include "compute/pose_matrix.h"

#include "domain/data_structures_6D.h"
#include "domain/pose_file_io.h"

#include "AppBridge.h"
#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "OptimizerBridge.h"
#include "SettingsBridge.h"
#include "StudyBridge.h"

using gpu_cost_function::Pose;
using gpu_cost_function::GPUImage;
using gpu_cost_function::GPUModel;
using gpu_cost_function::GPUMetrics;

namespace {

/*---- Fixtures (repo-root WORKING_DIRECTORY, like jtml.oracle) ----*/

const std::string kStudyDir = "example_studies/Kneel_1/";
// The oracle's frame 0 base image (baseline.json base_images = 1024/*.tif).
const std::string kFrameImage = kStudyDir + "1024/2806.tif";
// Empirically pinned frame 0 <-> label pairing (verified in oracle_test.cpp:
// start-pose render IoU == 1.0; NOT name-aligned in the filesystem).
const std::string kLabel = kStudyDir +
                           "Labels/fem/AT_K1_V1_0160_label_fem.tif";
const std::string kFemStl = kStudyDir + "KR_right_7_fem.stl";
const std::string kFemJts = kStudyDir + "fem.jts";
const char* kCalibrationPath = "test/golden/calibration.txt";

const int kWidth = 1024;
const int kHeight = 1024;
const int kDevice = 0;

/*The oracle's appearance gate (baseline.json silhouette_iou_threshold).*/
const double kIouGate = 0.85;
/*Start-pose correspondence floor (oracle_test.cpp's REQUIRE(best_iou > 0.50)):
 * pins the frame<->label pairing empirically inside this binary too.*/
const double kStartPoseIouFloor = 0.50;
/*Watchdog for the run (budget-3000 trunk takes a few minutes on the RTX
 * 3090); the ctest TIMEOUT 3600 mirrors jtml.oracle.*/
const int kRunTimeoutMs = 30 * 60 * 1000;

/*Point6D order is (x,y,z, x_rot,y_rot,z_rot) — fem.jts columns are stored
 * (x_tran,y_tran,z_tran, z_rot,x_rot,y_rot) and pose_file_io maps them back
 * to Point6D order on read (the oracle's StartPoses values).*/
Pose ToPose(const Point6D& p) {
    return Pose(p.x, p.y, p.z, p.xa, p.ya, p.za);
}

/*The oracle's label loader (oracle_test.cpp GrayscaleUchar): the binary
 * label TIFFs are stored with a bottom-left y-origin while the GPU renderer
 * outputs top-left origin — flip the label vertically so both are in the
 * same image frame (verified there: fem.jts render vs flipped label has
 * IoU == 1.0).*/
std::vector<unsigned char> GrayscaleUchar(const std::string& path,
                                          bool flip_vertical = false) {
    cv::Mat rgb = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (rgb.empty()) throw std::runtime_error("could not read image: " + path);
    if (rgb.cols != kWidth || rgb.rows != kHeight)
        throw std::runtime_error("unexpected image size for " + path);
    if (flip_vertical) cv::flip(rgb, rgb, 0);
    std::vector<unsigned char> buf((size_t)kWidth * kHeight);
    for (int y = 0; y < kHeight; ++y) {
        const unsigned char* row = rgb.ptr<unsigned char>(y);
        std::copy(row, row + kWidth, buf.begin() + (size_t)y * kWidth);
    }
    return buf;
}

/*Records messageRequested emissions (the single QML Dialog analog) from both
 * bridges — load errors and run errors (intent-gate rejection, Initialize
 * failure, OptimizerError) all surface there.*/
struct MessageRecorder {
    QStringList titles;
    QStringList texts;
};

void connect_messages(StudyBridge* study, OptimizerBridge* optimizer,
                      MessageRecorder* recorder) {
    QObject::connect(
        study, &StudyBridge::messageRequested,
        [recorder](const QString& title, const QString& message) {
            recorder->titles.push_back(title);
            recorder->texts.push_back(message);
        });
    QObject::connect(
        optimizer, &OptimizerBridge::messageRequested,
        [recorder](const QString& title, const QString& message) {
            recorder->titles.push_back(title);
            recorder->texts.push_back(message);
        });
}

/*The parity fixture (the headless tests' BridgeFixture pattern): the app's
 * hub wired to an app-owned scene + an ini-backed SettingsService (the real
 * registry is never touched by tests).*/
struct ParityFixture {
    QTemporaryDir dir;
    jta::SettingsService settings_service{
        dir.filePath("settings.ini"), QSettings::IniFormat};
    ExperimentalScene scene;
    AppBridge hub{&scene, &settings_service};
    MessageRecorder messages;

    ParityFixture() {
        REQUIRE(dir.isValid());
        connect_messages(hub.studyBridge(), hub.optimizerBridge(), &messages);
    }
};

/*The parity run itself: run() + wait for a terminal state (Completed or
 * Error) on the main thread's event loop — the manager's signals arrive via
 * queued connections (manager lives in the worker thread). Returns the
 * terminal state; the watchdog prevents a hang (the ctest TIMEOUT 3600 is
 * the outer bound).*/
OptimizerBridge::RunState RunAndWait(OptimizerBridge* optimizer) {
    QEventLoop loop;
    QTimer watchdog;
    watchdog.setSingleShot(true);
    QObject::connect(&watchdog, &QTimer::timeout, &loop, &QEventLoop::quit);
    QObject::connect(
        optimizer, &OptimizerBridge::runStateChanged, &loop,
        [&]() {
            if (optimizer->runState() == OptimizerBridge::RunState::Completed ||
                optimizer->runState() == OptimizerBridge::RunState::Error) {
                loop.quit();
            }
        });
    optimizer->run();
    watchdog.start(kRunTimeoutMs);
    loop.exec();
    /*Deliver the queued finished()->onFinished() tail (thread teardown) so
     * the fixture destruction never races the worker.*/
    for (int i = 0; i < 4; ++i) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
    }
    return optimizer->runState();
}

/*The compute-layer comparator (the oracle's render + IoU mechanics): a
 * GPUModel with backface culling OFF (the oracle config — both sides must
 * see the same silhouette) + GPUMetrics::IOU against the pinned label.*/
struct Comparator {
    GPUMetrics metrics;
    GPUImage label;
    GPUModel model;

    Comparator(std::vector<float>& vertices, std::vector<float>& normals,
               int triangle_count, const CameraCalibration& camera)
        : label(kWidth, kHeight, kDevice,
                GrayscaleUchar(kLabel, /*flip_vertical=*/true).data()),
          model("femur", /*principal=*/true, kWidth, kHeight, kDevice,
                /*use_backface_culling=*/false, &vertices[0], &normals[0],
                triangle_count, camera) {}

    bool ok() {
        return metrics.IsInitializedCorrectly() &&
               label.IsInitializedCorrectly() && model.IsInitializedCorrectly();
    }

    /*Render the femur at the given pose and IoU it against the pinned label
     * (oracle_test.cpp's exact sequence).*/
    double IoUAt(const Point6D& pose) {
        const Pose p = ToPose(pose);
        model.SetCurrentPrimaryCameraPose(p);
        if (!model.RenderPrimaryCamera(p)) {
            return -1.0;
        }
        return metrics.IOU(model.GetPrimaryCameraRenderedImage(), &label);
    }
};

}  // namespace

int main(int argc, char* argv[]) {
    /*The manager's OptimizedFrame/UpdateDisplay/finished signals need a
     * main-thread event loop (queued connections from the worker thread).*/
    QCoreApplication app(argc, argv);
    return Catch::Session().run(argc, argv);
}

TEST_CASE(
    "U9 parity gate: app's OptimizerBridge run on Kneel_1 frame 0 with the "
    "oracle config lands within the oracle band (IoU >= 0.85)",
    "[oracle][gpu][parity]") {
    /*---- The app's own load path (StudyBridge three-action flow) ----*/
    ParityFixture f;
    StudyBridge* bridge = f.hub.studyBridge();
    ExperimentalSession* session = f.hub.session();

    bridge->loadCalibration(kCalibrationPath);
    REQUIRE(bridge->hasCalibration());
    REQUIRE(f.messages.titles.isEmpty());

    bridge->loadImages({QString::fromStdString(kFrameImage)});
    REQUIRE(bridge->frameCount() == 1);
    REQUIRE(bridge->currentFrame() == 0);
    REQUIRE(f.messages.titles.isEmpty());

    /*Oracle edge config (parity protocol): StudyBridge hard-codes the load
     * edge params {3,40,120,0}; the oracle's measured config is Canny
     * 3/0/150 + dilation 6 (baseline.json run_config). Replace the
     * app-loaded frame with the oracle-config Frame + compute the curvature
     * heatmaps (the widgets segment post-processing / oracle
     * BuildFramePipeline parity — the Frame ctor leaves
     * num_curvature_keypoints_ uninitialized and OptimizerManager::
     * Initialize uploads the heatmaps unconditionally).*/
    session->loaded_frames[0] = Frame(kFrameImage, /*aperture=*/3,
                                      /*low=*/0, /*high=*/150,
                                      /*dilation=*/6);
    session->loaded_frames[0].setCurvatureHeatmaps();

    bridge->loadModels({QString::fromStdString(kFemStl)});
    REQUIRE(bridge->modelCount() == 1);
    REQUIRE(session->loaded_models.size() == 1);
    REQUIRE(session->loaded_models[0].initialized_correctly_);
    REQUIRE(f.messages.titles.isEmpty());

    /*Start pose: fem.jts frame 0 via pose_file_io, set into LocationStorage
     * BEFORE the run + the scene mirror (the widgets Load_Pose then optimize
     * flow — run()'s SaveLastPose mirror then persists the same pose, so a
     * fresh-load default pose can never seed the run).*/
    std::vector<std::optional<Point6D>> fem_poses;
    const jta::pose_file::LoadResult load =
        jta::pose_file::ReadKinematicsFile(kFemJts, fem_poses);
    REQUIRE(load.ok);
    REQUIRE(fem_poses.size() >= 1);
    REQUIRE(fem_poses[0].has_value());
    const Point6D start = *fem_poses[0];
    REQUIRE(session->model_locations.GetFrameCount() == 1);
    session->model_locations.SavePose(0, 0, start);
    f.scene.setModelPose(0, start);

    /*Selection: frame 0 (default after load) + the femur as the primary
     * model (single-model v1 mode).*/
    bridge->toggleModelSelected(0);
    REQUIRE(bridge->primaryModelIndex() == 0);
    REQUIRE(bridge->selectedModelCount() == 1);

    /*---- The parity OptimizerSettings (the oracle config) ----*/
    SettingsBridge* settings = f.hub.settingsBridge();
    settings->setTrunkBudget(3000);
    settings->setTrunkRangeX(12.0);
    settings->setTrunkRangeY(12.0);
    settings->setTrunkRangeZ(15.0);
    settings->setTrunkRangeXA(15.0);
    settings->setTrunkRangeYA(15.0);
    settings->setTrunkRangeZA(15.0);
    /*The plan's primary single-stage-equivalent: branch/leaf ENABLED with
     * budget 0 (RunDirectStage at budget 0 = one cost eval, no search, no
     * error, running optimum preserved — verified against direct_optimizer
     * .cpp's loop guard). Active cost function is DIRECT_DILATION in all
     * three managers (the SettingsBridge constructor default). Plan 008 U9:
     * this degenerate 3000/0/0 shape runs through the script-driven loop —
     * BuildStageScript(settings, "Single") yields [Trunk 3000, Branch x2
     * budget 0, Leaf budget 0]; the budget-0 stages contribute exactly one
     * seed eval each (the cumulative (calls + offset) < budget_ guard trips
     * immediately), so the bridge run path must stay green through Cut B —
     * this instrument is the proof.*/
    settings->setBranchBudget(0);
    settings->setLeafBudget(0);
    settings->setTrunkDilation(6);  // DIRECT_DILATION Dilation param (default)

    /*---- The run: OptimizerBridge::run() end to end ----*/
    std::cout << "[parity] fem.jts start pose (frame 0): (" << start.x << ", "
              << start.y << ", " << start.z << ", " << start.xa << ", "
              << start.ya << ", " << start.za << ")" << std::endl;

    const OptimizerBridge::RunState terminal = RunAndWait(f.hub.optimizerBridge());

    /*Wiring correctness (plan 005 U9 test scenario b): the run reached the
     * Completed state — any Error here (intent-gate rejection, Initialize
     * failure, OptimizerError) is an app-wiring regression, not a backend
     * gap.*/
    INFO("terminal run state: " << static_cast<int>(terminal)
         << " (3=Completed, 4=Error); messages: "
         << f.messages.titles.join(" | ").toStdString());
    REQUIRE(terminal == OptimizerBridge::RunState::Completed);
    REQUIRE(f.messages.titles.isEmpty());

    /*SavePose landed: applyOptimizedFrame wrote the session storage + the
     * scene at the final pose (the app's result path).*/
    const Point6D recovered = session->model_locations.GetPose(0, 0);
    const Point6D scene_pose = f.scene.models()[0].pose;
    REQUIRE(recovered.x == scene_pose.x);
    REQUIRE(recovered.y == scene_pose.y);
    REQUIRE(recovered.z == scene_pose.z);
    REQUIRE(recovered.xa == scene_pose.xa);
    REQUIRE(recovered.ya == scene_pose.ya);
    REQUIRE(recovered.za == scene_pose.za);
    /*The optimizer actually moved: the recovered pose must differ from the
     * fresh-load default pose (0,0,-0.25*1198/0.373,0,0,0).*/
    const Point6D default_pose(0, 0, -0.25 * 1198.0 / 0.373, 0, 0, 0);
    const bool optimizer_moved =
        recovered.x != default_pose.x || recovered.y != default_pose.y ||
        recovered.z != default_pose.z || recovered.xa != default_pose.xa ||
        recovered.ya != default_pose.ya || recovered.za != default_pose.za;
    REQUIRE(optimizer_moved);

    std::cout << "[parity] cost calls: " << f.hub.optimizerBridge()->costCalls()
              << std::endl;
    std::cout << "[parity] recovered pose: (" << recovered.x << ", "
              << recovered.y << ", " << recovered.z << ", " << recovered.xa
              << ", " << recovered.ya << ", " << recovered.za << ")"
              << std::endl;
    std::cout << "[parity] gap vs fem.jts (informational): ("
              << recovered.x - start.x << ", " << recovered.y - start.y
              << ", " << recovered.z - start.z << ", "
              << recovered.xa - start.xa << ", " << recovered.ya - start.ya
              << ", " << recovered.za - start.za << ")" << std::endl;

    /*---- The comparator: render at the final pose (backface OFF — the
     * oracle config) and IoU vs the pinned label (oracle mechanics). ----*/
    Model& femur = session->loaded_models[0];
    const int triangle_count =
        static_cast<int>(femur.triangle_vertices_.size() / 9);
    REQUIRE(triangle_count > 0);
    /*The app-loaded calibration (test/golden/calibration.txt: principal
     * distance 1198, pixel pitch 0.373 — the oracle's pinned camera).*/
    Comparator cmp(femur.triangle_vertices_, femur.triangle_normals_,
                   triangle_count, session->calibration_file.camera_A_principal_);
    REQUIRE(cmp.ok());

    /*Diagnostic: the fem.jts start-pose silhouette must match the pinned
     * label well — pins the frame<->label pairing empirically in this binary
     * (oracle parity: 1.0 there; the 0.50 floor mirrors oracle_test.cpp).*/
    const double start_iou = cmp.IoUAt(start);
    std::cout << "[parity] start-pose IoU vs pinned label = " << start_iou
              << std::endl;
    REQUIRE(start_iou > kStartPoseIouFloor);

    /*THE LOAD-BEARING APPEARANCE GATE: the app-run final pose must land
     * within the oracle's band (observed oracle frame-0 recovered IoU =
     * 0.9936; the 0.85 gate has a healthy margin for hardware/float
     * variance).*/
    const double iou = cmp.IoUAt(recovered);
    const double l1 = cmp.metrics.L_1_1_MatrixDifferenceNorm(
        cmp.model.GetPrimaryCameraRenderedImage(), &cmp.label);
    const double per_px = l1 / (double)(kWidth * kHeight);
    std::cout << "[parity] recovered-pose IoU vs pinned label = " << iou
              << std::endl;
    std::cout << "[parity] recovered-pose L1 pixel-diff = " << l1
              << " (per-px " << per_px << ")" << std::endl;
    INFO("recovered IoU " << iou << " vs gate " << kIouGate
         << "; start-pose correspondence " << start_iou);
    REQUIRE(iou >= kIouGate);
}
