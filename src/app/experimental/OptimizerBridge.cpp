// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U6 / 006 U5: OptimizerBridge implementation — the QML-facing thin
// shell over the shared OptimizerRunController (see the header for the
// contract). Every step of the drive sequence (SaveLastPose mirror ->
// intent gate -> seed -> fresh manager + thread -> Initialize by value ->
// the 8 binds -> start) now lives in the controller; this bridge keeps the
// QML surface (Q_PROPERTYs, SingleModelOnly pre-check, Dialog mapping, the
// scene pose writes) and the app->controller wiring. The seam is untouched
// (R13 — no change to OptimizerManager).

#include "OptimizerBridge.h"

#include <QAbstractItemModel>

#include <algorithm>

// The shared controller shell + driver seam (coordinator lib). Included
// first: the driver header pulls CostFunctionManager.h (torch ATen headers).
#include "coordinator/optimizer_run_controller.h"

#include "AppBridge.h"
#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "SettingsBridge.h"
#include "StudyBridge.h"
#include "view/model_list_model.h" // complete type for the index() build

/*---- Gate (bridge policy: the shared gate + the v1 single-model rule) ----*/

OptimizerBridge::GateResult OptimizerBridge::EvaluateGate(
    const jta::OptimizerRunControllerCore::GateInput& in) {
    GateResult result;
    /*The shared gate's first guard first (mirror of LaunchOptimizer,
     * mainscreen.cpp:4152-4186).*/
    const auto shared = jta::OptimizerRunControllerCore::EvaluateGate(in);
    result.intent = shared.intent;
    if (shared.status ==
        jta::OptimizeIntentController::Status::SelectFrameAndModel) {
        result.status = GateStatus::SelectFrameAndModel;
        return result;
    }
    /*v1 single-model-mode rule (plan 005 U6 review fix): pose ops are
     * primary-model-only in v1, so a multi-select run is rejected (the
     * widgets' single_model_radio semantics, pinned for the QML app). A
     * bridge policy — NOT in the shared controller.*/
    if (in.selected_model_rows.size() != 1) {
        result.status = GateStatus::SingleModelOnly;
        return result;
    }
    if (shared.status ==
        jta::OptimizeIntentController::Status::PoseMatrixDimensionMismatch) {
        result.status = GateStatus::PoseMatrixDimensionMismatch;
        return result;
    }
    result.status = GateStatus::Ok;
    return result;
}

/*---- Lifecycle ----*/

OptimizerBridge::OptimizerBridge(
    AppBridge* hub,
    ExperimentalSession* session,
    ExperimentalScene* scene,
    StudyBridge* study_bridge,
    SettingsBridge* settings_bridge,
    QObject* parent)
    : QObject(parent),
      controller_(new OptimizerRunController(jta::CreateOptimizerManagerRunDriver, this)),
      hub_(hub),
      session_(session),
      scene_(scene),
      study_bridge_(study_bridge),
      settings_bridge_(settings_bridge) {
    /*Controller relays -> the QML surface (re-emitted on this thread; the
     * controller already re-emitted the manager's worker-thread signals
     * by-value on its own thread — QTBUG-2842).*/
    connect(
        controller_, &OptimizerRunController::runStateChanged,
        this, &OptimizerBridge::onControllerRunStateChanged);
    connect(
        controller_, &OptimizerRunController::progressChanged,
        this, &OptimizerBridge::onControllerProgressChanged);
    connect(
        controller_, &OptimizerRunController::messageRequested,
        this, &OptimizerBridge::onControllerMessage);
    connect(
        controller_, &OptimizerRunController::poseUpdated,
        this, &OptimizerBridge::onControllerPoseUpdated);
    connect(
        controller_, &OptimizerRunController::optimizedFrameRelayed,
        this, &OptimizerBridge::onControllerOptimizedFrame);
    connect(
        controller_, &OptimizerRunController::dilationBackgroundRequested,
        this, &OptimizerBridge::onControllerDilationBackground);
    connect(
        controller_, &OptimizerRunController::orientationSymTrapUpdated,
        this, &OptimizerBridge::onControllerOrientationSymTrap);
    connect(
        controller_, &OptimizerRunController::seedApplied,
        this, &OptimizerBridge::onControllerSeedApplied);
    connect(
        controller_, &OptimizerRunController::seedRestored,
        this, &OptimizerBridge::onControllerSeedRestored);
}

OptimizerBridge::~OptimizerBridge() = default;

void OptimizerBridge::run() {
    /*Re-run guard (plan 005 U6 review fix): no second manager mid-run. The
     * controller's Start gate adds the ghost-window rejection with a
     * message (H1 — the enumerated QML delta).*/
    if (controller_->running()) {
        return;
    }

    /*v1 pre-check (bridge policy): the shared gate's first guard, then the
     * single-model rule, then the second guard. Pure — no side effects; the
     * controller's start() runs the SaveLastPose mirror + the shared gate
     * again in the pinned order (mirror before gate).*/
    const jta::OptimizerRunControllerCore::GateInput gate_in = buildGateInput();
    const GateResult gate = EvaluateGate(gate_in);
    if (gate.status == GateStatus::SelectFrameAndModel) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Select Frame and Model First!"));
        return;
    }
    if (gate.status == GateStatus::SingleModelOnly) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Single-model mode: select exactly one model "
                           "(v1)"));
        return;
    }
    if (gate.status == GateStatus::PoseMatrixDimensionMismatch) {
        emit messageRequested(
            QStringLiteral("Critical Error!"),
            QStringLiteral("Pose Dimension Matrix Differs in Size from Frame "
                           "and Models Loaded! Please Contact Support!"));
        return;
    }

    /*The shared controller: SaveLastPose mirror -> gate -> seed -> fresh
     * driver (manager + thread) -> finished bound before Initialize -> the
     * 7 binds -> Initialize by value -> thread start.*/
    OptimizerRunRequest req;
    req.directive = OptimizerRunController::Directive::Single;

    /*SaveLastPose mirror (mainscreen.cpp:4137 / U3 core): persist the live
     * scene poses of the current frame's selected models before the gate
     * (QML call-site table row: current selection, current frame, scene
     * source, never convert).*/
    req.save_frame = study_bridge_->currentFrame();
    const QVariantList selected = study_bridge_->selectedModels();
    req.save_rows.reserve(static_cast<size_t>(selected.size()));
    for (const QVariant& row : selected) {
        req.save_rows.push_back(row.toInt());
    }
    const std::vector<SceneModel> scene_models = scene_->models();
    req.save_pose_source = [scene_models](int model_row) {
        if (model_row >= 0 &&
            model_row < static_cast<int>(scene_models.size())) {
            return scene_models[static_cast<size_t>(model_row)].pose;
        }
        return Point6D(); /* out-of-range -> the core skips the row */
    };
    req.camera_is_a = true;  // monoplane v1 — unused by NeverConvert
    req.save_convert_rule = jta::SavePoseConvertRule::NeverConvert;

    /*Gate input (H2: previous == current by construction).*/
    req.selected_model_rows = req.save_rows;
    req.current_frame = study_bridge_->currentFrame();
    req.frame_count = study_bridge_->frameCount();
    req.model_current_index =
        req.selected_model_rows.empty() ? -1 : req.selected_model_rows.front();
    req.model_count = study_bridge_->modelCount();
    req.pose_frame_count = session_->model_locations.GetFrameCount();
    req.pose_model_count = session_->model_locations.GetModelCount();
    req.storage = &session_->model_locations;

    /*Initialize payload (the app's containers BY VALUE + the QModelIndexList
     * built from the direct-compiled model list WITHOUT QItemSelectionModel
     * — model->index(row, 0)). v1 is monoplane: the camera-B frame list is
     * empty; the directive is always "Single".*/
    req.launch.calibration = session_->calibration_file;
    req.launch.camera_a_frames = session_->loaded_frames;
    req.launch.camera_b_frames = {};
    req.launch.models = session_->loaded_models;
    req.launch.selected_model_indexes = selectedModelIndexes();
    req.launch.pose_matrix = session_->model_locations;
    req.launch.settings = settings_bridge_->optimizerSettings();
    req.launch.trunk_manager = *settings_bridge_->trunkManager();
    req.launch.branch_manager = *settings_bridge_->branchManager();
    req.launch.leaf_manager = *settings_bridge_->leafManager();
    req.launch.iter_count = 0;

    controller_->start(req);
}

void OptimizerBridge::stop() {
    controller_->stop();
}

/*---- U7: ML-estimate starting-pose seed ----*/

void OptimizerBridge::setSeedPose(
    double x, double y, double z, double xa, double ya, double za) {
    controller_->setSeedPose(
        x, y, z, xa, ya, za, study_bridge_->currentFrame(),
        study_bridge_->primaryModelIndex());
}

void OptimizerBridge::clearSeedPose() {
    controller_->clearSeedPose();
}

bool OptimizerBridge::hasSeedPose() const {
    return controller_->hasSeedPose();
}

void OptimizerBridge::applySeedPose() {
    controller_->applySeedPose(
        &session_->model_locations, study_bridge_->currentFrame(),
        study_bridge_->primaryModelIndex(), study_bridge_->modelCount());
}

/*---- State + progress reads ----*/

OptimizerBridge::RunState OptimizerBridge::runState() const {
    return static_cast<RunState>(controller_->runState());
}

bool OptimizerBridge::running() const {
    return controller_->running();
}

bool OptimizerBridge::canRun() const {
    return controller_->canRun();
}

QString OptimizerBridge::stageText() const {
    return controller_->stageText();
}

int OptimizerBridge::costCalls() const {
    return controller_->costCalls();
}

double OptimizerBridge::currentMinimum() const {
    return controller_->currentMinimum();
}

double OptimizerBridge::progress() const {
    return controller_->progress();
}

/*---- Headless-testable core delegates ----*/

jta::OptimizerRunControllerCore::GateInput OptimizerBridge::buildGateInput()
    const {
    jta::OptimizerRunControllerCore::GateInput in;
    /*Selected model rows (any order; ascending from the delegate set).*/
    const QVariantList selected = study_bridge_->selectedModels();
    in.selected_model_rows.reserve(static_cast<size_t>(selected.size()));
    for (const QVariant& row : selected) {
        in.selected_model_rows.push_back(row.toInt());
    }
    /*The QML app has no separate last-viewed frame (the selection IS the
     * current view — setCurrentFrame fires immediately), so previous ==
     * current, matching the shared gate's Ok precondition (H2).*/
    in.current_frame = study_bridge_->currentFrame();
    in.frame_count = study_bridge_->frameCount();
    /*Widgets parity: model_current_index is the model list's current row —
     * the primary (first selected) row when a selection exists.*/
    in.model_current_index =
        in.selected_model_rows.empty() ? -1 : in.selected_model_rows.front();
    in.model_count = study_bridge_->modelCount();
    in.pose_frame_count = session_->model_locations.GetFrameCount();
    in.pose_model_count = session_->model_locations.GetModelCount();
    return in;
}

/*---- Relay slots (the controller's re-emitted binds) ----*/

void OptimizerBridge::onControllerRunStateChanged() {
    emit runStateChanged();
}

void OptimizerBridge::onControllerProgressChanged() {
    emit progressChanged();
}

void OptimizerBridge::onControllerMessage(
    const QString& title, const QString& message,
    jta::OptimizerRunControllerCore::Severity /*severity*/) {
    /*Dialog mapping: the QML app has one Dialog; the severity is ignored
     * (L14 — the widgets preserves its box-type distinctions).*/
    emit messageRequested(title, message);
}

void OptimizerBridge::onControllerPoseUpdated(
    double x, double y, double z, double xa, double ya, double za,
    unsigned int primary_model_index) {
    /*Live pose bind (mirror of onUpdateOptimum, mainscreen.cpp:4359-4391):
     * the primary model follows the search. The scene write + poseUpdated
     * relay drive the viewport re-render (main.qml glue -> updatePose).*/
    if (primary_model_index < session_->loaded_models.size()) {
        scene_->setModelPose(
            static_cast<int>(primary_model_index),
            Point6D(x, y, z, xa, ya, za));
    }
    emit poseUpdated(static_cast<int>(primary_model_index));
}

void OptimizerBridge::onControllerOptimizedFrame(
    double x, double y, double z, double xa, double ya, double za,
    bool move_next_frame, unsigned int primary_model_index,
    bool error_occurred, const QString& optimizer_directive,
    bool model_out_of_bounds) {
    /*v1 run scope: the current frame only (directive Single — the manager's
     * Initialize sets progress_next_frame_ false for Single, so
     * move_next_frame never arrives true; the All/Each/From/Backward
     * advance loop is deferred per the plan's review fix). The controller
     * already persisted the pose in the session storage + moved the run to
     * its terminal state; this bridge maps the scene + the QML relay.*/
    Q_UNUSED(move_next_frame);
    Q_UNUSED(error_occurred);
    Q_UNUSED(optimizer_directive);
    const int frame_index = study_bridge_->currentFrame();
    if (!model_out_of_bounds &&
        primary_model_index < session_->loaded_models.size()) {
        scene_->setModelPose(
            static_cast<int>(primary_model_index),
            Point6D(x, y, z, xa, ya, za));
    }
    emit frameOptimized(frame_index, static_cast<int>(primary_model_index));
}

void OptimizerBridge::onControllerDilationBackground() {
    /*v1 relay: the widgets re-applies the dilation display (mainscreen.cpp:
     * 4616); the QML app has no dilation display mode yet — QML may ignore.*/
    emit dilationBackgroundRequested();
}

void OptimizerBridge::onControllerOrientationSymTrap(
    double x, double y, double z, double xa, double ya, double za) {
    /*Sym-trap directive is not in v1 — relay anyway (plan 005 U6).*/
    emit orientationSymTrapUpdated(x, y, z, xa, ya, za);
}

void OptimizerBridge::onControllerSeedApplied(int frame, int model) {
    /*The controller wrote the pending seed to the storage; mirror it onto
     * the scene (the run's viewport shows the estimate pose).*/
    Q_UNUSED(frame);
    if (model >= 0 && model < static_cast<int>(session_->loaded_models.size())) {
        scene_->setModelPose(model, session_->model_locations.GetPose(frame, model));
    }
}

void OptimizerBridge::onControllerSeedRestored(int frame, int model) {
    /*M10a (the enumerated QML delta): a failed Initialize restored the
     * pre-seed storage snapshot — re-sync the scene so the estimate is not
     * silently kept on screen either.*/
    if (model >= 0 && model < static_cast<int>(session_->loaded_models.size())) {
        scene_->setModelPose(model, session_->model_locations.GetPose(frame, model));
    }
}

/*---- Private helpers ----*/

/*The QModelIndexList for Initialize, built from the direct-compiled model
 * list WITHOUT QItemSelectionModel (model->index(row, 0) — the plan's
 * verified trick). The first row is the primary (DelegateSelection ascending
 * rule), matching Initialize's selected_models[0].row() == primary check.*/
QModelIndexList OptimizerBridge::selectedModelIndexes() const {
    QModelIndexList indexes;
    QAbstractItemModel* model = study_bridge_->modelListModel();
    const QVariantList selected = study_bridge_->selectedModels();
    for (const QVariant& row : selected) {
        indexes.push_back(model->index(row.toInt(), 0));
    }
    return indexes;
}
