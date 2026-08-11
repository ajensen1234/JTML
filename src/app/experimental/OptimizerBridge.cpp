// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U6: OptimizerBridge implementation — see the header for the contract.
// Every step mirrors the widgets MainScreen::LaunchOptimizer drive sequence
// (mainscreen.cpp:4135) with the same connect set; the seam is untouched
// (R13 — no change to OptimizerManager).

#include "OptimizerBridge.h"

#include <QAbstractItemModel>
#include <QThread>

#include <algorithm>
#include <cmath>

// The real optimizer seam (coordinator lib). Included first: the header
// pulls CostFunctionManager.h (torch ATen headers).
#include "coordinator/optimizer_manager.h"
#include "services/save_last_pose.h"

#include "AppBridge.h"
#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "SettingsBridge.h"
#include "StudyBridge.h"
#include "view/model_list_model.h" // complete type for the index() build

namespace {

/*Widgets onUpdateDisplay level classification (mainscreen.cpp:4521-4550):
 * cumulative-budget arithmetic — Trunk / Branch n / Extra Z-Translation /
 * Finished. The widgets divides by branch_budget unguarded; the guard below
 * only prevents a divide-by-zero on degenerate settings, the labels are
 * identical.*/
QString StageLabel(const OptimizerSettings& s, int calls) {
    const int branch_budget =
        s.enable_branch_ ? std::max(1, s.branch_budget) : 1;
    const int branch_total =
        s.enable_branch_ ? s.number_branches * s.branch_budget : 0;
    if (calls < s.trunk_budget) {
        return QStringLiteral("Trunk");
    }
    if (calls < s.trunk_budget + branch_total) {
        return QStringLiteral("Branch %1")
            .arg((calls - s.trunk_budget) / branch_budget + 1);
    }
    if (calls < s.trunk_budget + branch_total +
                   (s.enable_leaf_ ? s.leaf_budget : 0)) {
        return QStringLiteral("Extra Z-Translation");
    }
    return QStringLiteral("Finished");
}

}  // namespace

/*---- Gate (headless-testable core) ----*/

OptimizerBridge::GateResult OptimizerBridge::EvaluateGate(
    const jta::OptimizeIntentController::Input& in) {
    GateResult result;
    /*The widget-free controller's two guards first (mirror of
     * LaunchOptimizer, mainscreen.cpp:4152-4186).*/
    result.intent = jta::OptimizeIntentController::Evaluate(in);
    if (result.intent.status ==
        jta::OptimizeIntentController::Status::SelectFrameAndModel) {
        result.status = GateStatus::SelectFrameAndModel;
        return result;
    }
    /*v1 single-model-mode rule (plan 005 U6 review fix): pose ops are
     * primary-model-only in v1, so a multi-select run is rejected (the
     * widgets' single_model_radio semantics, pinned for the QML app).*/
    if (in.selected_model_rows.size() != 1) {
        result.status = GateStatus::SingleModelOnly;
        return result;
    }
    if (result.intent.status ==
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
      hub_(hub),
      session_(session),
      scene_(scene),
      study_bridge_(study_bridge),
      settings_bridge_(settings_bridge) {}

OptimizerBridge::~OptimizerBridge() = default;

void OptimizerBridge::run() {
    /*Re-run guard (plan 005 U6 review fix): no second manager mid-run. The
     * widgets has no equivalent because DisableAll gates the UI; QML bindings
     * can race, so the bridge guards itself.*/
    if (state_ == RunState::Running || state_ == RunState::Stopping) {
        return;
    }

    /*SaveLastPose mirror (mainscreen.cpp:4137): persist the live scene poses
     * of the current frame's selected models (ModelMode interaction can
     * drift the scene ahead of the storage) before the gate.*/
    saveScenePosesForCurrentSelection();

    /*Entry gate — mirrors the LaunchOptimizer gates (frame selected, models
     * selected, single-model mode for pose ops, PoseMatrixDimensionMismatch).
     * Rejections surface through the single QML Dialog mechanism and change
     * nothing (no manager, no thread, state stays as-is).*/
    const GateResult gate = EvaluateGate(buildGateInput());
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

    /*U7: the ML-estimate starting-pose seed (R8 — the estimate seeds the
     * optimizer). Applied AFTER the gate so a rejected run never consumes
     * it, and after the SaveLastPose mirror so the estimate wins over scene
     * drift; the manager's Optimize() reads its starting point from the
     * by-value pose matrix Initialize copies below (the widgets equivalent:
     * the estimate slots' SavePose feeding LaunchOptimizer).*/
    applySeedPose();

    /*Thread lifecycle (mirror of mainscreen.cpp:4186-4189): a fresh manager
     * + thread per run. Initialize() (called below, before the start — the
     * plan's pinned order) wires started->Optimize and the finished->
     * quit/deleteLater cleanup chain internally.*/
    manager_ = new OptimizerManager();
    optimizer_thread_ = new QThread();
    manager_->moveToThread(optimizer_thread_);

    /*Initialize with the app's containers BY VALUE + the QModelIndexList
     * built from the direct-compiled model list WITHOUT QItemSelectionModel
     * (model->index(row, 0) — the plan's verified trick). v1 is monoplane:
     * the camera-B frame list is empty. Directive is always "Single" (v1
     * run scope = the current frame only).*/
    QString error_message;
    const bool initialized_correctly = manager_->Initialize(
        *optimizer_thread_,
        session_->calibration_file,
        session_->loaded_frames,
        {},  // camera-B frames: monoplane v1
        static_cast<unsigned int>(gate.intent.current_frame),
        session_->loaded_models,
        selectedModelIndexes(),
        static_cast<unsigned int>(gate.intent.primary_model_index),
        session_->model_locations,
        settings_bridge_->optimizerSettings(),
        *settings_bridge_->trunkManager(),
        *settings_bridge_->branchManager(),
        *settings_bridge_->leafManager(),
        QStringLiteral("Single"),
        error_message,
        0 /* iter_count */);

    if (!initialized_correctly) {
        /*R13-preserved quirk (mirror of mainscreen.cpp:4222-4231): the
         * thread is started BEFORE the error box and early return, so a
         * failed Initialize leaks BOTH the manager (never deleted) and the
         * thread (started, never quit/waited). Behavior preserved
         * deliberately — the fix is the deferred follow-up cut (plan
         * "Deferred to Follow-Up Work").*/
        optimizer_thread_->start();
        emit messageRequested(QStringLiteral("Error!"), error_message);
        setState(RunState::Error);
        return;
    }

    /*The 7 binds (mirror of mainscreen.cpp:4234-4303): 6 manager signals
     * relayed to QML + the app->manager StopOptimizer reverse bind. New-style
     * member-pointer connects — same semantics as the widgets SIGNAL/SLOT
     * strings (Auto for the relays, DirectConnection for the stop).*/
    connect(
        manager_,
        &OptimizerManager::UpdateDisplay,
        this,
        &OptimizerBridge::onUpdateDisplay);
    connect(
        manager_,
        &OptimizerManager::OptimizerError,
        this,
        &OptimizerBridge::onOptimizerError);
    connect(
        manager_,
        &OptimizerManager::UpdateOptimum,
        this,
        &OptimizerBridge::onUpdateOptimum);
    connect(
        manager_,
        &OptimizerManager::OptimizedFrame,
        this,
        &OptimizerBridge::onOptimizedFrame);
    connect(
        this,
        &OptimizerBridge::StopOptimizer,
        manager_,
        &OptimizerManager::onStopOptimizer,
        Qt::DirectConnection);
    connect(
        manager_,
        &OptimizerManager::UpdateDilationBackground,
        this,
        &OptimizerBridge::onUpdateDilationBackground);
    connect(
        manager_,
        &OptimizerManager::onUpdateOrientationSymTrap,
        this,
        &OptimizerBridge::onOrientationSymTrap);
    /*finished() — the plan's optional completion hook: waits the thread out
     * so a re-run never races the old one.*/
    connect(
        manager_,
        &OptimizerManager::finished,
        this,
        &OptimizerBridge::onFinished);

    setState(RunState::Running);
    optimizer_thread_->start();
}

void OptimizerBridge::stop() {
    if (state_ != RunState::Running && state_ != RunState::Stopping) {
        return;
    }
    if (state_ == RunState::Running) {
        /*DirectConnection reverse bind — mirrors the widgets
         * on_actionStop_Optimizer_triggered (mainscreen.cpp:1469-1475):
         * onStopOptimizer just flips the worker's error flag; the run
         * completes through the normal OptimizedFrame/finished path.*/
        emit StopOptimizer();
    }
    setState(RunState::Stopping);
}

/*---- U7: ML-estimate starting-pose seed ----*/

void OptimizerBridge::setSeedPose(
    double x, double y, double z, double xa, double ya, double za) {
    seed_pose_ = Point6D(x, y, z, xa, ya, za);
    seed_frame_ = study_bridge_->currentFrame();
    seed_model_ = study_bridge_->primaryModelIndex();
    has_seed_pose_ = true;
}

void OptimizerBridge::clearSeedPose() {
    has_seed_pose_ = false;
}

void OptimizerBridge::applySeedPose() {
    if (!has_seed_pose_) {
        return;
    }
    /*One-shot + stale guards: the seed applies only when the run's frame is
     * still the seeded frame and the seeded model is still the primary
     * selection (MlBridge clears the seed on any selection change, but the
     * guard keeps the contract self-contained). Any other state drops the
     * seed silently — a stale-frame estimate must never override a
     * different frame's pose.*/
    if (study_bridge_->currentFrame() != seed_frame_ ||
        study_bridge_->primaryModelIndex() != seed_model_ ||
        seed_model_ < 0 ||
        seed_model_ >= static_cast<int>(session_->loaded_models.size())) {
        clearSeedPose();
        return;
    }
    session_->model_locations.SavePose(seed_frame_, seed_model_, seed_pose_);
    scene_->setModelPose(seed_model_, seed_pose_);
    has_seed_pose_ = false;
}

/*---- State + progress reads ----*/

OptimizerBridge::RunState OptimizerBridge::runState() const {
    return state_;
}

bool OptimizerBridge::running() const {
    return state_ == RunState::Running || state_ == RunState::Stopping;
}

bool OptimizerBridge::canRun() const {
    return state_ == RunState::Idle || state_ == RunState::Completed ||
           state_ == RunState::Error;
}

QString OptimizerBridge::stageText() const {
    return stage_text_;
}

int OptimizerBridge::costCalls() const {
    return cost_calls_;
}

double OptimizerBridge::currentMinimum() const {
    return current_minimum_;
}

double OptimizerBridge::progress() const {
    return progress_;
}

/*---- Headless-testable core ----*/

jta::OptimizeIntentController::Input OptimizerBridge::buildGateInput() const {
    jta::OptimizeIntentController::Input in;
    /*Selected model rows (any order; ascending from the delegate set).*/
    const QVariantList selected = study_bridge_->selectedModels();
    in.selected_model_rows.reserve(selected.size());
    for (const QVariant& row : selected) {
        in.selected_model_rows.push_back(row.toInt());
    }
    /*The QML app has no separate last-viewed frame (the selection IS the
     * current view — setCurrentFrame fires immediately), so previous ==
     * current, matching the controller's Ok precondition.*/
    in.previous_frame_index = study_bridge_->currentFrame();
    in.current_frame = in.previous_frame_index;
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

void OptimizerBridge::applyOptimizedFrame(
    double x, double y, double z, double xa, double ya, double za,
    bool move_next_frame, unsigned int primary_model_index,
    bool error_occurred, const QString& optimizer_directive) {
    /*v1 run scope: the current frame only (directive Single — the manager's
     * Initialize sets progress_next_frame_ false for Single, so
     * move_next_frame never arrives true; the All/Each/From/Backward
     * advance loop is deferred per the plan's review fix).*/
    Q_UNUSED(move_next_frame);
    Q_UNUSED(error_occurred);
    Q_UNUSED(optimizer_directive);

    /*SavePose mirror of onOptimizedFrame (mainscreen.cpp:4383-4502): the
     * result lands in the session's LocationStorage (the manager worked on
     * a by-value copy) and the scene pose updates for the viewport re-render
     * at the result pose.*/
    const int frame_index = study_bridge_->currentFrame();
    if (primary_model_index < session_->loaded_models.size()) {
        const Point6D pose(x, y, z, xa, ya, za);
        session_->model_locations.SavePose(
            frame_index, static_cast<int>(primary_model_index), pose);
        scene_->setModelPose(static_cast<int>(primary_model_index), pose);
    }
    emit frameOptimized(frame_index, static_cast<int>(primary_model_index));

    /*Final state: completed unless an OptimizerError already moved the run
     * to error (a stopped run — stop() set Stopping — completes re-runnable,
     * plan 005 U6 test scenario d).*/
    if (state_ != RunState::Error) {
        setState(RunState::Completed);
    }
}

/*---- Relay slots (the 7 binds) ----*/

void OptimizerBridge::onUpdateDisplay(
    double iteration_speed,
    int current_iteration,
    double current_minimum,
    unsigned int primary_model_index) {
    Q_UNUSED(iteration_speed);
    Q_UNUSED(primary_model_index);
    refreshProgress(current_iteration, current_minimum);
}

void OptimizerBridge::onOptimizerError(const QString& error_message) {
    setState(RunState::Error);
    emit messageRequested(QStringLiteral("Error!"), error_message);
}

void OptimizerBridge::onUpdateOptimum(
    double x,
    double y,
    double z,
    double xa,
    double ya,
    double za,
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

void OptimizerBridge::onOptimizedFrame(
    double x, double y, double z, double xa, double ya, double za,
    bool move_next_frame, unsigned int primary_model_index,
    bool error_occurred, const QString& optimizer_directive) {
    applyOptimizedFrame(
        x, y, z, xa, ya, za, move_next_frame, primary_model_index,
        error_occurred, optimizer_directive);
}

void OptimizerBridge::onUpdateDilationBackground() {
    /*v1 relay: the widgets re-applies the dilation display (mainscreen.cpp:
     * 4616); the QML app has no dilation display mode yet — QML may ignore.*/
    emit dilationBackgroundRequested();
}

void OptimizerBridge::onOrientationSymTrap(
    double x, double y, double z, double xa, double ya, double za) {
    /*Sym-trap directive is not in v1 — relay anyway (plan 005 U6).*/
    emit orientationSymTrapUpdated(x, y, z, xa, ya, za);
}

void OptimizerBridge::onFinished() {
    /*The manager's internal chain (wired in Initialize) already quit the
     * thread and deleteLater'd both objects; wait it out so a re-run never
     * races the old thread (mirror of the widgets lifecycle). QPointer
     * guards the failure-quirk ordering.*/
    if (optimizer_thread_) {
        optimizer_thread_->wait();
        optimizer_thread_ = nullptr;
    }
    manager_ = nullptr;
}

/*---- Private helpers ----*/

void OptimizerBridge::setState(RunState state) {
    if (state_ == state) {
        return;
    }
    state_ = state;
    emit runStateChanged();
}

/*SaveLastPose mirror (mainscreen.cpp:4101-4117 -> shared core, plan 006
 * U3): persist the scene poses of the current frame's selected models into
 * the storage so the optimizer initializes from the live view (ModelMode
 * interaction can drift the scene ahead of the storage). No-op with no
 * current frame. QML call-site table row: current selection, current frame,
 * scene source, never convert. The old row-range guard (model_row >= 0 &&
 * < models.size()) now lives in the lambda as the all-zero sentinel the
 * core skips.*/
void OptimizerBridge::saveScenePosesForCurrentSelection() {
    const int frame = study_bridge_->currentFrame();
    if (frame < 0) {
        return;
    }
    const std::vector<SceneModel> models = scene_->models();
    const QVariantList selected = study_bridge_->selectedModels();
    std::vector<int> rows;
    rows.reserve(static_cast<size_t>(selected.size()));
    for (const QVariant& row : selected) {
        rows.push_back(row.toInt());
    }
    jta::SaveLastPoseToStorage(
        frame,
        rows,
        [&models](int model_row) {
            if (model_row >= 0 &&
                model_row < static_cast<int>(models.size())) {
                return models[static_cast<size_t>(model_row)].pose;
            }
            return Point6D(); /* out-of-range -> core skips the row */
        },
        /*camera_is_a: monoplane v1 — camera A is the only camera; unused by
         * NeverConvert.*/
        true,
        jta::SavePoseConvertRule::NeverConvert,
        session_->calibration_file,
        session_->model_locations);
}

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

void OptimizerBridge::refreshProgress(int calls, double minimum) {
    cost_calls_ = calls;
    current_minimum_ = minimum;
    const OptimizerSettings& s = settings_bridge_->optimizerSettings();
    const int cumulative =
        s.trunk_budget +
        (s.enable_branch_ ? s.number_branches * s.branch_budget : 0) +
        (s.enable_leaf_ ? s.leaf_budget : 0);
    stage_text_ = StageLabel(s, calls);
    progress_ = cumulative > 0 ? std::min(1.0, calls / double(cumulative)) : 0.0;
    emit progressChanged();
}
