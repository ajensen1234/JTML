// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U5: OptimizerRunController implementation — see the header for
// the contract. The drive sequence is the pinned widgets/QML sequence
// (mainscreen.cpp:4135 / OptimizerBridge.cpp:114) with the plan's
// lifecycle fixes: finished bound before Initialize (M6), epoch-tagged
// relay drops + threadActive Start gate (H1), destructor contract (H3),
// seed restore on Initialize failure (M10a).

#include "coordinator/optimizer_run_controller.h"

#include <QThread>
#include <utility>

#include "services/save_last_pose.h"

namespace {

const char* kSelectFrameAndModelText = "Select Frame and Model First!";
const char* kPoseDimensionMismatchText =
    "Pose Dimension Matrix Differs in Size from Frame and Models Loaded! "
    "Please Contact Support!";
const char* kReRunRejectedText =
    "Optimizer is still finishing the previous run!";
const char* kStillStoppingText = "Optimizer is still stopping...";

}  // namespace

/*---- Start (the full drive sequence) ----*/

bool OptimizerRunController::start(const OptimizerRunRequest& req) {
    /*Start gate (H1/M6): no run in flight + no previous thread alive (covers
     * the Initialize-failure ghost + the widgets' terminal-frame -> thread-
     * death window — the acknowledged re-run rejection delta). A run still
     * draining its cooperative stop gets its OWN distinct message (the user
     * pressed Run while Stop was in flight — review fix S1); the re-run
     * rejection text is reserved for a live thread outside the Stopping
     * drain (Running / Completed / Error with a thread still finishing).*/
    if (core_.state() == RunState::Stopping) {
        emit messageRequested(
            QStringLiteral("Warning!"),
            QString::fromLatin1(kStillStoppingText),
            Severity::Warning);
        return false;
    }
    if (core_.state() == RunState::Running ||
        (driver_ && driver_->ThreadActive())) {
        emit messageRequested(
            QStringLiteral("Warning!"),
            QString::fromLatin1(kReRunRejectedText),
            Severity::Warning);
        return false;
    }

    /*Initialize payload copy (BY VALUE) — the non-const calibration also
     * feeds the SaveLastPose mirror below (the U3 core takes Calibration&).*/
    jta::OptimizerRunLaunch launch = req.launch;

    /*SaveLastPose mirror (U3 core), pinned BEFORE the gate (widgets
     * mainscreen.cpp:4137 / QML OptimizerBridge.cpp:114).*/
    if (req.storage) {
        jta::SaveLastPoseToStorage(
            req.save_frame,
            req.save_rows,
            req.save_pose_source,
            req.camera_is_a,
            req.save_convert_rule,
            launch.calibration,
            *req.storage);
    }

    /*Entry gate (H2 — Input assembled with previous == current; the two
     * statuses surface with the widgets' distinct message texts, L14).*/
    jta::OptimizerRunControllerCore::GateInput gate_in;
    gate_in.selected_model_rows = req.selected_model_rows;
    gate_in.current_frame = req.current_frame;
    gate_in.frame_count = req.frame_count;
    gate_in.model_current_index = req.model_current_index;
    gate_in.model_count = req.model_count;
    gate_in.pose_frame_count = req.pose_frame_count;
    gate_in.pose_model_count = req.pose_model_count;
    const auto gate = jta::OptimizerRunControllerCore::EvaluateGate(gate_in);
    if (gate.status ==
        jta::OptimizeIntentController::Status::SelectFrameAndModel) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QString::fromLatin1(kSelectFrameAndModelText),
            Severity::Critical);
        return false;
    }
    if (gate.status ==
        jta::OptimizeIntentController::Status::PoseMatrixDimensionMismatch) {
        emit messageRequested(
            QStringLiteral("Critical Error!"),
            QString::fromLatin1(kPoseDimensionMismatchText),
            Severity::Critical);
        return false;
    }

    /*Seed (M10a): applied AFTER the gate so a rejected run never consumes
     * it; snapshot kept so an Initialize failure restores the pre-seed
     * pose (the estimate-wins-over-drift guarantee survives a failed run —
     * the estimate's own storage write is untouched).*/
    const auto seed = core_.takeSeedForRun(
        req.current_frame, gate.intent.primary_model_index, req.model_count);
    Point6D seed_snapshot;
    bool have_seed_snapshot = false;
    if (seed.applied && req.storage) {
        seed_snapshot = req.storage->GetPose(seed.frame, seed.model);
        have_seed_snapshot = true;
        req.storage->SavePose(seed.frame, seed.model, seed.pose);
        emit seedApplied(seed.frame, seed.model);
    }

    /*Payload refresh (review fix P1-2): the SaveLastPose mirror AND the
     * seed write above both landed in req.storage — the manager's
     * Initialize consumes launch.pose_matrix (the by-value copy the view
     * captured BEFORE start()), so the payload is refreshed here. Without
     * it the seed never reached the run (the manager started from the
     * stale/drifted pose) and the mirror's persistence missed the payload
     * vs the pre-refactor flow — the estimate-wins-over-drift guarantee
     * was false.*/
    if (req.storage) {
        launch.pose_matrix = *req.storage;
    }

    /*Run state + epoch, then a fresh driver per run (M12/Q8).*/
    core_.onRunStarted();
    emit runStateChanged();
    run_epoch_ = core_.epoch();
    /*Tracked frame (review fix P1-1): the manager's frame sequence starts
     * at 0 for All/Each (start_frame_index_ = 0, and the widgets view
     * reset its selection to 0 BEFORE start() — M11 two-phase), so the
     * terminal-frame persistence rows start at 0 there; the other
     * directives keep the run's current frame.*/
    current_frame_ =
        (req.directive == Directive::All || req.directive == Directive::Each)
        ? 0
        : req.current_frame;
    frame_count_ = req.frame_count;
    model_count_ = req.model_count;
    storage_ = req.storage;
    budgets_ = BudgetsFromSettings(req.launch.settings);
    driver_ = factory_();
    bindManager();

    /*Initialize with the app's containers BY VALUE + the QModelIndexList
     * built by the view; the gate intent packs the primary model + current
     * frame, and the typed directive maps to the widgets' strings.*/
    launch.current_frame_index =
        static_cast<unsigned int>(gate.intent.current_frame);
    launch.primary_model_index =
        static_cast<unsigned int>(gate.intent.primary_model_index);
    launch.directive = DirectiveToString(req.directive);
    launch.iter_count = req.iter_count;
    QString error_message;
    bool initialized_correctly = false;
    if (driver_) {
        initialized_correctly = driver_->Initialize(launch, error_message);
    }

    if (!initialized_correctly) {
        /*R13-preserved quirk (mainscreen.cpp:4216-4228): the thread is
         * started BEFORE the error box and early return, so a failed
         * Initialize's ghost thread runs and terminates through the
         * manager's internal finished chain — observed via the finished
         * bind placed BEFORE Initialize (M6) and waited at the next
         * onManagerFinished delivery.*/
        if (driver_) {
            driver_->Start();
        }
        /*M10a: restore the pre-seed snapshot on the failure path (the
         * estimate is not silently kept).*/
        if (have_seed_snapshot && req.storage) {
            req.storage->SavePose(seed.frame, seed.model, seed_snapshot);
            emit seedRestored(seed.frame, seed.model);
        }
        core_.onInitializeFailed();
        emit runStateChanged();
        emit messageRequested(
            QStringLiteral("Error!"), error_message, Severity::Critical);
        return false;
    }

    driver_->Start();
    return true;
}

void OptimizerRunController::stop() {
    if (core_.state() != RunState::Running &&
        core_.state() != RunState::Stopping) {
        return;
    }
    if (core_.state() == RunState::Running) {
        /*DirectConnection reverse bind — the widgets' StopOptimizer analog:
         * onStopOptimizer just flips the worker's error flag; the run
         * completes through the normal OptimizedFrame/finished path.*/
        emit StopOptimizer();
    }
    core_.requestStop();
    emit runStateChanged();
}

void OptimizerRunController::applySeedPose(
    LocationStorage* storage,
    int current_frame,
    int primary_model_index,
    int model_count) {
    if (!storage) {
        return;
    }
    const auto seed =
        core_.takeSeedForRun(current_frame, primary_model_index, model_count);
    if (seed.applied) {
        storage->SavePose(seed.frame, seed.model, seed.pose);
        emit seedApplied(seed.frame, seed.model);
    }
}

/*---- Lifecycle ----*/

OptimizerRunController::OptimizerRunController(
    DriverFactory factory,
    QObject* parent) :
    QObject(parent), factory_(std::move(factory)) {}

/*---- Destructor contract (H3) ----*/

OptimizerRunController::~OptimizerRunController() {
    /*Request the cooperative stop (prompt for bounded DIRECT iterations),
     * then quit + bounded wait — the adapter warns and keeps waiting on
     * expiry; a running thread is NEVER deleted.*/
    if (driver_) {
        driver_->Stop();
        driver_->Wait();
    }
}

/*---- Relays (the 8 binds, re-emitted on the controller thread) ----*/

bool OptimizerRunController::isCurrentRun(QObject* sender) const {
    return driver_ && run_epoch_ == core_.epoch() &&
        sender == driver_->Manager() && core_.state() != RunState::Idle;
}

void OptimizerRunController::onManagerUpdateDisplay(
    double iteration_speed,
    int current_iteration,
    double current_minimum,
    unsigned int primary_model_index) {
    if (!isCurrentRun(sender())) {
        return;  // stale-epoch relay (H1)
    }
    core_.refreshProgress(budgets_, current_iteration, current_minimum);
    emit progressChanged();
    emit updateDisplayRelayed(
        iteration_speed,
        current_iteration,
        current_minimum,
        primary_model_index);
}

void OptimizerRunController::onManagerOptimizerError(
    const QString& error_message) {
    if (!isCurrentRun(sender())) {
        return;  // stale-epoch relay (H1)
    }
    core_.onOptimizerError();
    emit runStateChanged();
    emit messageRequested(
        QStringLiteral("Error!"), error_message, Severity::Critical);
}

void OptimizerRunController::onManagerUpdateOptimum(
    double x,
    double y,
    double z,
    double xa,
    double ya,
    double za,
    unsigned int primary_model_index) {
    if (!isCurrentRun(sender())) {
        return;  // stale-epoch relay (H1)
    }
    /*Live-pose relay (R16): the views map onto their own scenes/actors; the
     * display conversion stays view-side (L15).*/
    emit poseUpdated(x, y, z, xa, ya, za, primary_model_index);
}

void OptimizerRunController::onManagerOptimizedFrame(
    double x,
    double y,
    double z,
    double xa,
    double ya,
    double za,
    bool move_next_frame,
    unsigned int primary_model_index,
    bool error_occurred,
    QString optimizer_directive) {
    if (!isCurrentRun(sender())) {
        return;  // stale-epoch relay (H1)
    }
    /*Core (widgets onOptimizedFrame, mainscreen.cpp:4383-4502): persist the
     * result at the tracked current frame (the manager worked on a by-value
     * copy), then advance the tracked frame exactly when the widgets mapper
     * advances the view (move_next_frame + bounds). The out-of-bounds
     * status travels on the relay so the widgets view can box (L14); the
     * pose persistence is skipped on out-of-bounds (the old unconditional
     * SavePose on an out-of-range row was the documented never-occur crash
     * path).*/
    const bool model_out_of_bounds =
        static_cast<int>(primary_model_index) >= model_count_;
    if (storage_ && !model_out_of_bounds) {
        storage_->SavePose(
            current_frame_,
            static_cast<int>(primary_model_index),
            Point6D(x, y, z, xa, ya, za));
    }
    const bool advance = move_next_frame &&
        (optimizer_directive == QStringLiteral("Backward")
             ? current_frame_ > 0
             : current_frame_ + 1 < frame_count_);
    if (advance) {
        current_frame_ +=
            optimizer_directive == QStringLiteral("Backward") ? -1 : 1;
    }
    /*Terminal state: Completed unless an OptimizerError already moved the
     * run to Error (pinned QML semantics); the widgets mapper unlocks on
     * this relay regardless of the error bit (M8).*/
    core_.onTerminalFrame(error_occurred);
    emit runStateChanged();
    emit optimizedFrameRelayed(
        x,
        y,
        z,
        xa,
        ya,
        za,
        move_next_frame,
        primary_model_index,
        error_occurred,
        optimizer_directive,
        model_out_of_bounds);
}

void OptimizerRunController::onManagerUpdateDilationBackground() {
    if (!isCurrentRun(sender())) {
        return;  // stale-epoch relay (H1)
    }
    emit dilationBackgroundRequested();
}

void OptimizerRunController::onManagerOrientationSymTrap(
    double x,
    double y,
    double z,
    double xa,
    double ya,
    double za) {
    if (!isCurrentRun(sender())) {
        return;  // stale-epoch relay (H1)
    }
    emit orientationSymTrapUpdated(x, y, z, xa, ya, za);
}

void OptimizerRunController::onManagerFinished() {
    if (!isCurrentRun(sender())) {
        return;  // stale-epoch relay (H1)
    }
    /*The manager's internal chain (wired in Initialize) already quit the
     * thread and deleteLater'd both objects; wait it out so a re-run never
     * races the old thread. The driver is released at the next start()
     * (its connections die with it then); until then the epoch + sender
     * guard drops any straggler relays from the finished run (H1 — the
     * ghost's pre-Initialize finished bind observes the same termination,
     * M6).*/
    if (driver_) {
        driver_->Wait();
    }
}

/*---- Private helpers ----*/

void OptimizerRunController::bindManager() {
    QObject* manager = driver_->Manager();
    /*finished FIRST (M6): a failed Initialize's ghost thread termination is
     * observed before Initialize's internal failure path can race a re-run.*/
    connect(manager, SIGNAL(finished()), this, SLOT(onManagerFinished()));
    /*The 7 binds (widgets' order, mainscreen.cpp:4230-4288): 6 relays + the
     * app->manager StopOptimizer reverse bind (DirectConnection). String
     * connects: the manager QObject is polymorphic through the driver seam
     * (the fake driver declares the same signals/slot).*/
    connect(
        manager,
        SIGNAL(UpdateDisplay(double, int, double, unsigned int)),
        this,
        SLOT(onManagerUpdateDisplay(double, int, double, unsigned int)));
    connect(
        manager,
        SIGNAL(OptimizerError(QString)),
        this,
        SLOT(onManagerOptimizerError(QString)));
    connect(
        manager,
        SIGNAL(UpdateOptimum(
            double, double, double, double, double, double, unsigned int)),
        this,
        SLOT(onManagerUpdateOptimum(
            double, double, double, double, double, double, unsigned int)));
    connect(
        manager,
        SIGNAL(OptimizedFrame(
            double,
            double,
            double,
            double,
            double,
            double,
            bool,
            unsigned int,
            bool,
            QString)),
        this,
        SLOT(onManagerOptimizedFrame(
            double,
            double,
            double,
            double,
            double,
            double,
            bool,
            unsigned int,
            bool,
            QString)));
    connect(
        this,
        SIGNAL(StopOptimizer()),
        manager,
        SLOT(onStopOptimizer()),
        Qt::DirectConnection);
    connect(
        manager,
        SIGNAL(UpdateDilationBackground()),
        this,
        SLOT(onManagerUpdateDilationBackground()));
    connect(
        manager,
        SIGNAL(onUpdateOrientationSymTrap(
            double, double, double, double, double, double)),
        this,
        SLOT(onManagerOrientationSymTrap(
            double, double, double, double, double, double)));
}

jta::OptimizerRunControllerCore::ProgressBudgets
OptimizerRunController::BudgetsFromSettings(const OptimizerSettings& settings) {
    jta::OptimizerRunControllerCore::ProgressBudgets b;
    b.trunk_budget = settings.trunk_budget;
    b.branch_budget = settings.branch_budget;
    b.number_branches = settings.number_branches;
    b.enable_branch = settings.enable_branch_;
    b.leaf_budget = settings.leaf_budget;
    b.enable_leaf = settings.enable_leaf_;
    return b;
}

QString OptimizerRunController::DirectiveToString(Directive directive) {
    switch (directive) {
    case Directive::All:
        return QStringLiteral("All");
    case Directive::Each:
        return QStringLiteral("Each");
    case Directive::From:
        return QStringLiteral("From");
    case Directive::Backward:
        return QStringLiteral("Backward");
    case Directive::SymTrap:
        return QStringLiteral("Sym_Trap");
    case Directive::Single:
        break;
    }
    return QStringLiteral("Single");
}
