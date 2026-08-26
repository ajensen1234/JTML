// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// Plan 006 U5: OptimizerRunController — the shared optimizer-run controller
// (R7, R13, R15, R16; F1; AE1). Both front-ends drive it: MainScreen's
// LaunchOptimizer becomes `start(directive)` + view-side mappers, and
// OptimizerBridge thins onto it (Q_PROPERTYs + the SingleModelOnly policy +
// Dialog mapping).
//
// Drive sequence (pinned, mainscreen.cpp:4135 / OptimizerBridge.cpp:114):
// SaveLastPose mirror (U3 core) -> intent gate (previous == current, H2) ->
// seed applied (M10a, snapshot kept) -> fresh driver (M12/Q8) -> finished
// bound BEFORE Initialize (M6) -> the 7 binds (L13) -> Initialize by value
// -> quirk on failure (thread started before the box; ghost reaped via the
// pre-Initialize finished bind) -> thread start.
//
// Run-state machine + epoch + progress + seed state live in the Qt/GPU-free
// OptimizerRunControllerCore; this shell owns the driver seam, the thread
// lifecycle + destructor contract (H3), and the by-value relay signals
// re-emitted on the controller thread (QTBUG-2842 — QSignalSpy must observe
// controller signals, never worker-thread emissions).
//
// Start gate (H1/M6): Idle/Completed/Error-with-no-thread AND !threadActive;
// stale-epoch relays (sender + epoch guard) are dropped.
//
// Signals: messageRequested(title, message, severity) is the single error
// channel (L14 — widgets boxes by severity; QML ignores it). The
// optimizedFrameRelayed relay carries the out-of-bounds status so the
// widgets view can box (L14); the widgets unlocks (EnableAll) on that relay
// even with the error bit set (pinned unlock-after-error, M8).

#ifndef OPTIMIZER_RUN_CONTROLLER_H
#define OPTIMIZER_RUN_CONTROLLER_H

#include <QObject>
#include <QString>
#include <functional>
#include <memory>

#include "coordinator/optimizer_run_controller_core.h"
#include "coordinator/optimizer_run_driver.h"
#include "services/save_last_pose.h"

class LocationStorage;

/*Everything the controller's start() consumes: the save-last-pose mirror
 * args (pinned before the gate), the gate input, the live storage (terminal
 * SavePose + seed apply/restore), and the Initialize payload (forwarded by
 * value through the driver). The view captures the save-last-pose + gate
 * pieces BEFORE its directive reset (M11) so the launch sees the user's
 * pre-reset mirrors.*/
struct OptimizerRunRequest {
    jta::OptimizerRunControllerCore::Directive directive =
        jta::OptimizerRunControllerCore::Directive::Single;

    /*SaveLastPose mirror (before the gate): previous selection at the
     * previous frame, pose-source functor (view-side reads), convert rule.*/
    int save_frame = -1;
    std::vector<int> save_rows;
    std::function<Point6D(int)> save_pose_source;
    bool camera_is_a = true;
    jta::SavePoseConvertRule save_convert_rule =
        jta::SavePoseConvertRule::NeverConvert;

    /*Gate input (plain values mirroring LaunchOptimizer's assembly).*/
    std::vector<int> selected_model_rows;
    int current_frame = -1;
    int frame_count = 0;
    int model_current_index = -1;
    int model_count = 0;
    int pose_frame_count = 0;
    int pose_model_count = 0;

    /*Live storage: terminal-frame SavePose + seed apply/restore. The
     * manager's Initialize gets its own by-value copy (launch.pose_matrix).*/
    LocationStorage* storage = nullptr;

    /*Initialize payload (containers + plain rows by value).*/
    jta::OptimizerRunLaunch launch;
    int iter_count = 0;
};

class OptimizerRunController : public QObject {
    Q_OBJECT

public:
    using RunState = jta::OptimizerRunControllerCore::RunState;
    using Severity = jta::OptimizerRunControllerCore::Severity;
    using Directive = jta::OptimizerRunControllerCore::Directive;
    /*The per-run driver factory (default: the production adapter). Tests
     * inject a fake driver factory. Shared ownership: a finished run's
     * driver is released at the next start(); its connections stay alive
     * until then so the epoch + sender guard can drop straggler relays
     * (H1).*/
    using DriverFactory =
        std::function<std::shared_ptr<jta::OptimizerRunDriver>()>;

    explicit OptimizerRunController(
        DriverFactory factory = jta::CreateOptimizerManagerRunDriver,
        QObject* parent = nullptr);
    ~OptimizerRunController() override;

    /*---- Run control ------------------------------------------------------*/
    /*The full drive sequence (save mirror -> gate -> seed -> driver ->
     * binds -> Initialize -> thread start). Returns false on gate rejection
     * or Initialize failure (both surface through messageRequested; the
     * Initialize-failure quirk + seed restore are applied inside; state
     * unchanged on gate rejection, seed NOT consumed). Rejected while a run
     * is in flight or a previous thread is still alive (H1/M6).*/
    bool start(const OptimizerRunRequest& req);
    /*Emergency stop (widgets action / QML button): emits StopOptimizer (the
     * per-run DirectConnection reverse bind) + Stopping; the run completes
     * through the normal terminal-frame/finished path. No-op outside a
     * run.*/
    void stop();

    /*---- State + progress reads (QML surface) -----------------------------*/
    RunState runState() const {
        return core_.state();
    }
    bool running() const {
        return core_.running();
    }
    bool canRun() const {
        return core_.canStart();
    }
    QString stageText() const {
        return QString::fromStdString(core_.stageText());
    }
    int costCalls() const {
        return core_.costCalls();
    }
    double currentMinimum() const {
        return core_.currentMinimum();
    }
    double progress() const {
        return core_.progress();
    }

    /*---- Seed lifecycle (M10a) --------------------------------------------*/
    /*One-shot ML-estimate starting-pose seed, estimated for (frame, model)
     * (the widgets equivalent: the estimate slots' direct SavePose into the
     * storage). start() applies it AFTER the gate (a rejected run never
     * consumes it) and before Initialize; on Initialize failure the storage
     * snapshot is restored and seedRestored is emitted.*/
    void setSeedPose(
        double x,
        double y,
        double z,
        double xa,
        double ya,
        double za,
        int frame,
        int model) {
        core_.setSeedPose(x, y, z, xa, ya, za, frame, model);
    }
    void clearSeedPose() {
        core_.clearSeedPose();
    }
    bool hasSeedPose() const {
        return core_.hasSeedPose();
    }
    /*Apply the pending seed to `storage` outside a run (the QML bridge's
     * headless-testable applySeedPose delegate): one-shot + stale guards;
     * emits seedApplied(frame, model) when applied (the view maps the scene
     * write).*/
    void applySeedPose(
        LocationStorage* storage,
        int current_frame,
        int primary_model_index,
        int model_count);

signals:
    void runStateChanged();
    void progressChanged();
    /*The single severity-carrying message channel (L14).*/
    void messageRequested(
        const QString& title,
        const QString& message,
        Severity severity);
    /*Relays (by-value, re-emitted on the controller thread — QTBUG-2842).*/
    void updateDisplayRelayed(
        double iteration_speed,
        int current_iteration,
        double current_minimum,
        unsigned int primary_model_index);
    void poseUpdated(
        double x,
        double y,
        double z,
        double xa,
        double ya,
        double za,
        unsigned int primary_model_index);
    /*Terminal frame relay: raw A-coord pose + advance decision + error bit
     * + directive + the out-of-bounds status (L14 — the widgets view boxes
     * and unlocks on this relay even with the error bit set, M8). The
     * controller has already persisted the pose at its tracked current
     * frame and advanced the tracked frame exactly when the view advances.*/
    void optimizedFrameRelayed(
        double x,
        double y,
        double z,
        double xa,
        double ya,
        double za,
        bool move_next_frame,
        unsigned int primary_model_index,
        bool error_occurred,
        const QString& optimizer_directive,
        bool model_out_of_bounds);
    void dilationBackgroundRequested();
    void orientationSymTrapUpdated(
        double x,
        double y,
        double z,
        double xa,
        double ya,
        double za);
    /*The pending seed was applied to the storage (view maps its scene).*/
    void seedApplied(int frame, int model);
    /*An Initialize failure restored the pre-seed storage snapshot (M10a —
     * the estimate is not silently kept; the view re-syncs its scene).*/
    void seedRestored(int frame, int model);
    /*App -> manager reverse bind (connected with Qt::DirectConnection to the
     * driver manager's onStopOptimizer slot in start()); stop() emits it.
     * Not meant for views.*/
    void StopOptimizer();

private slots:
    void onManagerUpdateDisplay(
        double iteration_speed,
        int current_iteration,
        double current_minimum,
        unsigned int primary_model_index);
    void onManagerOptimizerError(const QString& error_message);
    void onManagerUpdateOptimum(
        double x,
        double y,
        double z,
        double xa,
        double ya,
        double za,
        unsigned int primary_model_index);
    void onManagerOptimizedFrame(
        double x,
        double y,
        double z,
        double xa,
        double ya,
        double za,
        bool move_next_frame,
        unsigned int primary_model_index,
        bool error_occurred,
        QString optimizer_directive);
    void onManagerUpdateDilationBackground();
    void onManagerOrientationSymTrap(
        double x,
        double y,
        double z,
        double xa,
        double ya,
        double za);
    void onManagerFinished();

private:
    /*Epoch + sender guard: drops relays from stale runs (H1).*/
    bool isCurrentRun(QObject* sender) const;
    /*The 8 binds: finished FIRST (M6), then the 7 (widgets' order, L13).*/
    void bindManager();
    static jta::OptimizerRunControllerCore::ProgressBudgets BudgetsFromSettings(
        const OptimizerSettings& settings);
    static QString DirectiveToString(Directive directive);

    DriverFactory factory_;
    std::shared_ptr<jta::OptimizerRunDriver> driver_;
    jta::OptimizerRunControllerCore core_;
    int run_epoch_ = -1;
    /*Tracked run frame (terminal-frame SavePose + advance mirror of the
     * view's selection advance).*/
    int current_frame_ = -1;
    int frame_count_ = 0;
    int model_count_ = 0;
    LocationStorage* storage_ = nullptr;
    jta::OptimizerRunControllerCore::ProgressBudgets budgets_;
};

#endif /* OPTIMIZER_RUN_CONTROLLER_H */
