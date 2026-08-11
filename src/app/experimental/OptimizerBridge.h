// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U6: OptimizerBridge — the thin optimizer-run adapter (R6, R7, R13).
// Drives the REAL OptimizerManager exactly like MainScreen::LaunchOptimizer
// (mainscreen.cpp:4135): SaveLastPose mirror -> OptimizeIntentController::
// Evaluate gate (+ the v1 single-model-mode rule) -> new OptimizerManager +
// new QThread + moveToThread -> Initialize(...) with the app's containers BY
// VALUE and a QModelIndexList built from the direct-compiled model list
// (model->index(row, 0) — no QItemSelectionModel anywhere) -> the 7 signal
// binds relayed to QML -> thread start.
//
// Thread lifecycle (mirror of the widgets flow): a fresh manager + thread
// per run. The manager's own Initialize() wires started->Optimize and
// finished->quit/deleteLater chains internally; onFinished() waits the
// thread out so a re-run never races the old one. The R13-preserved
// Initialize-FAILURE quirk (thread started before the error box/return,
// manager + thread never cleaned up) is replicated deliberately with a
// comment — the fix is the deferred follow-up cut (plan "Deferred to
// Follow-Up Work").
//
// Run-state machine (plan 005 U6 review fix): idle -> running -> stopping ->
// completed/error, exposed to QML (runState + running/canRun helpers). Run
// is enabled in idle/completed/error; Stop while running/stopping; a re-run
// guard rejects a second manager mid-run. While running, QML locks every
// other control (mirror of the widgets DisableAll — load, lists, settings,
// pose surfaces disabled; stop always enabled). Errors (intent-gate
// rejection, Initialize failure, OptimizerError) all surface through the
// single messageRequested -> QML Dialog mechanism.
//
// v1 run scope: the current frame only (directive "Single"; move_next_frame
// is false by construction — no auto-advance loop; All/Each/From/Backward
// deferred per the plan's review fix).
//
// Testability split (plan 005 U6 test scenario e): the gate
// (EvaluateGate/buildGateInput) and the state + persistence core
// (applyOptimizedFrame) are public and Qt/GPU-free, so the intent/state
// parts are headless-testable; only the manager launch itself needs the GPU
// (manual-visual under xcb, later U9's parity run).

#pragma once

#include <QModelIndexList>
#include <QObject>
#include <QPointer>

#include "domain/data_structures_6D.h"
#include "domain/optimize_intent_controller.h"

class AppBridge;
class ExperimentalScene;
class ExperimentalSession;
class OptimizerManager;
class QThread;
class SettingsBridge;
class StudyBridge;

class OptimizerBridge : public QObject {
    Q_OBJECT

public:
    // Run-state machine (plan 005 U6 review fix): idle -> running ->
    // stopping -> completed/error. QML compares runState against
    // OptimizerBridge.Completed etc. (type registered in main.cpp for the
    // enum surface) or binds the running/canRun helpers.
    enum class RunState { Idle = 0, Running = 1, Stopping = 2,
                          Completed = 3, Error = 4 };
    Q_ENUM(RunState)

    Q_PROPERTY(RunState runState READ runState NOTIFY runStateChanged)
    // True while a run is in flight (running or stopping): QML locks every
    // other control (DisableAll mirror) and enables Stop.
    Q_PROPERTY(bool running READ running NOTIFY runStateChanged)
    // True in idle/completed/error: the Run button's enabled binding.
    Q_PROPERTY(bool canRun READ canRun NOTIFY runStateChanged)

    // Progress surface (driven by the UpdateDisplay bind): stage
    // (Trunk / Branch n / Extra Z-Translation / Finished — the widgets
    // onUpdateDisplay level math), cumulative cost-function calls, current
    // minimum, and a 0..1 progress from calls vs the cumulative budget.
    Q_PROPERTY(QString stageText READ stageText NOTIFY progressChanged)
    Q_PROPERTY(int costCalls READ costCalls NOTIFY progressChanged)
    Q_PROPERTY(double currentMinimum READ currentMinimum NOTIFY
                   progressChanged)
    Q_PROPERTY(double progress READ progress NOTIFY progressChanged)

    // The gate's failure taxonomy: the two OptimizeIntentController statuses
    // (mirroring LaunchOptimizer's two guards) + the v1 single-model-mode
    // rule (multi-select rejected — pose ops are primary-model-only in v1).
    enum class GateStatus {
        Ok = 0,
        SelectFrameAndModel,
        SingleModelOnly,
        PoseMatrixDimensionMismatch,
    };

    // Evaluate the entry gate on plain values (headless-testable; the
    // controller's Input mirrors everything LaunchOptimizer reads at the
    // gate). Multi-select is rejected AFTER the controller's first guard, so
    // an empty/invalid selection still reports SelectFrameAndModel first.
    struct GateResult {
        GateStatus status = GateStatus::SelectFrameAndModel;
        jta::OptimizeIntentController::Intent intent;
    };
    static GateResult
    EvaluateGate(const jta::OptimizeIntentController::Input& in);

    explicit OptimizerBridge(
        AppBridge* hub,
        ExperimentalSession* session,
        ExperimentalScene* scene,
        StudyBridge* study_bridge,
        SettingsBridge* settings_bridge,
        QObject* parent = nullptr);
    ~OptimizerBridge() override;

    // ---- Run control (QML buttons) --------------------------------------
    // Entry gate -> manager + thread lifecycle -> 7 binds -> thread start.
    // Rejects (messageRequested, state unchanged) before any manager or
    // thread exists — the rejection paths are headless-safe.
    Q_INVOKABLE void run();
    // Emergency stop (app -> manager, Qt::DirectConnection — the widgets
    // StopOptimizer reverse bind; onStopOptimizer just flips a worker flag).
    // No-op unless a run is in flight.
    Q_INVOKABLE void stop();

    // ---- U7: ML-estimate starting-pose seed ------------------------------
    // The one-shot starting-pose seed the ML estimate sets before a run
    // (R8 — the estimate seeds the optimizer). run() applies it (session
    // storage + scene) AFTER the gate passes, so the estimate wins over the
    // SaveLastPose scene-drift mirror and a rejected run never consumes it.
    // The widgets equivalent is the estimate slots' SavePose into
    // model_locations_ feeding LaunchOptimizer's by-value pose matrix — the
    // manager's Optimize() reads the starting point from that matrix.
    Q_INVOKABLE void setSeedPose(double x, double y, double z, double xa,
                                 double ya, double za);
    // Drop the pending seed (MlBridge clears it on a selection change — a
    // stale-frame seed must never override a different frame's pose).
    Q_INVOKABLE void clearSeedPose();
    // Headless-testable core (plan 005 U7): applies the pending seed to the
    // seeded frame's model in the session storage + scene, then clears it.
    // Stale guards: the seed applies only when the current frame is still
    // the seeded frame and the seeded model is still the primary selection;
    // otherwise it is dropped. run() calls this after the gate passes,
    // before Initialize.
    void applySeedPose();

    // ---- State + progress reads ------------------------------------------
    RunState runState() const;
    bool running() const;
    bool canRun() const;
    QString stageText() const;
    int costCalls() const;
    double currentMinimum() const;
    double progress() const;

    // ---- Headless-testable core (plan 005 U6 test scenario e) ------------
    // The gate Input exactly as run() builds it from the app's current
    // dataset + selection state (mirrors the LaunchOptimizer Input
    // assembly, mainscreen.cpp:4152-4164).
    jta::OptimizeIntentController::Input buildGateInput() const;
    // The OptimizedFrame handling core (the private slot relays the signal
    // here): SavePose into the session's LocationStorage + scene pose
    // update + run-state -> completed (kept at error if an OptimizerError
    // already moved the run there). v1: directive Single — no auto-advance
    // loop (move_next_frame is false by construction).
    void applyOptimizedFrame(
        double x, double y, double z, double xa, double ya, double za,
        bool move_next_frame, unsigned int primary_model_index,
        bool error_occurred, const QString& optimizer_directive);
    // The UpdateDisplay handling core (the private slot relays the signal
    // here): stage (widgets onUpdateDisplay level math), cumulative calls,
    // current minimum, and calls-vs-budget progress.
    void refreshProgress(int calls, double minimum);

signals:
    void runStateChanged();
    void progressChanged();
    // Relay binds for the QML glue (main.qml forwards the pose ones to
    // viewport.updatePose — the bridge already wrote the scene pose).
    void poseUpdated(int modelIndex);          // UpdateOptimum bind
    void frameOptimized(int frameIndex, int modelIndex);  // OptimizedFrame bind
    void dilationBackgroundRequested();        // UpdateDilationBackground bind
    void orientationSymTrapUpdated(            // onUpdateOrientationSymTrap bind
        double x, double y, double z, double xa, double ya, double za);
    // The single QML Dialog mechanism (same channel as StudyBridge's):
    // intent-gate rejection, Initialize failure, OptimizerError.
    void messageRequested(const QString& title, const QString& message);
    // App -> manager reverse bind (connected with Qt::DirectConnection to
    // OptimizerManager::onStopOptimizer in run()); not meant for QML.
    void StopOptimizer();

private slots:
    void onUpdateDisplay(
        double iteration_speed, int current_iteration, double current_minimum,
        unsigned int primary_model_index);
    void onOptimizerError(const QString& error_message);
    void onUpdateOptimum(
        double x, double y, double z, double xa, double ya, double za,
        unsigned int primary_model_index);
    void onOptimizedFrame(
        double x, double y, double z, double xa, double ya, double za,
        bool move_next_frame, unsigned int primary_model_index,
        bool error_occurred, const QString& optimizer_directive);
    void onUpdateDilationBackground();
    void onOrientationSymTrap(
        double x, double y, double z, double xa, double ya, double za);
    void onFinished();

private:
    void setState(RunState state);
    void saveScenePosesForCurrentSelection();  // SaveLastPose mirror
    QModelIndexList selectedModelIndexes() const;
    AppBridge* hub_ = nullptr;
    ExperimentalSession* session_ = nullptr;
    ExperimentalScene* scene_ = nullptr;
    StudyBridge* study_bridge_ = nullptr;
    SettingsBridge* settings_bridge_ = nullptr;

    // Per-run manager + thread (self-delete after the run via the manager's
    // internal finished chain; QPointer guards the failure quirk + teardown
    // ordering).
    QPointer<OptimizerManager> manager_;
    QPointer<QThread> optimizer_thread_;

    RunState state_ = RunState::Idle;
    QString stage_text_ = QStringLiteral("Idle");
    int cost_calls_ = 0;
    double current_minimum_ = 0.0;
    double progress_ = 0.0;

    // U7 seed state: pending pose + the frame/model it was estimated for
    // (stale guards in applySeedPose).
    bool has_seed_pose_ = false;
    Point6D seed_pose_;
    int seed_frame_ = -1;
    int seed_model_ = -1;
};
