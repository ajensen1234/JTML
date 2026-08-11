// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U6 / 006 U5: OptimizerBridge — the QML-facing thin shell over the
// shared OptimizerRunController (plan 006 U5: the run-state machine, thread
// lifecycle, progress math, and save-last-pose mirror move INTO the
// controller; this bridge keeps its Q_PROPERTY surface, the SingleModelOnly
// pre-check (a bridge policy, NOT in the controller — after the shared
// gate's first guard), and the Dialog mapping (severity ignored)).
//
// QML deltas (enumerated in plan 006 U5): (1) a re-run in the ghost window
// (terminal frame delivered, old thread still finishing) is rejected by the
// controller's Start gate with a message (H1 — deliberate, tested); (2)
// after a failed Initialize the seed-restore reverts storage/scene to the
// pre-seed pose (M10a — the estimate is not silently kept).
//
// The headless-testable core (EvaluateGate/buildGateInput/applySeedPose)
// now delegates to the shared core + controller: gate Input assembly
// (previous == current, H2) + gate evaluation live in
// jta::OptimizerRunControllerCore; the state + persistence live in
// OptimizerRunController. The GPU run itself remains manual-visual under
// xcb / oracle-arbitrated.

#pragma once

#include <QModelIndexList>
#include <QObject>
#include <QVariantList>

#include "coordinator/optimizer_run_controller_core.h"

class AppBridge;
class ExperimentalScene;
class ExperimentalSession;
class OptimizerRunController;
class QThread;
class SettingsBridge;
class StudyBridge;

class OptimizerBridge : public QObject {
    Q_OBJECT

public:
    // Run-state machine (values mirror the shared core; QML compares
    // runState against OptimizerBridge.Completed etc. — the type is
    // registered in main.cpp for the enum surface).
    enum class RunState { Idle = 0, Running = 1, Stopping = 2,
                          Completed = 3, Error = 4 };
    Q_ENUM(RunState)

    Q_PROPERTY(RunState runState READ runState NOTIFY runStateChanged)
    // True while a run is in flight (running or stopping): QML locks every
    // other control (DisableAll mirror) and enables Stop.
    Q_PROPERTY(bool running READ running NOTIFY runStateChanged)
    // True in idle/completed/error: the Run button's enabled binding.
    Q_PROPERTY(bool canRun READ canRun NOTIFY runStateChanged)

    // Progress surface (driven by the controller's UpdateDisplay bind):
    // stage (Trunk / Branch n / Extra Z-Translation / Finished), cumulative
    // cost-function calls, current minimum, and calls-vs-budget progress.
    Q_PROPERTY(QString stageText READ stageText NOTIFY progressChanged)
    Q_PROPERTY(int costCalls READ costCalls NOTIFY progressChanged)
    Q_PROPERTY(double currentMinimum READ currentMinimum NOTIFY
                   progressChanged)
    Q_PROPERTY(double progress READ progress NOTIFY progressChanged)

    // The gate's failure taxonomy: the two OptimizeIntentController statuses
    // + the v1 single-model-mode rule (multi-select rejected — pose ops are
    // primary-model-only in v1).
    enum class GateStatus {
        Ok = 0,
        SelectFrameAndModel,
        SingleModelOnly,
        PoseMatrixDimensionMismatch,
    };

    // Evaluate the entry gate on plain values. Multi-select is rejected
    // AFTER the shared gate's first guard, so an empty/invalid selection
    // still reports SelectFrameAndModel first.
    struct GateResult {
        GateStatus status = GateStatus::SelectFrameAndModel;
        jta::OptimizeIntentController::Intent intent;
    };
    static GateResult
    EvaluateGate(const jta::OptimizerRunControllerCore::GateInput& in);

    explicit OptimizerBridge(
        AppBridge* hub,
        ExperimentalSession* session,
        ExperimentalScene* scene,
        StudyBridge* study_bridge,
        SettingsBridge* settings_bridge,
        QObject* parent = nullptr);
    ~OptimizerBridge() override;

    // ---- Run control (QML buttons) --------------------------------------
    // SingleModelOnly pre-check -> the shared controller's start() (gate +
    // SaveLastPose mirror + seed + thread lifecycle + the 8 binds). The
    // rejection paths are headless-safe (no driver is created).
    Q_INVOKABLE void run();
    // Emergency stop (delegates to the controller — the app -> manager
    // reverse bind; onStopOptimizer just flips a worker flag). No-op unless
    // a run is in flight.
    Q_INVOKABLE void stop();

    // ---- U7: ML-estimate starting-pose seed ------------------------------
    // The one-shot starting-pose seed the ML estimate sets before a run
    // (R8 — the estimate seeds the optimizer). The controller applies it
    // AFTER the gate passes, so the estimate wins over the SaveLastPose
    // mirror and a rejected run never consumes it.
    Q_INVOKABLE void setSeedPose(double x, double y, double z, double xa,
                                 double ya, double za);
    // Drop the pending seed (MlBridge clears it on a selection change — a
    // stale-frame seed must never override a different frame's pose).
    Q_INVOKABLE void clearSeedPose();
    // Headless-testable core delegate: applies the pending seed to the
    // session storage + scene (one-shot; stale guards live in the shared
    // core). run() applies it via the controller after the gate passes.
    void applySeedPose();

    // ---- State + progress reads (derived from the controller) -----------
    RunState runState() const;
    bool running() const;
    bool canRun() const;
    QString stageText() const;
    int costCalls() const;
    double currentMinimum() const;
    double progress() const;

    // ---- Headless-testable core delegates --------------------------------
    // The gate Input exactly as run() builds it from the app's current
    // dataset + selection state (previous == current by construction, H2).
    jta::OptimizerRunControllerCore::GateInput buildGateInput() const;

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
    // intent-gate rejection, Initialize failure, OptimizerError. The
    // controller's severity is ignored (Dialog mapping).
    void messageRequested(const QString& title, const QString& message);

private slots:
    void onControllerRunStateChanged();
    void onControllerProgressChanged();
    void onControllerMessage(
        const QString& title, const QString& message,
        jta::OptimizerRunControllerCore::Severity severity);
    void onControllerPoseUpdated(
        double x, double y, double z, double xa, double ya, double za,
        unsigned int primary_model_index);
    void onControllerOptimizedFrame(
        double x, double y, double z, double xa, double ya, double za,
        bool move_next_frame, unsigned int primary_model_index,
        bool error_occurred, const QString& optimizer_directive,
        bool model_out_of_bounds);
    void onControllerDilationBackground();
    void onControllerOrientationSymTrap(
        double x, double y, double z, double xa, double ya, double za);
    void onControllerSeedApplied(int frame, int model);
    void onControllerSeedRestored(int frame, int model);

private:
    QModelIndexList selectedModelIndexes() const;
    OptimizerRunController* controller_ = nullptr;
    AppBridge* hub_ = nullptr;
    ExperimentalSession* session_ = nullptr;
    ExperimentalScene* scene_ = nullptr;
    StudyBridge* study_bridge_ = nullptr;
    SettingsBridge* settings_bridge_ = nullptr;
};
