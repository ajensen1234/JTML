// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U2/U4: AppBridge — the QML-exposed hub (session / settings / pose
// surfaces) that owns the thin per-seam adapters: StudyBridge (U4),
// SettingsBridge (U5), OptimizerBridge (U6), MlBridge (U7), PoseBridge (U8).
//
// Thinness rule (plan 005, bridge decomposition review fix): the bridge and
// its adapters are pass-throughs — no logic; all behavior lives in the seams
// they delegate to (SessionController, OptimizerSettings, CostFunctionManager,
// OptimizerManager, ...). U4: the hub owns the app-owned dataset
// (ExperimentalSession — frames/models/LocationStorage/calibration, R3) and
// creates StudyBridge; the hub's headline counts (frameCount/modelCount) are
// refreshed by StudyBridge after each load/populate (the widgets
// SyncSessionState() tail) and drive the QML "Frames (n) / Models (n)"
// labels. U5: the hub also owns the settings surface — a SettingsService
// (real registry by default; tests inject an ini-backed one) + SettingsBridge
// (the session-local editor state); the bridge loads the persisted settings
// at startup (U5 load-on-startup; U8 wires the full startup flow).

#pragma once

#include <QObject>

#include "OptimizerBridge.h" // Q_PROPERTY pointer type must be complete for moc
#include "SettingsBridge.h" // Q_PROPERTY pointer type must be complete for moc

class ExperimentalScene;
class StudyBridge;
struct ExperimentalSession;
namespace jta {
class SettingsService;
}

class AppBridge : public QObject {
    Q_OBJECT

    // Study surface: dataset counts (refreshed by U4's StudyBridge after
    // SessionController parse/populate).
    Q_PROPERTY(int frameCount READ frameCount NOTIFY sessionChanged)
    Q_PROPERTY(int modelCount READ modelCount NOTIFY sessionChanged)

    // Settings surface (U5): the session-local settings adapter (panel
    // form + explicit save/load/reset).
    Q_PROPERTY(SettingsBridge* settingsBridge READ settingsBridge CONSTANT)

    // U6: the optimizer-run adapter (entry gate + thread lifecycle + the 7
    // signal binds + run-state machine; drives the real OptimizerManager).
    Q_PROPERTY(OptimizerBridge* optimizerBridge READ optimizerBridge CONSTANT)

public:
    // The app-owned scene (R7/R11) is bound by the composition root and
    // handed in — the hub owns the dataset + adapters around it. An
    // injected SettingsService (tests: ini-backed, never the real registry)
    // is not owned; a null service makes the hub create + own the default
    // real-registry instance.
    explicit AppBridge(
        ExperimentalScene* scene,
        jta::SettingsService* settings_service = nullptr,
        QObject* parent = nullptr);
    ~AppBridge() override;

    int frameCount() const;
    int modelCount() const;

    // Called by U4's StudyBridge after SessionController parse/populate
    // (mirror of the widgets SyncSessionState tail).
    void setFrameCount(int count);
    void setModelCount(int count);

    // U4: the hub's owned dataset + the study-load adapter (the later
    // adapters U6/U7/U8 read the dataset through these).
    ExperimentalSession* session();
    StudyBridge* studyBridge();

    // U5: the settings adapter (+ the registry service it saves through).
    SettingsBridge* settingsBridge();

    // U6: the optimizer-run adapter.
    OptimizerBridge* optimizerBridge();

signals:
    // Placeholder surface-change signals (U2). Later units refine these into
    // the per-surface signals QML binds to (settings panel, pose table,
    // optimizer progress).
    void sessionChanged();
    void settingsChanged();
    void poseTableChanged();

private:
    ExperimentalSession* session_ = nullptr;
    StudyBridge* study_bridge_ = nullptr;
    jta::SettingsService* settings_service_ = nullptr;
    bool owns_settings_service_ = false;
    SettingsBridge* settings_bridge_ = nullptr;
    OptimizerBridge* optimizer_bridge_ = nullptr;
    int frame_count_ = 0;
    int model_count_ = 0;
};
