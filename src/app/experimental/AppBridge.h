// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U2/U4: AppBridge — the QML-exposed hub (session / settings / pose
// surfaces) that owns the thin per-seam adapters: StudyBridge (U4),
// OptimizerBridge (U6), MlBridge (U7), PoseBridge (U8).
//
// Thinness rule (plan 005, bridge decomposition review fix): the bridge and
// its adapters are pass-throughs — no logic; all behavior lives in the seams
// they delegate to (SessionController, OptimizerSettings, CostFunctionManager,
// OptimizerManager, ...). U4: the hub owns the app-owned dataset
// (ExperimentalSession — frames/models/LocationStorage/calibration, R3) and
// creates StudyBridge; the hub's headline counts (frameCount/modelCount) are
// refreshed by StudyBridge after each load/populate (the widgets
// SyncSessionState() tail) and drive the QML "Frames (n) / Models (n)"
// labels.

#pragma once

#include <QObject>

class ExperimentalScene;
class StudyBridge;
struct ExperimentalSession;

class AppBridge : public QObject {
    Q_OBJECT

    // Study surface: dataset counts (refreshed by U4's StudyBridge after
    // SessionController parse/populate).
    Q_PROPERTY(int frameCount READ frameCount NOTIFY sessionChanged)
    Q_PROPERTY(int modelCount READ modelCount NOTIFY sessionChanged)

public:
    // The app-owned scene (R7/R11) is bound by the composition root and
    // handed in — the hub owns the dataset + adapters around it.
    explicit AppBridge(ExperimentalScene* scene, QObject* parent = nullptr);
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
    int frame_count_ = 0;
    int model_count_ = 0;
};
