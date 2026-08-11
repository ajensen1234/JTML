// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U2: AppBridge — the QML-exposed hub (session / settings / pose
// surfaces) that will own the four thin per-seam adapters: StudyBridge (U4),
// OptimizerBridge (U6), MlBridge (U7), PoseBridge (U8).
//
// Thinness rule (plan 005, bridge decomposition review fix): the bridge and
// its adapters are pass-throughs — no logic; all behavior lives in the seams
// they delegate to (SessionController, OptimizerSettings, CostFunctionManager,
// OptimizerManager, ...). This class is the surface skeleton for U2: dataset
// counts (the study surface's headline numbers, driven by U4's StudyBridge)
// + placeholder signals for the surfaces QML will bind in later units.
//
// The app-owned dataset containers (frames/models/LocationStorage, R3) are
// owned by the composition root (main.cpp) and handed to the adapters when
// they land; AppBridge itself stays thin.

#pragma once

#include <QObject>

class AppBridge : public QObject {
    Q_OBJECT

    // Study surface: dataset counts (0 until U4's StudyBridge populates the
    // app-owned containers through SessionController).
    Q_PROPERTY(int frameCount READ frameCount NOTIFY sessionChanged)
    Q_PROPERTY(int modelCount READ modelCount NOTIFY sessionChanged)

public:
    explicit AppBridge(QObject* parent = nullptr);

    int frameCount() const;
    int modelCount() const;

    // Called by U4's StudyBridge after SessionController parse/populate
    // (mirror of the widgets SyncSessionState tail).
    void setFrameCount(int count);
    void setModelCount(int count);

signals:
    // Placeholder surface-change signals (U2). Later units refine these into
    // the per-surface signals QML binds to (settings panel, pose table,
    // optimizer progress).
    void sessionChanged();
    void settingsChanged();
    void poseTableChanged();

private:
    int frame_count_ = 0;
    int model_count_ = 0;
};
