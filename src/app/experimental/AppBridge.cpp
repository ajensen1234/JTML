// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "AppBridge.h"

#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "MlBridge.h"
#include "OptimizerBridge.h"
#include "PoseBridge.h"
#include "SettingsBridge.h"
#include "StudyBridge.h"
#include "coordinator/session_state_controller.h"
#include "services/settings_service.h"

AppBridge::AppBridge(
    ExperimentalScene* scene,
    jta::SettingsService* settings_service,
    QObject* parent) : QObject(parent) {
    /*U4: the hub owns the app-owned dataset (R3) + the study-load adapter.
     * The list models are created inside StudyBridge (direct-compiled).*/
    session_ = new ExperimentalSession;

    /*Plan 006 U6: the shared session-state controller — wraps the
     * dataset's SessionState (the hub owns the state; the controller is
     * the diff + notification layer). The lambdas capture `this` and are
     * invoked only after construction (by UpdateSession/runInFlight), so
     * the not-yet-created optimizer bridge is guarded with null checks.*/
    session_state_controller_ = new SessionStateController(
        &session_->session_state,
        [this] {
            return optimizer_bridge_ != nullptr &&
                   optimizer_bridge_->running();
        },
        [this] {
            if (optimizer_bridge_ != nullptr) {
                optimizer_bridge_->clearSeedPose();
            }
        },
        this);
    study_bridge_ =
        new StudyBridge(this, session_, scene, session_state_controller_, this);

    /*U5: the settings surface — the registry service (default: the real
     * registry; tests inject an ini-backed one) + the session-local
     * settings adapter. The persisted settings are loaded at startup
     * (first run = defaults; the registry is never written here — Save is
     * explicit).*/
    owns_settings_service_ = (settings_service == nullptr);
    settings_service_ =
        owns_settings_service_ ? new jta::SettingsService : settings_service;
    settings_bridge_ = new SettingsBridge(settings_service_, this);
    settings_bridge_->load();

    /*U6: the optimizer-run adapter — the thin pass-through that drives the
     * real OptimizerManager (entry gate + thread lifecycle + the 7 signal
     * binds + run-state machine), reading the app dataset + selection
     * (StudyBridge) + the settings surface (SettingsBridge).*/
    optimizer_bridge_ = new OptimizerBridge(
        this, session_, scene, study_bridge_, settings_bridge_, this);

    /*U7: the ML adapter — per-implant .pt pickers (femur/tibia segment + one
     * estimate model) with env fallback, per-frame segment/estimate on the
     * current frame (v1 loop scope), the estimate seeding the optimizer
     * (OptimizerBridge::setSeedPose), and graceful degradation without .pt
     * models (AE4).*/
    ml_bridge_ = new MlBridge(
        this, session_, scene, study_bridge_, settings_bridge_,
        optimizer_bridge_, this);

    /*U8: the pose-editing adapter — the pose table over the primary model's
     * LocationStorage poses (PoseTableModel), per-cell SavePose commits,
     * copy-prev/next via pose_copy, pose + kinematics file save/load via
     * pose_file_io (false returns surface + keep in-memory state), the
     * dirty + inline-validation surface. Reads the dataset + selection
     * through the session/study bridge and refreshes the table on their
     * signals.*/
    pose_bridge_ = new PoseBridge(this, session_, scene, study_bridge_, this);
}

AppBridge::~AppBridge() {
    /*study_bridge_ and settings_bridge_ are parented to this and die with
     * the QObject chain; the dataset is a plain struct.*/
    delete session_;
    if (owns_settings_service_) {
        delete settings_service_;
    }
}

int AppBridge::frameCount() const {
    return frame_count_;
}

int AppBridge::modelCount() const {
    return model_count_;
}

void AppBridge::setFrameCount(int count) {
    if (frame_count_ == count) {
        return;
    }
    frame_count_ = count;
    emit sessionChanged();
}

void AppBridge::setModelCount(int count) {
    if (model_count_ == count) {
        return;
    }
    model_count_ = count;
    emit sessionChanged();
}

ExperimentalSession* AppBridge::session() {
    return session_;
}

StudyBridge* AppBridge::studyBridge() {
    return study_bridge_;
}

SettingsBridge* AppBridge::settingsBridge() {
    return settings_bridge_;
}

OptimizerBridge* AppBridge::optimizerBridge() {
    return optimizer_bridge_;
}

MlBridge* AppBridge::mlBridge() {
    return ml_bridge_;
}

PoseBridge* AppBridge::poseBridge() {
    return pose_bridge_;
}
