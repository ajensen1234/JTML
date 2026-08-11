// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "AppBridge.h"

#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "SettingsBridge.h"
#include "StudyBridge.h"
#include "services/settings_service.h"

AppBridge::AppBridge(
    ExperimentalScene* scene,
    jta::SettingsService* settings_service,
    QObject* parent) : QObject(parent) {
    /*U4: the hub owns the app-owned dataset (R3) + the study-load adapter.
     * The list models are created inside StudyBridge (direct-compiled).*/
    session_ = new ExperimentalSession;
    study_bridge_ = new StudyBridge(this, session_, scene, this);

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
