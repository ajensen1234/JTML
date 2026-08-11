// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "AppBridge.h"

#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "StudyBridge.h"

AppBridge::AppBridge(ExperimentalScene* scene, QObject* parent)
    : QObject(parent) {
    /*U4: the hub owns the app-owned dataset (R3) + the study-load adapter.
     * The list models are created inside StudyBridge (direct-compiled).*/
    session_ = new ExperimentalSession;
    study_bridge_ = new StudyBridge(this, session_, scene, this);
}

AppBridge::~AppBridge() {
    /*study_bridge_ is parented to this and dies with the QObject chain; the
     * dataset is a plain struct.*/
    delete session_;
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
