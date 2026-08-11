// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "AppBridge.h"

AppBridge::AppBridge(QObject* parent) : QObject(parent) {}

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
