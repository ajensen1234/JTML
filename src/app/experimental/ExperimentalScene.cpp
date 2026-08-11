// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U3: ExperimentalScene — plain-data implementation (see the header for
// the threading contract). No logic beyond store/copy: all behavior lives in
// the renderer (QmlVtkRenderer) and, later, the seams that populate the
// scene (StudyBridge U4, PoseBridge U8).

#include "ExperimentalScene.h"

void ExperimentalScene::setBackgroundImage(const cv::Mat& image) {
    background_image_ = image;  // refcounted shallow copy
}

cv::Mat ExperimentalScene::backgroundImage() const {
    return background_image_;
}

void ExperimentalScene::setBackgroundMode(BackgroundMode mode) {
    background_mode_ = mode;
}

BackgroundMode ExperimentalScene::backgroundMode() const {
    return background_mode_;
}

void ExperimentalScene::setModels(const std::vector<SceneModel>& models) {
    models_ = models;
}

std::vector<SceneModel> ExperimentalScene::models() const {
    return models_;
}

void ExperimentalScene::setModelPose(int index, const Point6D& pose) {
    if (index < 0 || index >= static_cast<int>(models_.size())) {
        return;
    }
    models_[static_cast<size_t>(index)].pose = pose;
}

void ExperimentalScene::setCameraViewAngle(double degrees) {
    camera_view_angle_ = degrees;
}

double ExperimentalScene::cameraViewAngle() const {
    return camera_view_angle_;
}

void ExperimentalScene::setFocalLengthPx(double px) {
    focal_length_px_ = px;
}

double ExperimentalScene::focalLengthPx() const {
    return focal_length_px_;
}
