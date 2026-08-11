// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U3: ExperimentalScene — the app-thread-owned scene state for the QML
// render seam (R7/R11).
//
// Threading contract (review-verified against the pinned VTK 9.3 source):
// this is PLAIN DATA owned by the app (GUI) thread. The renderer
// (QmlVtkRenderer) never reads it from render-thread code: its GUI-thread
// slots copy what they need to locals and dispatch_async lambdas capture by
// value. Mutex-free by design — only the app thread writes, and every read
// happens either on the app thread or at the scene-graph sync point (GUI
// thread blocked inside QQuickVTKItem::updatePaintNode).
//
// Semantics mirror the widgets render pipeline (Viewer + mainscreen):
//  - the background image is the current frame's ORIGINAL image; the display
//    mode (Original/Inverted) is applied at render time (the widgets
//    update_display_background_to_* family);
//  - model poses are Point6D with widgets semantics: translations in mm
//    relative to the image center, orientations in degrees;
//  - camera: view angle (mainscreen CalculateViewingAngle output) + focal
//    length in px (Viewer setup_camera_calibration consumes fy). Later units
//    derive both from the loaded Calibration; U3 keeps them as plain scene
//    state with sane defaults (Kneel_1's fy = 1198).

#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "domain/data_structures_6D.h"

// Background display modes (U3: original/inverted; edge/dilation extend this
// enum when the ML view lands).
enum class BackgroundMode { Original = 0, Inverted = 1 };

// One model in the scene: an STL path (loaded on the render thread through
// the existing services Model path) + its pose.
struct SceneModel {
    std::string path;
    std::string name;
    Point6D pose;
};

class ExperimentalScene {
public:
    // --- Background -----------------------------------------------------
    // Stores a refcounted copy of the frame's ORIGINAL image. The caller
    // must not mutate the buffer in place after handing it over (swap in a
    // new Mat instead) — the render thread imports the buffer directly.
    void setBackgroundImage(const cv::Mat& image);
    cv::Mat backgroundImage() const;

    void setBackgroundMode(BackgroundMode mode);
    BackgroundMode backgroundMode() const;

    // --- Models ---------------------------------------------------------
    void setModels(const std::vector<SceneModel>& models);
    std::vector<SceneModel> models() const;
    void setModelPose(int index, const Point6D& pose);

    // --- Camera (widgets semantics) -------------------------------------
    void setCameraViewAngle(double degrees);  // scene renderer view angle
    double cameraViewAngle() const;
    void setFocalLengthPx(double px);  // fy: clipping ranges + image plane Z
    double focalLengthPx() const;

private:
    cv::Mat background_image_;
    BackgroundMode background_mode_ = BackgroundMode::Original;
    std::vector<SceneModel> models_;

    double camera_view_angle_ = 10.0;  // degrees (CalculateViewingAngle)
    double focal_length_px_ = 1198.0;  // Kneel_1 calibration fy
};
