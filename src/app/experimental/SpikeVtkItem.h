// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U1 spike: QQuickVTKItem subclass for the QML+VTK viability gate.
//
// Render-thread contract (QQuickVTKItem.h, VTK 9.3):
//  - ALL VTK objects are owned by and run on the QML render thread. They are
//    created in initializeVTK(), stored in the returned vtkUserData, and
//    reachable ONLY from initializeVTK / destroyingVTK / dispatch_async
//    lambdas. The GUI thread is blocked during initializeVTK/destroyingVTK.
//  - dispatch_async() lambdas run on the Qt Quick RENDER thread (the queue is
//    drained in QQuickVTKItem::updatePaintNode). The app thread must copy any
//    state it wants the lambda to see and capture it BY VALUE.
//
// This item: loads example_studies/Kneel_1/KR_right_7_fem.stl through the
// existing services path (Model wraps vtkSTLReader / stl_reader), renders the
// silhouette, and exposes GUI-thread entry points (setActorOffset,
// sampleCamera) that follow the dispatch_async discipline.

#pragma once

#include <QQuickVTKItem.h>
#include <vtkSmartPointer.h>

#include <atomic>
#include <string>

class SpikeVtkItem : public QQuickVTKItem {
    Q_OBJECT

public:
    explicit SpikeVtkItem(QQuickItem* parent = nullptr);

    // Render-thread contract: creates the whole pipeline and returns the
    // vtkUserData that owns every VTK object (owned by the QML render thread).
    vtkUserData initializeVTK(vtkRenderWindow* renderWindow) override;

    // GUI-thread entry point for the dynamic-update leg: copies the offset
    // and dispatches to the render thread (the lambda runs in
    // QQuickVTKItem::updatePaintNode on the Qt Quick render thread).
    Q_INVOKABLE void setActorOffset(double x, double y, double z);

    // GUI-thread entry point that reads the camera position on the render
    // thread into an atomic, for the smoke's interaction leg.
    Q_INVOKABLE void sampleCamera();

    std::string stlPath() const { return stl_path_; }
    double lastCameraPositionX() const { return camera_pos_x_.load(); }

private:
    static std::string ResolveFemStlPath();

    std::string stl_path_;
    std::atomic<double> camera_pos_x_{0.0};
};
