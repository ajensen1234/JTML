// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U3: QmlVtkRenderer — the render seam (R7/R11). The minimal VTK
// pipeline under QQuickVTKItem's render-thread contract: models at pose over
// the fluoro background, pose updates via dispatch_async.
//
// Render-thread contract (QQuickVTKItem.h + QQuickVTKItem.cxx, VTK 9.3,
// review-verified):
//  - ALL VTK objects are created in initializeVTK(), stored in the returned
//    vtkUserData, and reachable ONLY from initializeVTK / destroyingVTK /
//    dispatch_async bodies. The item never touches VTK state anywhere else.
//  - initializeVTK/destroyingVTK run inside QQuickVTKItem::updatePaintNode
//    (Qt Quick render thread) with the GUI thread blocked at the scene-graph
//    sync point — reading the app-thread-owned mirror members is safe there.
//  - dispatch_async() lambdas ALSO run on the Qt Quick render thread (the
//    queue is drained in updatePaintNode). The GUI-thread slots therefore
//    copy scene state to locals and capture BY VALUE into the lambda — the
//    lambda never reads app-owned mutable state (no `this`, no scene
//    pointer, no member reads).
//  - The scene-graph can delete the underlying node at any moment (window
//    teardown, item removal), in which case initializeVTK runs again with a
//    fresh render window: initializeVTK rebuilds the whole pipeline from the
//    mirror copy kept on the app thread.
//
// Pipeline mirror (widgets Viewer, unchanged there):
//  - background: vtkImageImport -> vtkImageData -> vtkDataSetMapper ->
//    vtkActor (Viewer::initialize_vtk_mappers + update_display_background),
//    non-pickable, added to the layer-0 background renderer;
//  - models: vtkSTLReader -> vtkPolyDataMapper -> vtkActor
//    (Viewer::load_3d_models_into_actor_and_mapper_list) in the layer-1
//    scene renderer, pose applied as SetPosition/SetOrientation (the
//    widgets set_model_position/orientation_at_index);
//  - camera: background renderer = parallel, position (0,0,0), focal
//    (0,0,-1), scale 0.5*image height, clipping (0.1, 2*fy)
//    (Viewer::setup_camera_calibration + place_image_actors_according_to_
//    calibration); scene renderer = perspective, position (0,0,0), focal
//    (0,0,-1), view angle from the scene (mainscreen CalculateViewingAngle
//    output), clipping (0.1*fy, 1.75*fy).

#pragma once

#include <QQuickVTKItem.h>
#include <vtkSmartPointer.h>

#include <vector>

#include "ExperimentalScene.h"
#include "domain/data_structures_6D.h"

class QmlVtkRenderer : public QQuickVTKItem {
    Q_OBJECT

    // U3 debugging surface: the last pose applied to model 0, formatted on
    // the GUI thread in the update slots (main.qml binds a small readout).
    Q_PROPERTY(QString poseReadout READ poseReadout NOTIFY sceneChanged)

    // Interaction mode (plan 005 feedback): CameraMode = trackball camera
    // (rotates the camera about the focal point, which we pin at the primary
    // model so the view pivots around the model); ModelMode = rotate the
    // primary model about its own center (the widgets app's trackball-actor
    // mode, without the picking dependency — the primary model is rotated
    // directly, sidestepping the QQuickVTKItem pick-position bug tail).
    Q_PROPERTY(int interactionMode READ interactionMode WRITE setInteractionMode
                   NOTIFY interactionModeChanged)

public:
    enum InteractionMode { CameraMode = 0, ModelMode = 1 };
    Q_ENUM(InteractionMode)

    explicit QmlVtkRenderer(QQuickItem* parent = nullptr);

    // --- Render-thread contract (see file comment) ----------------------
    vtkUserData initializeVTK(vtkRenderWindow* renderWindow) override;
    void destroyingVTK(vtkRenderWindow* renderWindow, vtkUserData userData)
        override;

    // --- App-thread (GUI) entry points ----------------------------------
    // Binds the app-owned scene. Non-owning: the scene must outlive this
    // item (it is the app-owned state that makes dispatch_async after
    // destruction safe). Call before the first update slot.
    void setScene(ExperimentalScene* scene);
    ExperimentalScene* scene() const;

    // GUI-thread slots: copy the relevant scene state to locals, refresh
    // the app-thread mirror (for initializeVTK re-runs), and dispatch a
    // by-value lambda to the Qt Quick render thread. Safe to call at any
    // time; a dispatch queued when the item is destroyed is dropped with
    // the item (its captures are app-thread-owned values).
    Q_INVOKABLE void
    applyScene();  // full resync (background + models + camera)
    Q_INVOKABLE void updateBackground();  // frame image + display mode
    Q_INVOKABLE void updatePose(int modelIndex);
    Q_INVOKABLE void updateModels();
    Q_INVOKABLE void updateCamera();

    // Which scene model the model-centric interactor moves (owner feedback
    // 2026-08-11): the movable actor follows the session's PRIMARY model
    // selection instead of being pinned to scene model 0. Negative/out-of-
    // range indices clear the implicit pick (nothing movable). Applied on
    // the render thread via dispatch_async.
    Q_INVOKABLE void setActiveModel(int sceneIndex);

    // Interaction mode switch (CameraMode / ModelMode). Applied on the
    // render thread via dispatch_async (the interactor is render-thread
    // owned — created by QQuickVTKItem's own initializeVTK wrapper).
    Q_INVOKABLE void setInteractionMode(int mode);
    int interactionMode() const;

    // Model-centric interaction pose sync (plan-005 feedback #2): after an
    // EndInteractionEvent on the model style, the render-thread observer
    // reads the primary actor's transform and reports it here; the signal is
    // emitted with by-value data (AutoConnection queues delivery to
    // GUI-thread receivers — the app writes the pose into LocationStorage +
    // the scene, so the optimizer starts from the visually arranged pose).
    void reportModelPoseAdjusted(
        int sceneModelIndex,
        double x,
        double y,
        double z,
        double xa,
        double ya,
        double za);

    QString poseReadout() const;

signals:
    void sceneChanged();
    void interactionModeChanged();
    // (sceneModelIndex, x, y, z, xa, ya, za) — emitted on the GUI thread
    // after a model-centric drag ends.
    void modelPoseAdjusted(
        int sceneModelIndex,
        double x,
        double y,
        double z,
        double xa,
        double ya,
        double za);

private:
    void copySceneMirror();
    void refreshPoseReadout();

    // App-thread-owned mirror of the bound scene. Written only in the slots
    // above (GUI thread); read by initializeVTK at the scene-graph sync
    // point and by the by-value captures of the dispatch lambdas.
    ExperimentalScene scene_mirror_;
    ExperimentalScene* bound_scene_ = nullptr;
    QString pose_readout_;
    // Default = Model mode: the owner's workflow is "line up the model, let
    // the optimizer refine" (plan-005 feedback) — the drag rotates the MODEL
    // and the pose syncs live. Camera mode is the secondary view-orbit mode.
    int interaction_mode_ = ModelMode;
};
