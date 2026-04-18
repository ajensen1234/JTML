// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#ifndef SCENE_CONTROLLER_H
#define SCENE_CONTROLLER_H

#pragma once

#include <QObject>
#include <QString>
#include <QVector>

#include <vector>

#include "core/session_context.h"

class Viewer;
class vtkActor;

namespace jta_gui {

enum class ModelOpacityMode {
    Original,
    Solid,
    Transparent,
    Wireframe,
};

struct SelectionSyncState {
    int frame_index = -1;
    int previous_frame_index = -1;
    bool camera_a_selected = true;
    bool currently_optimizing = false;
    bool calibrated_for_biplane_viewport = false;
    bool actor_text_visible = false;
    ModelOpacityMode opacity_mode = ModelOpacityMode::Original;
    std::vector<int> selected_model_indices;
};

class SceneController : public QObject {
    Q_OBJECT

public:
    SceneController(
        Viewer& primary_viewer,
        Viewer& coronal_viewer,
        jta_core::SessionContext& session,
        QObject* parent = nullptr);

    bool OnCameraASelected(const SelectionSyncState& state);
    bool OnCameraBSelected(const SelectionSyncState& state);
    bool OnImageSelectionChanged(const SelectionSyncState& state);
    bool OnModelSelectionChanged(const SelectionSyncState& state);
    bool OnMakePrincipalActor(
        vtkActor* new_principal_actor,
        const SelectionSyncState& state,
        QString& error_message);

    void SyncSelectedModelOpacity(
        const std::vector<int>& selected_model_indices,
        ModelOpacityMode opacity_mode);

Q_SIGNALS:
    void requestThresholdControlSync(int aperture, int low_threshold, int high_threshold);
    void requestCameraButtonSync(bool disable_camera_a, bool disable_camera_b);
    void requestInteractorCameraMode(bool camera_b_mode);
    void requestPrincipalSelectionOrder(QVector<int> reordered_indices);
    void requestUiRefresh();

private:
    [[nodiscard]] bool FrameIndexValid(int frame_index, bool camera_a_selected) const;
    [[nodiscard]] bool ModelIndexValid(int model_index) const;
    [[nodiscard]] int ResolveModelIndex(vtkActor* actor) const;

    static double CalculateViewingAngle(
        const jta_core::SessionContext& session,
        int height,
        bool camera_a_selected);

    void SaveCurrentPoseToSession(
        int frame_index,
        const std::vector<int>& selected_model_indices,
        bool source_camera_b);
    void ApplyImagePlacementAndViewAngle(int frame_index, bool camera_a_selected);
    void ApplyModelPoseFromSession(
        int frame_index,
        int model_index,
        bool camera_a_selected);
    void UpdateActorText(
        int frame_index,
        int model_index,
        bool camera_a_selected,
        bool actor_text_visible);

    void SyncSelectionPosesAndText(const SelectionSyncState& state);
    void SyncSelectionColors(const std::vector<int>& selected_model_indices);

    void ApplyOpacityToModel(int model_index, ModelOpacityMode opacity_mode);
    void ApplySelectionOpacity(
        const std::vector<int>& selected_model_indices,
        ModelOpacityMode opacity_mode);
    void ApplyPrincipalOpacity(int model_index, ModelOpacityMode opacity_mode);

    void RenderAndNotifyUi();

    Viewer& primary_viewer_;
    Viewer& coronal_viewer_;
    jta_core::SessionContext& session_;
};

} // namespace jta_gui

#endif /* SCENE_CONTROLLER_H */
