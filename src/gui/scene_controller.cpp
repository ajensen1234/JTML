// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

#include "gui/scene_controller.h"

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkRenderer.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>

#include "gui/viewer.h"

namespace {

std::string BuildInfoTextFromPose(const Point6D& pose) {
    std::string info_text = "Location: <";
    info_text += std::to_string(static_cast<long double>(pose.x)) + "," +
                 std::to_string(static_cast<long double>(pose.y)) + "," +
                 std::to_string(static_cast<long double>(pose.z)) +
                 ">\nOrientation: <" +
                 std::to_string(static_cast<long double>(pose.xa)) + "," +
                 std::to_string(static_cast<long double>(pose.ya)) + "," +
                 std::to_string(static_cast<long double>(pose.za)) + ">";
    return info_text;
}

} // namespace

namespace jta_gui {

SceneController::SceneController(
    Viewer& primary_viewer,
    Viewer& coronal_viewer,
    jta_core::SessionContext& session,
    QObject* parent) :
    QObject(parent),
    primary_viewer_(primary_viewer),
    coronal_viewer_(coronal_viewer),
    session_(session) {}

bool SceneController::OnCameraASelected(const SelectionSyncState& state) {
    if (!FrameIndexValid(state.frame_index, true)) {
        return false;
    }

    emit requestInteractorCameraMode(false);

    if (state.calibrated_for_biplane_viewport) {
        emit requestCameraButtonSync(true, false);
        if (!state.currently_optimizing && state.previous_frame_index >= 0) {
            SaveCurrentPoseToSession(
                state.previous_frame_index,
                state.selected_model_indices,
                true);
        }
    }

    const Frame& frame = session_.loaded_frames[state.frame_index];
    emit requestThresholdControlSync(
        frame.GetAperture(), frame.GetLowThreshold(), frame.GetHighThreshold());

    ApplyImagePlacementAndViewAngle(state.frame_index, true);

    if (state.currently_optimizing && state.calibrated_for_biplane_viewport) {
        for (const int model_index : state.selected_model_indices) {
            if (!ModelIndexValid(model_index)) {
                continue;
            }

            const double* position =
                primary_viewer_.get_model_position_at_index(model_index);
            const double* orientation =
                primary_viewer_.get_model_orientation_at_index(model_index);
            Point6D current_pose(
                position[0],
                position[1],
                position[2],
                orientation[0],
                orientation[1],
                orientation[2]);
            current_pose =
                session_.calibration_file_.convert_Pose_B_to_Pose_A(current_pose);

            primary_viewer_.set_model_position_at_index(
                model_index, current_pose.x, current_pose.y, current_pose.z);
            primary_viewer_.set_model_orientation_at_index(
                model_index, current_pose.xa, current_pose.ya, current_pose.za);
            coronal_viewer_.set_model_position_at_index(
                model_index, current_pose.x, current_pose.y, current_pose.z);
            coronal_viewer_.set_model_orientation_at_index(
                model_index, current_pose.xa, current_pose.ya, current_pose.za);

            UpdateActorText(
                state.frame_index,
                model_index,
                true,
                state.actor_text_visible);
        }
    } else {
        SyncSelectionPosesAndText(state);
    }

    RenderAndNotifyUi();
    return true;
}

bool SceneController::OnCameraBSelected(const SelectionSyncState& state) {
    if (!FrameIndexValid(state.frame_index, false)) {
        return false;
    }

    emit requestInteractorCameraMode(true);
    emit requestCameraButtonSync(false, true);

    if (!state.currently_optimizing && state.previous_frame_index >= 0) {
        SaveCurrentPoseToSession(
            state.previous_frame_index,
            state.selected_model_indices,
            false);
    }

    const Frame& frame = session_.loaded_frames_B[state.frame_index];
    emit requestThresholdControlSync(
        frame.GetAperture(), frame.GetLowThreshold(), frame.GetHighThreshold());

    ApplyImagePlacementAndViewAngle(state.frame_index, false);

    if (state.currently_optimizing && state.calibrated_for_biplane_viewport) {
        for (const int model_index : state.selected_model_indices) {
            if (!ModelIndexValid(model_index)) {
                continue;
            }

            const double* position =
                primary_viewer_.get_model_position_at_index(model_index);
            const double* orientation =
                primary_viewer_.get_model_orientation_at_index(model_index);
            Point6D current_pose(
                position[0],
                position[1],
                position[2],
                orientation[0],
                orientation[1],
                orientation[2]);
            current_pose =
                session_.calibration_file_.convert_Pose_A_to_Pose_B(current_pose);

            primary_viewer_.set_model_position_at_index(
                model_index, current_pose.x, current_pose.y, current_pose.z);
            primary_viewer_.set_model_orientation_at_index(
                model_index, current_pose.xa, current_pose.ya, current_pose.za);
            coronal_viewer_.set_model_position_at_index(
                model_index, current_pose.x, current_pose.y, current_pose.z);
            coronal_viewer_.set_model_orientation_at_index(
                model_index, current_pose.xa, current_pose.ya, current_pose.za);

            UpdateActorText(
                state.frame_index,
                model_index,
                false,
                state.actor_text_visible);
        }
    } else {
        SyncSelectionPosesAndText(state);
    }

    RenderAndNotifyUi();
    return true;
}

bool SceneController::OnImageSelectionChanged(const SelectionSyncState& state) {
    if (!FrameIndexValid(state.frame_index, state.camera_a_selected)) {
        return false;
    }

    if (state.camera_a_selected) {
        const Frame& frame = session_.loaded_frames[state.frame_index];
        emit requestThresholdControlSync(
            frame.GetAperture(), frame.GetLowThreshold(), frame.GetHighThreshold());
    } else {
        const Frame& frame = session_.loaded_frames_B[state.frame_index];
        emit requestThresholdControlSync(
            frame.GetAperture(), frame.GetLowThreshold(), frame.GetHighThreshold());
    }

    ApplyImagePlacementAndViewAngle(state.frame_index, state.camera_a_selected);

    if (state.selected_model_indices.empty()) {
        primary_viewer_.make_actor_text_invisible();
        RenderAndNotifyUi();
        return true;
    }

    primary_viewer_.make_actor_text_visible();
    ApplySelectionOpacity(state.selected_model_indices, state.opacity_mode);
    SyncSelectionPosesAndText(state);
    RenderAndNotifyUi();
    return true;
}

bool SceneController::OnModelSelectionChanged(const SelectionSyncState& state) {
    primary_viewer_.make_all_models_invisible();
    coronal_viewer_.make_all_models_invisible();

    if (state.selected_model_indices.empty()) {
        primary_viewer_.make_actor_text_invisible();
        RenderAndNotifyUi();
        return true;
    }

    primary_viewer_.make_actor_text_visible();
    SyncSelectionColors(state.selected_model_indices);
    ApplySelectionOpacity(state.selected_model_indices, state.opacity_mode);

    if (FrameIndexValid(state.frame_index, state.camera_a_selected)) {
        SyncSelectionPosesAndText(state);
    }

    RenderAndNotifyUi();
    return true;
}

bool SceneController::OnMakePrincipalActor(
    vtkActor* new_principal_actor,
    const SelectionSyncState& state,
    QString& error_message) {
    error_message.clear();

    const int principal_index = ResolveModelIndex(new_principal_actor);
    if (principal_index < 0) {
        error_message = "Couldn't find model index to make principal!";
        return false;
    }

    if (state.selected_model_indices.size() <= 1) {
        return true;
    }

    const auto principal_it = std::find(
        state.selected_model_indices.begin(),
        state.selected_model_indices.end(),
        principal_index);
    if (principal_it != state.selected_model_indices.end()) {
        QVector<int> reordered_indices;
        reordered_indices.reserve(
            static_cast<int>(state.selected_model_indices.size()));
        reordered_indices.push_back(principal_index);
        for (const int model_index : state.selected_model_indices) {
            if (model_index == principal_index) {
                continue;
            }
            reordered_indices.push_back(model_index);
        }
        emit requestPrincipalSelectionOrder(reordered_indices);
    }

    ApplyPrincipalOpacity(principal_index, state.opacity_mode);
    if (FrameIndexValid(state.frame_index, state.camera_a_selected)) {
        ApplyModelPoseFromSession(
            state.frame_index,
            principal_index,
            state.camera_a_selected);
        UpdateActorText(
            state.frame_index,
            principal_index,
            state.camera_a_selected,
            state.actor_text_visible);
    }

    RenderAndNotifyUi();
    return true;
}

void SceneController::SyncSelectedModelOpacity(
    const std::vector<int>& selected_model_indices,
    ModelOpacityMode opacity_mode) {
    ApplySelectionOpacity(selected_model_indices, opacity_mode);
    RenderAndNotifyUi();
}

bool SceneController::FrameIndexValid(int frame_index, bool camera_a_selected) const {
    if (frame_index < 0) {
        return false;
    }

    if (camera_a_selected) {
        return frame_index < static_cast<int>(session_.loaded_frames.size());
    }

    return frame_index < static_cast<int>(session_.loaded_frames_B.size());
}

bool SceneController::ModelIndexValid(int model_index) const {
    return model_index >= 0 &&
           model_index < static_cast<int>(session_.loaded_models.size());
}

int SceneController::ResolveModelIndex(vtkActor* actor) const {
    if (actor == nullptr) {
        return -1;
    }

    const int actor_count = primary_viewer_.model_actor_list_size();
    for (int index = 0; index < actor_count; ++index) {
        if (primary_viewer_.get_model_actor_at_index(index) == actor) {
            return index;
        }
    }

    return -1;
}

double SceneController::CalculateViewingAngle(
    const jta_core::SessionContext& session,
    int height,
    bool camera_a_selected) {
    constexpr double kRadiansToDegrees = 180.0 / std::numbers::pi_v<double>;

    if (camera_a_selected) {
        const double y =
            height * session.calibration_file_.camera_A_principal_.pixel_pitch_ /
                2.0 +
            std::abs(session.calibration_file_.camera_A_principal_.principal_y_);
        return kRadiansToDegrees * 2.0 *
               std::atan2(
                   y,
                   session.calibration_file_.camera_A_principal_
                       .principal_distance_);
    }

    const double y =
        height * session.calibration_file_.camera_B_principal_.pixel_pitch_ / 2.0 +
        std::abs(session.calibration_file_.camera_B_principal_.principal_y_);
    return kRadiansToDegrees * 2.0 *
           std::atan2(
               y, session.calibration_file_.camera_B_principal_.principal_distance_);
}

void SceneController::SaveCurrentPoseToSession(
    int frame_index,
    const std::vector<int>& selected_model_indices,
    bool source_camera_b) {
    if (frame_index < 0 ||
        frame_index >= static_cast<int>(session_.loaded_frames.size())) {
        return;
    }

    for (const int model_index : selected_model_indices) {
        if (!ModelIndexValid(model_index)) {
            continue;
        }

        const double* position =
            primary_viewer_.get_model_position_at_index(model_index);
        const double* orientation =
            primary_viewer_.get_model_orientation_at_index(model_index);
        Point6D pose(
            position[0],
            position[1],
            position[2],
            orientation[0],
            orientation[1],
            orientation[2]);

        if (source_camera_b) {
            pose = session_.calibration_file_.convert_Pose_B_to_Pose_A(pose);
        }

        session_.model_locations_.SavePose(frame_index, model_index, pose);
    }
}

void SceneController::ApplyImagePlacementAndViewAngle(
    int frame_index,
    bool camera_a_selected) {
    if (camera_a_selected) {
        Frame& frame = session_.loaded_frames[frame_index];
        primary_viewer_.place_image_actors_according_to_calibration(
            session_.calibration_file_,
            frame.GetOriginalImage().rows,
            frame.GetOriginalImage().cols);
        coronal_viewer_.place_image_actors_according_to_calibration(
            session_.calibration_file_,
            frame.GetOriginalImage().rows,
            frame.GetOriginalImage().cols);

        const double view_angle = CalculateViewingAngle(
            session_,
            frame.GetOriginalImage().rows,
            true);
        primary_viewer_.get_renderer()->GetActiveCamera()->SetViewAngle(view_angle);
        coronal_viewer_.get_renderer()->GetActiveCamera()->SetViewAngle(view_angle);
        return;
    }

    Frame& frame = session_.loaded_frames_B[frame_index];
    primary_viewer_.place_image_actors_according_to_calibration(
        session_.calibration_file_.camera_B_principal_,
        frame.GetOriginalImage().rows,
        frame.GetOriginalImage().cols);
    coronal_viewer_.place_image_actors_according_to_calibration(
        session_.calibration_file_.camera_B_principal_,
        frame.GetOriginalImage().rows,
        frame.GetOriginalImage().cols);

    const double view_angle = CalculateViewingAngle(
        session_,
        frame.GetOriginalImage().rows,
        false);
    primary_viewer_.get_renderer()->GetActiveCamera()->SetViewAngle(view_angle);
    coronal_viewer_.get_renderer()->GetActiveCamera()->SetViewAngle(view_angle);
}

void SceneController::ApplyModelPoseFromSession(
    int frame_index,
    int model_index,
    bool camera_a_selected) {
    if (!ModelIndexValid(model_index) ||
        frame_index < 0 ||
        frame_index >= static_cast<int>(session_.loaded_frames.size())) {
        return;
    }

    Point6D pose = session_.model_locations_.GetPose(frame_index, model_index);
    if (!camera_a_selected) {
        pose = session_.calibration_file_.convert_Pose_A_to_Pose_B(pose);
    }

    primary_viewer_.set_model_position_at_index(model_index, pose.x, pose.y, pose.z);
    primary_viewer_.set_model_orientation_at_index(
        model_index, pose.xa, pose.ya, pose.za);
    coronal_viewer_.set_model_position_at_index(model_index, pose.x, pose.y, pose.z);
    coronal_viewer_.set_model_orientation_at_index(
        model_index, pose.xa, pose.ya, pose.za);
}

void SceneController::UpdateActorText(
    int frame_index,
    int model_index,
    bool camera_a_selected,
    bool actor_text_visible) {
    if (!actor_text_visible || !ModelIndexValid(model_index)) {
        return;
    }

    if (camera_a_selected) {
        primary_viewer_.set_actor_text(
            primary_viewer_.print_location_and_orientation_of_model_at_index(
                model_index));
        primary_viewer_.set_actor_text_color_to_model_color_at_index(model_index);
        return;
    }

    if (frame_index < 0 ||
        frame_index >= static_cast<int>(session_.loaded_frames.size())) {
        return;
    }

    const Point6D pose = session_.model_locations_.GetPose(frame_index, model_index);
    primary_viewer_.set_actor_text(BuildInfoTextFromPose(pose));
    primary_viewer_.set_actor_text_color_to_model_color_at_index(model_index);
}

void SceneController::SyncSelectionPosesAndText(const SelectionSyncState& state) {
    for (const int model_index : state.selected_model_indices) {
        if (!ModelIndexValid(model_index)) {
            continue;
        }

        ApplyModelPoseFromSession(
            state.frame_index, model_index, state.camera_a_selected);
        UpdateActorText(
            state.frame_index,
            model_index,
            state.camera_a_selected,
            state.actor_text_visible);
    }
}

void SceneController::SyncSelectionColors(
    const std::vector<int>& selected_model_indices) {
    double uf_orange[3] = {255, 77, 0};
    double uf_blue[3] = {0, 72, 204};

    for (std::size_t index = 0; index < selected_model_indices.size(); ++index) {
        const int model_index = selected_model_indices[index];
        if (!ModelIndexValid(model_index)) {
            continue;
        }

        if (index == 0) {
            primary_viewer_.set_3d_model_color(model_index, uf_orange);
            coronal_viewer_.set_3d_model_color(model_index, uf_orange);
            continue;
        }

        primary_viewer_.set_3d_model_color(model_index, uf_blue);
        coronal_viewer_.set_3d_model_color(model_index, uf_blue);
    }
}

void SceneController::ApplyOpacityToModel(
    int model_index,
    ModelOpacityMode opacity_mode) {
    if (!ModelIndexValid(model_index)) {
        return;
    }

    switch (opacity_mode) {
        case ModelOpacityMode::Original:
            primary_viewer_.change_model_opacity_to_original(model_index);
            coronal_viewer_.change_model_opacity_to_original(model_index);
            break;
        case ModelOpacityMode::Solid:
            primary_viewer_.change_model_opacity_to_solid(model_index);
            coronal_viewer_.change_model_opacity_to_solid(model_index);
            break;
        case ModelOpacityMode::Transparent:
            primary_viewer_.change_model_opacity_to_transparent(model_index);
            coronal_viewer_.change_model_opacity_to_transparent(model_index);
            break;
        case ModelOpacityMode::Wireframe:
            primary_viewer_.change_model_opacity_to_wire_frame(model_index);
            coronal_viewer_.change_model_opacity_to_wire_frame(model_index);
            break;
    }
}

void SceneController::ApplySelectionOpacity(
    const std::vector<int>& selected_model_indices,
    ModelOpacityMode opacity_mode) {
    for (const int model_index : selected_model_indices) {
        ApplyOpacityToModel(model_index, opacity_mode);
    }
}

void SceneController::ApplyPrincipalOpacity(
    int model_index,
    ModelOpacityMode opacity_mode) {
    if (!ModelIndexValid(model_index)) {
        return;
    }

    switch (opacity_mode) {
        case ModelOpacityMode::Original:
            primary_viewer_.change_model_opacity_to_original(model_index);
            break;
        case ModelOpacityMode::Solid:
            primary_viewer_.change_model_opacity_to_solid(model_index);
            break;
        case ModelOpacityMode::Transparent:
            primary_viewer_.change_model_opacity_to_transparent(model_index);
            break;
        case ModelOpacityMode::Wireframe:
            primary_viewer_.change_model_opacity_to_wire_frame(model_index);
            break;
    }

    coronal_viewer_.change_model_opacity_to_wire_frame(model_index);
}

void SceneController::RenderAndNotifyUi() {
    primary_viewer_.render_scene();
    coronal_viewer_.render_scene();
    emit requestUiRefresh();
}

} // namespace jta_gui
